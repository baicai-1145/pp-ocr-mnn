"""M2-EXPORT-SWEEP part 2: Paddle inference direct API.

Run the det model through paddle.inference directly (no PaddleX
pipeline), with various optimization toggles, and diff the
output against the PaddleX pipeline reference
(/tmp/m2num/det_output_paddle.npy).

Combos:
  - ir_optim: True/False
  - enable_mkldnn: True/False
  - use_gpu: True/False (use_gpu=True also tries tensorrt)
  - enable_new_executor: True/False
  - precision: int (Precision::kFloat32) or Precision::kHalf

For each combo: load model, set input from /tmp/m2num/det_input_paddle.npy,
run, get output, diff.

The PaddleX reference output IS the "ground truth" for our pipeline
(M2-MATRIX baseline). The question: does Paddle inference (with
optimizations OFF) match the pipeline output? If yes, optimizations
are NOT the diff source. If no, optimizations cause the diff.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_v", "0")

import numpy as np
import paddle
import paddle.inference as paddle_infer

PADDLE_REF = Path("/tmp/m2num/det_output_paddle.npy")
INPUT_NPY   = Path("/tmp/m2num/det_input_paddle.npy")
DET_DIR     = "/root/ppocr_models/PP-OCRv6_tiny_det"


def run_paddle_inference(ir_optim, enable_mkldnn, use_gpu, enable_new_executor,
                          use_tensorrt=False, precision="float32"):
    cfg = paddle_infer.Config(
        os.path.join(DET_DIR, "inference.json"),
        os.path.join(DET_DIR, "inference.pdiparams"))
    cfg.disable_glog_info()
    cfg.set_optimization_level(0 if not ir_optim else 3)
    cfg.switch_ir_optim(ir_optim)
    if enable_mkldnn:
        try:
            cfg.enable_mkldnn()
        except Exception:
            pass
    else:
        try:
            cfg.disable_mkldnn()
            cfg.disable_onednn()
        except Exception:
            pass
    if use_gpu:
        cfg.enable_use_gpu(100, 0)
        if use_tensorrt:
            try:
                cfg.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    max_batch_size=1,
                    min_subgraph_size=3,
                    precision_mode=paddle_infer.PrecisionType.Float32,
                    use_static=False,
                    use_calib_mode=False)
            except Exception as e:
                pass
    else:
        cfg.disable_gpu()
    if enable_new_executor:
        try:
            cfg.enable_new_executor()
        except Exception:
            pass
    else:
        try:
            cfg.disable_new_executor()
        except Exception:
            pass

    pred = paddle_infer.create_predictor(cfg)
    inp_name = pred.get_input_names()[0]
    inp = pred.get_input_handle(inp_name)
    x = np.load(INPUT_NPY)[np.newaxis, ...].astype(np.float32)
    inp.reshape(x.shape)
    inp.copy_from_cpu(x)
    pred.run()
    out_name = pred.get_output_names()[0]
    out = pred.get_output_handle(out_name)
    y = out.copy_to_cpu()
    return np.asarray(y)


def diff_stats(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    if a.size != b.size:
        return {"max_abs": float("nan"), "mean_abs": float("nan"),
                "size_mismatch": True, "n_a": int(a.size), "n_b": int(b.size)}
    d = np.abs(a - b)
    return {"max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "pct_gt_001": 100.0 * float((d > 0.001).mean()),
            "pct_gt_01": 100.0 * float((d > 0.01).mean()),
            "n": int(d.size)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/m2pdi/paddle_inference_sweep.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    paddle_ref = np.load(PADDLE_REF)
    print(f"PaddleX ref: shape={paddle_ref.shape}, mean={paddle_ref.mean():.4f}")
    print(f"Input: shape={(1,) + np.load(INPUT_NPY).shape}")

    combos = [
        # (ir_optim, enable_mkldnn, use_gpu, enable_new_executor, use_tensorrt)
        # CPU baseline
        (True,  True,  False, True,  False),  # default-ish CPU
        (True,  False, False, True,  False),
        (False, False, False, True,  False),  # ir_optim OFF
        (False, False, False, False, False),  # ir_optim OFF + new_executor OFF
        (True,  True,  False, False, False),  # new_executor OFF
        # GPU
        (True,  False, True,  True,  False),  # GPU w/ ir_optim
        (False, False, True,  True,  False),  # GPU ir_optim OFF
        (False, False, True,  False, False),  # GPU all off
        # TensorRT
        (True,  False, True,  True,  True),
        (False, False, True,  True,  True),
    ]

    results = []
    for i, (ir_optim, mkldnn, gpu, new_ex, trt) in enumerate(combos):
        tag = (f"ir={int(ir_optim)}_mkldnn={int(mkldnn)}_gpu={int(gpu)}_"
               f"nex={int(new_ex)}_trt={int(trt)}")
        print(f"  [{i+1}/{len(combos)}] {tag}")
        try:
            t0 = time.time()
            y = run_paddle_inference(ir_optim, mkldnn, gpu, new_ex, trt)
            dt = time.time() - t0
            d = diff_stats(y, paddle_ref)
            print(f"    vs PaddleX: max={d['max_abs']:.4f} mean={d['mean_abs']:.4e} %>0.01={d['pct_gt_01']:.4f}% (time {dt:.2f}s)")
            results.append({
                "tag": tag,
                "combo": dict(ir_optim=ir_optim, mkldnn=mkldnn, gpu=gpu,
                              new_executor=new_ex, tensorrt=trt),
                "diff": d, "time_s": dt,
                "y_shape": list(y.shape),
            })
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            results.append({
                "tag": tag,
                "combo": dict(ir_optim=ir_optim, mkldnn=mkldnn, gpu=gpu,
                              new_executor=new_ex, tensorrt=trt),
                "error": str(e)[:300],
            })

    Path(args.out).write_text(json.dumps({
        "n_combos": len(combos),
        "results": results,
    }, indent=2, default=str))
    print(f"\nreport -> {args.out}")

    print()
    print("=== Summary (sorted by mean diff to PaddleX) ===")
    valid = [r for r in results if "diff" in r]
    valid.sort(key=lambda r: r["diff"]["mean_abs"])
    print(f"{'rank':>4s}  {'combo':50s}  {'max':>10s}  {'mean':>10s}  {'%>0.01':>7s}")
    for i, r in enumerate(valid):
        d = r["diff"]
        print(f"  {i:2d}  {r['tag'][:49]:50s}  {d['max_abs']:>10.4f}  {d['mean_abs']:>10.4e}  {d['pct_gt_01']:>6.2f}%")


if __name__ == "__main__":
    sys.exit(main())
