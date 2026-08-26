"""M2-FINAL-DIAG Task 1: single first-conv 4-way comparison.

Build a minimal ONNX model with just the first Conv2D of the det
network (the one that maps 3-channel input to 16-channel stem
output). Use the EXACT same weight, bias, strides, padding. Run
on:
  - paddle.nn.functional.conv2d on CPU (FP32, no TF32, no IR opt)
  - paddle.nn.functional.conv2d on GPU (FP32, no TF32, no IR opt)
  - onnxruntime on CPU
  - onnxruntime on CUDA
  - MNN CPU (via the det model engine, but we just pull the
    op0001_Add.1 output which is conv + bias add; for the bare
    conv we'd need a separate MNN model)

If the bare conv 4-way match is exact (<1e-6), then the per-conv
kernel is fine and the issue is downstream (BN folding, runtime
graph optimization, etc.).

The actual isolated conv uses NCHW input [1, 3, 704, 1280], weight
shape [16, 3, 3, 3], bias shape [16], stride [1,1], padding
['SAME', 1, 1] (Paddle's same padding for kernel 3).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Make Paddle CPU first (and quiet)
os.environ.setdefault("GLOG_v", "0")

import paddle  # noqa: E402
import paddle.nn.functional as F  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

ROOT = Path("/root/pp-ocr-mnn")
WORK = Path("/tmp/m2fd")
WORK.mkdir(parents=True, exist_ok=True)


def setup_paddle(device: str):
    paddle.set_device(device)
    if device == "gpu":
        try:
            paddle.set_flags({
                "FLAGS_cudnn_deterministic": True,
                "FLAGS_cudnn_exhaustive_search": False,  # mutually exclusive with deterministic
                "FLAGS_use_cuda_aligned_allocator": False,
                "FLAGS_enable_cublas_tensor_op_math": False,  # disable TF32
            })
        except Exception as e:
            print(f"  paddle flag set: {e}")
    else:
        try:
            paddle.set_flags({
                "FLAGS_cudnn_deterministic": True,
                "FLAGS_cudnn_exhaustive_search": True,
            })
        except Exception:
            pass


def get_first_conv_weights():
    """Use Paddle Inference to read the first conv's filter and bias
    from the Paddle model. Returns (W, b) as float32 numpy."""
    import paddle.inference as paddle_infer
    cfg = paddle_infer.Config(
        "/root/ppocr_models/PP-OCRv6_tiny_det/inference.json",
        "/root/ppocr_models/PP-OCRv6_tiny_det/inference.pdiparams")
    cfg.disable_gpu()
    cfg.disable_mkldnn()
    cfg.disable_onednn()
    cfg.disable_glog_info()
    cfg.switch_ir_optim(False)
    cfg.set_optimization_level(0)
    pred = paddle_infer.create_predictor(cfg)
    # No Paddle 3.x API to enumerate ops directly; the next-on-list
    # approach: use the converted ONNX model to find the first
    # Conv2D's weight name, then read it from the .pdiparams file.
    return None  # see below


def get_first_conv_weights_via_onnx():
    """Find the first Conv in the ONNX model, get its weight name,
    then look it up in the ONNX initializers."""
    m = onnx.load("/tmp/m2fd/v6_tiny_det.onnx")
    g = m.graph
    for n in g.node:
        if n.op_type == "Conv":
            w_name = n.input[1]
            if len(n.input) > 2:
                b_name = n.input[2]
            else:
                b_name = None
            for init in g.initializer:
                if init.name == w_name:
                    W = onnx.numpy_helper.to_array(init).copy()
                if b_name and init.name == b_name:
                    b = onnx.numpy_helper.to_array(init).copy()
            return n, W, b if b_name else None
    return None, None, None


def make_single_conv_onnx(W, b, input_shape, strides, pads, dilations, groups, out_path):
    """Build a minimal ONNX with one Conv (and optional Add for bias)."""
    import onnx.helper as oh
    inp = oh.make_tensor_value_info("x", onnx.TensorProto.FLOAT, input_shape)
    out_shape = [input_shape[0], W.shape[0],
                 (input_shape[2] + pads[0] + pads[2] - W.shape[2]) // strides[0] + 1,
                 (input_shape[3] + pads[1] + pads[3] - W.shape[3]) // strides[1] + 1]
    out = oh.make_tensor_value_info("y", onnx.TensorProto.FLOAT, out_shape)
    W_init = oh.make_tensor("W", onnx.TensorProto.FLOAT, list(W.shape), W.flatten().tolist())
    nodes = [oh.make_node("Conv", ["x", "W"], ["conv_out"],
                          kernel_shape=list(W.shape[2:]),
                          strides=list(strides),
                          pads=list(pads),
                          dilations=list(dilations),
                          group=groups)]
    if b is not None:
        b_init = oh.make_tensor("b", onnx.TensorProto.FLOAT, list(b.shape), b.flatten().tolist())
        nodes.append(oh.make_node("Add", ["conv_out", "b"], ["y"]))
        initializers = [W_init, b_init]
    else:
        nodes[-1].output[0] = "y"
        initializers = [W_init]
    g = oh.make_graph(nodes, "single_conv", [inp], [out], initializer=initializers)
    m = oh.make_model(g, opset_imports=[oh.make_opsetid("", 13)])
    m.ir_version = 7
    onnx.save(m, str(out_path))


def diff_stats(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    d = np.abs(a - b)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "pct_gt_001": 100.0 * float((d > 0.001).mean()),
        "pct_gt_01": 100.0 * float((d > 0.01).mean()),
        "n": int(d.size),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/m2fd/single_conv.json")
    args = p.parse_args()

    # 1. Get the first conv weights from ONNX
    print("== extracting first conv weights from ONNX ==")
    conv_node, W, b = get_first_conv_weights_via_onnx()
    assert conv_node is not None, "no Conv found in ONNX"
    print(f"   first Conv: name={conv_node.name} inputs={list(conv_node.input)}")
    print(f"   attrs: kernel={list(conv_node.attribute[0].ints)} "
          f"strides={list(conv_node.attribute[1].ints)} "
          f"pads={list(conv_node.attribute[2].ints) if len(conv_node.attribute) > 2 else None} "
          f"dilations={list(conv_node.attribute[3].ints) if len(conv_node.attribute) > 3 else None}")
    print(f"   W shape: {W.shape}, dtype={W.dtype}, mean={W.mean():.6f}, std={W.std():.6f}")
    if b is not None:
        print(f"   b shape: {b.shape}, mean={b.mean():.6f}")

    # 2. Input
    x = np.load("/tmp/m2num/det_input_paddle.npy")  # CHW 3x704x1280
    x_nchw = x[np.newaxis, ...].astype(np.float32)  # NCHW
    print(f"   input shape: {x_nchw.shape}")

    # 3. Paddle CPU conv2d
    print("== paddle.nn.functional.conv2d CPU ==")
    setup_paddle("cpu")
    t0 = time.time()
    w = paddle.to_tensor(W).astype("float32")
    bias = paddle.to_tensor(b).astype("float32") if b is not None else None
    inp = paddle.to_tensor(x_nchw).astype("float32")
    # From the ONNX attrs: kernel=3x3, strides=2x2, pads=[1,1,1,1]
    y_cpu = F.conv2d(inp, w, bias=bias, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1)
    paddle.disable_static()
    y_cpu_np = y_cpu.numpy().copy()
    print(f"   shape: {y_cpu_np.shape}, mean={y_cpu_np.mean():.6f}, time={time.time()-t0:.2f}s")

    # 4. Paddle GPU conv2d
    y_gpu_np = None
    try:
        print("== paddle.nn.functional.conv2d GPU ==")
        setup_paddle("gpu")
        w = paddle.to_tensor(W).astype("float32")
        bias = paddle.to_tensor(b).astype("float32") if b is not None else None
        inp = paddle.to_tensor(x_nchw).astype("float32")
        t0 = time.time()
        y_gpu = F.conv2d(inp, w, bias=bias, stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1)
        paddle.device.synchronize()
        y_gpu_np = y_gpu.numpy().copy()
        print(f"   shape: {y_gpu_np.shape}, mean={y_gpu_np.mean():.6f}, time={time.time()-t0:.2f}s")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"   GPU failed: {e}")

    # 5. ONNX Runtime CPU
    print("== onnxruntime CPU ==")
    onnx_path = WORK / "single_conv.onnx"
    if not onnx_path.exists():
        make_single_conv_onnx(W, b,
                                input_shape=[1, 3, 704, 1280],
                                strides=[2, 2],
                                pads=[1, 1, 1, 1],  # top, left, bottom, right
                                dilations=[1, 1],
                                groups=1,
                                out_path=onnx_path)
    t0 = time.time()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    y_ort_cpu = sess.run(None, {"x": x_nchw.astype(np.float32)})[0]
    print(f"   shape: {y_ort_cpu.shape}, mean={y_ort_cpu.mean():.6f}, time={time.time()-t0:.2f}s")

    # 6. ONNX Runtime CUDA
    y_ort_cuda = None
    try:
        print("== onnxruntime CUDA ==")
        t0 = time.time()
        sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
        print(f"   active providers: {sess.get_providers()}")
        y_ort_cuda = sess.run(None, {"x": x_nchw.astype(np.float32)})[0]
        print(f"   shape: {y_ort_cuda.shape}, mean={y_ort_cuda.mean():.6f}, time={time.time()-t0:.2f}s")
    except Exception as e:
        print(f"   ORT CUDA failed: {e}")

    # Save all
    np.save(WORK / "single_conv_W.npy", W)
    if b is not None: np.save(WORK / "single_conv_b.npy", b)
    np.save(WORK / "single_conv_x.npy", x_nchw)
    np.save(WORK / "single_conv_y_paddle_cpu.npy", y_cpu_np)
    if y_gpu_np is not None: np.save(WORK / "single_conv_y_paddle_gpu.npy", y_gpu_np)
    np.save(WORK / "single_conv_y_ort_cpu.npy", y_ort_cpu)
    if y_ort_cuda is not None: np.save(WORK / "single_conv_y_ort_cuda.npy", y_ort_cuda)

    # 7. Diff table
    print()
    print("== Diff table (single first conv) ==")
    print(f"   reference: paddle CPU (FP32, no TF32)")
    results = {}
    for label, arr in [("paddle_GPU", y_gpu_np),
                       ("ort_CPU", y_ort_cpu),
                       ("ort_CUDA", y_ort_cuda)]:
        if arr is None: continue
        s = diff_stats(y_cpu_np, arr)
        results[label] = s
        print(f"   vs {label:10s}: max={s['max_abs']:.6e} mean={s['mean_abs']:.6e} "
              f"%>0.001={s['pct_gt_001']:.4f}% %>0.01={s['pct_gt_01']:.4f}%")
    # also: self-compare
    s = diff_stats(y_cpu_np, y_cpu_np)
    print(f"   self (CPU vs CPU): max={s['max_abs']:.2e}")

    Path(args.out).write_text(json.dumps({
        "weight_shape": list(W.shape),
        "input_shape": list(x_nchw.shape),
        "diff": results,
    }, indent=2, default=str))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
