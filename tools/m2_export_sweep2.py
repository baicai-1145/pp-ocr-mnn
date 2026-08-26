"""M2-EXPORT-SWEEP v2: paddle2onnx parameter sweep, focused on the
flags that v1 missed: --enable_dist_prim_all (operator decomposition),
plus --optimize_tool=polygraphy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_v", "0")

import numpy as np
import onnx
import onnxruntime as ort

OUT = Path("/tmp/m2es2")
OUT.mkdir(parents=True, exist_ok=True)

DET_DIR  = "/root/ppocr_models/PP-OCRv6_tiny_det"
PADDLE_REF = Path("/tmp/m2num/det_output_paddle.npy")
INPUT_NPY   = Path("/tmp/m2num/det_input_paddle.npy")


def export_paddle_to_onnx(model_dir, model_filename, params_filename,
                          save_file, opset_version,
                          optimize_tool="onnxoptimizer",
                          enable_dist_prim_all=False,
                          enable_auto_update_opset=True,
                          enable_onnx_checker=True):
    cmd = ["paddle2onnx",
           "--model_dir", model_dir,
           "--model_filename", model_filename,
           "--params_filename", params_filename,
           "--save_file", save_file,
           "--opset_version", str(opset_version),
           "--optimize_tool", str(optimize_tool),
           "--enable_dist_prim_all", str(enable_dist_prim_all),
           "--enable_auto_update_opset", str(enable_auto_update_opset),
           "--enable_onnx_checker", str(enable_onnx_checker)]
    t0 = time.time()
    rc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    if rc.returncode != 0:
        return False, dt, (rc.stdout + rc.stderr)[-1500:]
    return True, dt, ""


def run_ort(onnx_path, input_data, providers=None):
    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    out = sess.run([out_name], {inp_name: input_data})[0]
    return out


def diff_stats(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    if a.size != b.size:
        return {"max_abs": float("nan"), "mean_abs": float("nan"),
                "size_mismatch": True}
    d = np.abs(a - b)
    return {"max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "pct_gt_001": 100.0 * float((d > 0.001).mean()),
            "pct_gt_01": 100.0 * float((d > 0.01).mean()),
            "n": int(d.size)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT / "sweep2.json"))
    args = p.parse_args()

    paddle_ref = np.load(PADDLE_REF)
    x = np.load(INPUT_NPY)[np.newaxis, ...].astype(np.float32)
    print(f"Paddle ref: shape={paddle_ref.shape}, mean={paddle_ref.mean():.4f}")
    print(f"Input:      shape={x.shape}")

    combos = []
    # operator decomposition ON
    for ov in [9, 11, 13, 15, 17]:
        for opt in ["onnxoptimizer", "None"]:
            combos.append({
                "opset_version": ov,
                "optimize_tool": opt,
                "enable_dist_prim_all": True,  # KEY: decompose combined ops
                "enable_auto_update_opset": True,
                "enable_onnx_checker": True,
                "tag": f"ov{ov}_{opt}_decomp",
            })
    # polygraphy
    for ov in [13, 15]:
        combos.append({
            "opset_version": ov,
            "optimize_tool": "polygraphy",
            "enable_dist_prim_all": False,
            "enable_auto_update_opset": True,
            "enable_onnx_checker": True,
            "tag": f"ov{ov}_polygraphy",
        })

    print(f"Running {len(combos)} export combos...")
    results = []
    for i, c in enumerate(combos):
        tag = c["tag"]
        onnx_path = OUT / f"v6_tiny_det_{tag}.onnx"
        ok, dt, err = export_paddle_to_onnx(
            DET_DIR, "inference.json", "inference.pdiparams",
            str(onnx_path), c["opset_version"],
            optimize_tool=c["optimize_tool"],
            enable_dist_prim_all=c["enable_dist_prim_all"],
            enable_auto_update_opset=c["enable_auto_update_opset"],
            enable_onnx_checker=c["enable_onnx_checker"])
        if not ok:
            print(f"  [{i+1}/{len(combos)}] {tag}: EXPORT FAILED ({dt:.1f}s)")
            print(f"    err tail: {err[-200:]}")
            results.append({"combo": c, "export_ok": False, "error": err[:500]})
            continue
        try:
            m = onnx.load(str(onnx_path))
            size = onnx_path.stat().st_size
            opset = m.opset_import[0].version
            n_nodes = len(m.graph.node)
            n_inits = len(m.graph.initializer)
        except Exception as e:
            print(f"  [{i+1}/{len(combos)}] {tag}: load FAILED: {e}")
            results.append({"combo": c, "export_ok": True, "load_ok": False})
            continue

        try:
            t0 = time.time()
            y_ort = run_ort(onnx_path, x)
            ort_dt = time.time() - t0
        except Exception as e:
            print(f"  [{i+1}/{len(combos)}] {tag}: ORT FAILED: {str(e)[:200]}")
            results.append({"combo": c, "export_ok": True, "load_ok": True,
                            "ort_ok": False, "error": str(e)[:200]})
            continue
        d = diff_stats(y_ort, paddle_ref)
        print(f"  [{i+1}/{len(combos)}] {tag}: max={d['max_abs']:.4f} mean={d['mean_abs']:.4e} %>0.1={d['pct_gt_01']:.4f}% (nodes={n_nodes} size={size} ORT {ort_dt:.2f}s)")
        results.append({"combo": c, "export_ok": True, "load_ok": True,
                        "ort_ok": True, "diff_vs_paddle": d,
                        "ort_time_s": ort_dt, "onnx_size": size,
                        "n_nodes": n_nodes, "opset": opset})

    Path(args.out).write_text(json.dumps({
        "n_combos": len(combos),
        "results": results,
    }, indent=2, default=str))
    print(f"\nreport -> {args.out}")

    print()
    print("=== Summary (sorted by mean diff to Paddle) ===")
    print(f"{'rank':>4s}  {'combo':35s}  {'max':>10s}  {'mean':>10s}  {'%>0.1':>7s}  {'nodes':>6s}")
    valid = [r for r in results if r.get("ort_ok") and r.get("diff_vs_paddle")]
    valid.sort(key=lambda r: r["diff_vs_paddle"]["mean_abs"])
    for i, r in enumerate(valid):
        d = r["diff_vs_paddle"]
        c = r["combo"]
        print(f"  {i:2d}  {c['tag'][:34]:35s}  {d['max_abs']:>10.4f}  {d['mean_abs']:>10.4e}  {d['pct_gt_01']:>6.2f}%  {r.get('n_nodes', -1):6d}")


if __name__ == "__main__":
    sys.exit(main())
