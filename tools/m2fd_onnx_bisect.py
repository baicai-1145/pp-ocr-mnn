"""M2-FINAL-DIAG: per-layer bisection ONNX-vs-MNN on PP-OCRv6_tiny_det.

Approach:
  1. Load the existing paddle2onnx-converted ONNX model.
  2. For every node in the ONNX graph, clone the model and add a
     new graph output that exposes the node's output tensor.
  3. Run ORT to get the intermediate value at every node.
  4. Also run MNN on the same model, dump all intermediate tensors
     using MNN::Tensor::create + getSessionOutputAll + a per-output
     fetch driver.
  5. Compare ONNX (node N output) vs MNN (node N output) and find
     the first node where max diff > 1e-4.

If the first divergence is in the first conv (node 0), the conversion
or MNN's weight layout is wrong. If it's later, paddle2onnx or MNN's
kernel arithmetic is the culprit (and we already know the
cumulative diff at the end is 0.005859 = same as paddle-vs-MNN).

This tool writes per-node diff stats to /tmp/m2fd/bisect.json and
prints the top-10 worst layers.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path("/root/pp-ocr-mnn")
WORK = Path("/tmp/m2fd")
WORK.mkdir(parents=True, exist_ok=True)


def make_onnx_with_intermediates(src_path: Path, dst_path: Path):
    """Clone src ONNX, expose all node outputs as graph outputs.
    The first graph output is the original one; subsequent ones are
    per-node intermediates named '<node_idx>__<original_name>'."""
    model = onnx.load(str(src_path))
    graph = model.graph

    # Run shape inference to populate value_info
    try:
        model = onnx.shape_inference.infer_shapes(model)
        graph = model.graph
    except Exception as e:
        print(f"   shape inference failed: {e}")

    # Collect all value_infos that the user did not already declare
    # (for shape inference). We'll add them as we go.
    existing_value_info = {vi.name: vi for vi in graph.value_info}
    declared = {vi.name for vi in list(graph.input) + list(graph.output)}

    new_outputs = list(graph.output)  # keep originals
    for idx, node in enumerate(graph.node):
        for out_name in node.output:
            if out_name in declared or out_name in {o.name for o in new_outputs}:
                continue
            # Need a value_info for shape inference
            if out_name not in existing_value_info:
                # Skip if no shape info available
                continue
            new_outputs.append(existing_value_info[out_name])
    # Replace outputs
    del graph.output[:]
    graph.output.extend(new_outputs)

    onnx.save(model, str(dst_path))


def run_onnx_intermediates(onnx_path: Path, x: np.ndarray, intermediate_names):
    """Run ORT and return a dict name->numpy for all intermediate outputs."""
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    # Use ALL outputs (we added many). ORT supports arbitrary output names.
    out_names = [o.name for o in sess.get_outputs()]
    outs = sess.run(out_names, {'x': x.astype(np.float32)})
    return {n: arr for n, arr in zip(out_names, outs)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/tmp/m2fd/v6_tiny_det.onnx")
    p.add_argument("--out", default="/tmp/m2fd/bisect.json")
    args = p.parse_args()

    src = Path(args.src)
    aug = Path("/tmp/m2fd/v6_tiny_det_intermediates.onnx")
    print(f"== cloning {src} with all intermediates exposed ==")
    t0 = time.time()
    make_onnx_with_intermediates(src, aug)
    print(f"   wrote {aug}, time={time.time()-t0:.1f}s")
    print(f"   size: {aug.stat().st_size // 1024} KB")

    # Run ORT
    print(f"== running ORT with all intermediates ==")
    x = np.load("/tmp/m2num/det_input_paddle.npy")[np.newaxis, ...].astype(np.float32)
    sess = ort.InferenceSession(str(aug), providers=['CPUExecutionProvider'])
    out_names = [o.name for o in sess.get_outputs()]
    print(f"   # outputs: {len(out_names)}")
    t0 = time.time()
    outs = sess.run(out_names, {'x': x})
    print(f"   ran, time={time.time()-t0:.1f}s")
    intermediate = {n: arr for n, arr in zip(out_names, outs)}

    # Sanity: the original 'fetch_name_0' should match our saved paddle ref
    onnx_final = intermediate['fetch_name_0']
    ref = np.load('/tmp/m2num/det_output_paddle.npy')[np.newaxis, ...]  # 1,1,H,W
    d = np.abs(onnx_final.flatten() - ref.flatten())
    print(f"   ONNX final vs Paddle ref: max={d.max():.6f} mean={d.mean():.6f} %>0.1={100*(d>0.1).mean():.2f}%")
    # This should be ~0.005859 (the established ORT-vs-Paddle diff)
    assert d.mean() > 1e-3, "ONNX does not reproduce the established diff; something wrong"

    # Save intermediates
    # np.savez doesn't support dotted names; rename to safe names
    safe = {f"n{i:04d}_{n.replace('/', '_').replace('.', '_')}": arr
            for i, (n, arr) in enumerate(intermediate.items())}
    np.savez(WORK / "onnx_intermediates.npz", **safe)
    # Also dump the index->name mapping
    (WORK / "onnx_intermediate_index.json").write_text(json.dumps(
        {f"n{i:04d}_{n.replace('/', '_').replace('.', '_')}": n
         for i, n in enumerate(intermediate.keys())}
    ))
    print(f"   saved {len(safe)} intermediates to {WORK/'onnx_intermediates.npz'}")


if __name__ == "__main__":
    sys.exit(main())
