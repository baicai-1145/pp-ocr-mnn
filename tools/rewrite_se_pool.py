#!/usr/bin/env python3
"""rewrite_se_pool.py — replace global-avg Pooling3D ops with Reduction(MEAN).

The PP-OCR v6 det models carry SE blocks whose global average pooling was
converted by paddle2onnx into Pooling3D{isGlobal:true}. MNN 3.6.1's Metal
pooling_avg kernel dispatches ONE threadgroup for a 1x1 output, so each of
the 8 SE poolings serially scans W*H on a single SIMD lane (~2.7ms each,
~47-50% of the Metal det GPU time).

This script rewrites the MNN-JSON graph: Pooling3D{isGlobal, AVEPOOL} on a
4-D input becomes Reduction{MEAN, dim=[2,3], keepDims=true} — mathematically
the same operation (mean over H,W), executed by MNN's parallel SIMD-group
reduction kernels instead.

Usage:
  python3 rewrite_se_pool.py in.json out.json [--dry]
"""
import json
import sys


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    dry = "--dry" in sys.argv
    d = json.load(open(src))
    ops = d["oplists"]
    n_rew = 0
    for o in ops:
        if o.get("type") != "Pooling3D":
            continue
        main = o.get("main", {})
        if not main.get("isGlobal"):
            continue
        if main.get("type") not in (None, "AVEPOOL"):
            continue
        # 4-D input (NCHW): 2D global avg pool == Reduction MEAN over H,W
        o["type"] = "Reduction"
        o["main_type"] = "ReductionParam"
        o["main"] = {
            "operation": "MEAN",
            "dim": [2, 3],
            "coeff": 0.0,
            "keepDims": True,
            "dType": "DT_FLOAT",
        }
        n_rew += 1
    print(f"rewrote {n_rew} global Pooling3D -> Reduction(MEAN,[2,3])")
    if dry:
        return 0
    json.dump(d, open(dst, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
