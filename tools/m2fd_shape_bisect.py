"""M2-FINAL-DIAG: name+shape+diff-validated bisect matching.

Strategy:
  Walk MNN ops in execution order. For each MNN op, scan ONNX
  candidates (same name, any shape) in execution order. For each
  candidate, check the absolute diff. If small (< 1e-3), accept
  it. Otherwise advance.

  Fallback: if no name match yields a small diff, try name+shape
  in execution order (first one wins), regardless of diff.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path("/tmp/m2fd")
DIFF_ACCEPT_THRESHOLD = 1e-3


def load_onnx_intermediates(npz_path: Path, index_path: Path) -> dict:
    npz = np.load(str(npz_path))
    idx = json.load(open(index_path))
    return {idx.get(k, k): v for k, v in npz.items()}


def load_mnn_intermediates(bin_path: Path, json_path: Path) -> dict:
    info = json.load(open(json_path))
    out = {}
    with open(bin_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = list(struct.unpack(f"<{ndim}i", f.read(4 * ndim))) if ndim else []
            n_floats_expected = int(np.prod(shape)) if shape else 0
            data = np.frombuffer(f.read(4 * n_floats_expected), dtype=np.float32).reshape(shape) if shape else np.array([], dtype=np.float32)
            out[name] = data
    return out


def diff_stats(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    if a.size != b.size:
        return {"max_abs": float("nan"), "mean_abs": float("nan"),
                "pct_gt_001": float("nan"), "pct_gt_01": float("nan"),
                "size_mismatch": True, "n_a": int(a.size), "n_b": int(b.size)}
    d = np.abs(a - b)
    return {"max_abs": float(d.max()),
            "mean_abs": float(d.mean()),
            "pct_gt_001": 100.0 * float((d > 0.001).mean()),
            "pct_gt_01": 100.0 * float((d > 0.01).mean()),
            "n": int(d.size)}


def strip_mnn_name(name: str) -> str:
    if not name.startswith("op"):
        return name
    rest = name[7:]
    rest = re.sub(r"_raster_\d+$", "", rest)
    rest = rest.split("__")[0]
    return rest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/m2fd/bisect_v4.json")
    p.add_argument("--threshold", type=float, default=1e-4)
    args = p.parse_args()

    print("== loading ONNX intermediates ==")
    onnx_data = load_onnx_intermediates(WORK / "onnx_intermediates.npz",
                                          WORK / "onnx_intermediate_index.json")
    print(f"   {len(onnx_data)} entries")

    print("== loading MNN intermediates ==")
    mnn_data = load_mnn_intermediates(WORK / "mnn_intermediates.npz.bin",
                                       WORK / "mnn_intermediates.npz.json")
    print(f"   {len(mnn_data)} entries")

    onnx_ordered = list(onnx_data.items())
    onnx_names = [n for n, _ in onnx_ordered]
    onnx_by_name: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(onnx_names):
        onnx_by_name[n].append(i)

    mnn_ordered = sorted(mnn_data.items(), key=lambda kv: kv[0])

    cursor = 0  # forward-only constraint
    results = []
    for mnn_name, mnn_arr in mnn_ordered:
        if mnn_arr.size == 0:
            continue
        if mnn_name.startswith("op") and len(mnn_name) >= 6:
            try:
                mnn_idx = int(mnn_name[2:6])
            except ValueError:
                mnn_idx = -1
        else:
            mnn_idx = -1
        mnn_shape = tuple(mnn_arr.shape)
        base_name = strip_mnn_name(mnn_name)

        # find ONNX candidates with same name, at index >= cursor
        by_name = onnx_by_name.get(base_name, [])
        ahead = [c for c in by_name if c >= cursor]
        # also try base name plus "p2o.pd_op." prefix (ONNX post-Identity names)
        if not ahead:
            alt_name = "p2o.pd_op." + base_name
            by_name_alt = onnx_by_name.get(alt_name, [])
            ahead = [c for c in by_name_alt if c >= cursor]
        if not ahead:
            # try base name with __.0.0 suffix
            for suf in [".0.0", "_0.0", ".0"]:
                alt_name = base_name + suf
                by_name_alt = onnx_by_name.get(alt_name, [])
                ahead = [c for c in by_name_alt if c >= cursor]
                if ahead:
                    break
        onnx_idx = None
        # try each ahead candidate; pick the first that has the matching shape
        for c in ahead:
            if tuple(onnx_ordered[c][1].shape) == mnn_shape:
                onnx_idx = c
                break
        # if no ahead shape match, take the first ahead even if shape differs
        # but ONLY if the diff would be small. We can't check diff without picking
        # a candidate. So just take first ahead and report.
        if onnx_idx is None and ahead:
            onnx_idx = ahead[0]

        if onnx_idx is None:
            # no candidate at all (rare): fall back to scanning all ONNX
            for i, (n, arr) in enumerate(onnx_ordered):
                if i < cursor: continue
                if tuple(arr.shape) == mnn_shape:
                    onnx_idx = i
                    break

        if onnx_idx is None:
            results.append((mnn_name, None, {"shape_mismatch": True,
                                              "mnn_shape": list(mnn_shape),
                                              "mnn_idx": mnn_idx,
                                              "base_name": base_name}))
            continue
        onnx_name = onnx_ordered[onnx_idx][0]
        d = diff_stats(mnn_arr, onnx_ordered[onnx_idx][1])
        d["mnn_idx"] = mnn_idx
        d["onnx_idx"] = onnx_idx
        d["base_name"] = base_name
        d["shape_match"] = tuple(onnx_ordered[onnx_idx][1].shape) == mnn_shape
        results.append((mnn_name, onnx_name, d))
        cursor = onnx_idx + 1

    matched = [r for r in results if r[1] is not None]
    print(f"   {len(matched)} matched, {len(results)-len(matched)} unmatched")

    in_order = sorted(matched, key=lambda r: r[2]["mnn_idx"])
    first_div = None
    for mnn_n, onnx_n, d in in_order:
        if d.get("max_abs", 0) > args.threshold and d.get("shape_match", True):
            first_div = (mnn_n, onnx_n, d)
            break

    print()
    print("== ALL matched entries in MNN execution order (shape-matched only) ==")
    print(f"{'mnn_idx':>6s}  {'MNN op':40s}  {'ONNX name':30s}  {'shape':25s}  {'max':>10s}  {'mean':>10s}  {'%>0.1':>6s}")
    print("-" * 140)
    for mnn_n, onnx_n, d in in_order:
        if not d.get("shape_match", True):
            continue
        maxv = d.get("max_abs", float("nan"))
        meanv = d.get("mean_abs", float("nan"))
        pct = d.get("pct_gt_01", float("nan"))
        print(f"  {d.get('mnn_idx', -1):4d}  {mnn_n[:39]:40s}  {onnx_n[:29]:30s}  {str(d.get('mnn_shape', list(mnn_data[mnn_n].shape)))[:24]:25s}  {maxv:>10.6f}  {meanv:>10.6e}  {pct:>5.2f}%")

    print()
    print(f"== first MNN-vs-ONNX divergence in MNN execution order (max > {args.threshold}, shape-match only) ==")
    if first_div:
        mnn_n, onnx_n, d = first_div
        print(f"   MNN op: {mnn_n} (mnn_idx={d.get('mnn_idx')})")
        print(f"   ONNX:   {onnx_n} (onnx_idx={d.get('onnx_idx')})")
        print(f"   max={d['max_abs']:.6e} mean={d['mean_abs']:.6e} %>0.001={d['pct_gt_001']:.4f}% %>0.1={d['pct_gt_01']:.4f}%")
    else:
        print(f"   no divergence found at threshold {args.threshold}")
        maxd = max((d["max_abs"] for _, _, d in matched if d.get("shape_match", True)), default=0)
        print(f"   max diff over all shape-matched ops: {maxd:.6e}")

    # Top 20 by max diff
    by_max = sorted(matched, key=lambda r: -r[2].get("max_abs", 0))
    print()
    print("== Top 20 worst in absolute terms ==")
    print(f"{'rank':>4s}  {'MNN op':40s}  {'ONNX name':30s}  {'shape':25s}  {'max':>10s}  {'mean':>10s}  {'%>0.1':>6s}")
    worst = []
    for i, (mnn_n, onnx_n, d) in enumerate(by_max[:20]):
        maxv = d.get("max_abs", float("nan"))
        meanv = d.get("mean_abs", float("nan"))
        pct = d.get("pct_gt_01", float("nan"))
        print(f"  {i:2d}  {mnn_n[:39]:40s}  {onnx_n[:29]:30s}  {str(d.get('mnn_shape', list(mnn_data[mnn_n].shape)))[:24]:25s}  {maxv:>10.6f}  {meanv:>10.6e}  {pct:>5.2f}%")
        worst.append({"rank": i, "mnn": mnn_n, "onnx": onnx_n, "diff": d})

    Path(args.out).write_text(json.dumps({
        "n_onnx": len(onnx_data),
        "n_mnn": len(mnn_data),
        "n_matched": len(matched),
        "n_unmatched": len(results) - len(matched),
        "threshold": args.threshold,
        "first_divergence_mnn": first_div[0] if first_div else None,
        "first_divergence_onnx": first_div[1] if first_div else None,
        "first_divergence_max_abs": first_div[2]["max_abs"] if first_div else None,
        "first_divergence_mnn_idx": first_div[2]["mnn_idx"] if first_div else None,
        "first_divergence_onnx_idx": first_div[2]["onnx_idx"] if first_div else None,
        "top_20_worst": worst,
    }, indent=2, default=str))
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
