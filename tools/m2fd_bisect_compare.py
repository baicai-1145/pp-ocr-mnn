"""M2-FINAL-DIAG: bisect ONNX-vs-MNN per-layer diff using NAME + POSITION matching.

Both MNN and ONNX preserve the original Paddle op names. MNN's names
look like 'op<idx>_<orig_name>_<raster_0>' and ONNX's look like
'<orig_name>'. Names can collide (e.g. MNN has 'op0000_Add.1_raster_0'
and 'op0001_Add.1'; both refer to the same Paddle op 'Add.1' but
captured at different points in MNN's executor — pre-ReLU vs post-ReLU).

Strategy: match by NAME first, but when there are multiple MNN entries
with the same name (after suffix stripping), pick the one whose
**output shape** matches the ONNX entry. Tiebreak by closest execution
position (mnn_idx ~ 2 * onnx_idx).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path("/tmp/m2fd")


def load_onnx_intermediates(npz_path: Path, index_path: Path) -> dict:
    npz = np.load(str(npz_path))
    idx = json.load(open(index_path))
    out = {}
    for k, v in npz.items():
        orig = idx.get(k, k)
        out[orig] = v
    return out


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


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    if a.size != b.size:
        return {"max_abs": float("nan"), "mean_abs": float("nan"),
                "pct_gt_001": float("nan"), "pct_gt_01": float("nan"),
                "size_mismatch": True, "n_a": int(a.size), "n_b": int(b.size)}
    d = np.abs(a - b)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "pct_gt_001": 100.0 * float((d > 0.001).mean()),
        "pct_gt_01": 100.0 * float((d > 0.01).mean()),
        "n": int(d.size),
    }


def mnn_to_orig_names(mnn_name: str) -> list:
    """Return candidate orig names for an MNN op. Strips 'opNNNN_' prefix
    and various MNN suffixes."""
    if not mnn_name.startswith("op"):
        return [mnn_name]
    rest = mnn_name[7:]
    candidates = [rest]
    for c in list(candidates):
        if c.endswith("_raster_0"):
            candidates.append(c[:-len("_raster_0")])
    if "__" in rest:
        candidates.append(rest.split("__")[0])
    return list(dict.fromkeys(candidates))


def match_mnn_to_onnx(mnn_data: dict, onnx_data: dict) -> list:
    """For each MNN op-output, find the ONNX node with the same name
    (or name with _raster_0 stripped) AND same shape. If multiple
    candidates, pick the closest to 2*mnn_idx in execution order."""
    results = []
    onnx_names_ordered = list(onnx_data.keys())
    onnx_by_name_shape: dict[tuple, list[tuple]] = defaultdict(list)
    for i, n in enumerate(onnx_names_ordered):
        onnx_by_name_shape[(n, tuple(onnx_data[n].shape))].append((i, n))

    mnn_ordered = sorted(mnn_data.items(), key=lambda kv: kv[0])

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

        onnx_name = None
        candidates = mnn_to_orig_names(mnn_name)
        for cand in candidates:
            key = (cand, mnn_shape)
            if key in onnx_by_name_shape:
                opts = onnx_by_name_shape[key]
                if len(opts) == 1:
                    onnx_name = opts[0][1]
                else:
                    best = min(opts, key=lambda t: abs(t[0] - 2 * max(mnn_idx, 0)))
                    onnx_name = best[1]
                break

        if onnx_name is None:
            # fallback: closest shape match within ±30 of 2*mnn_idx
            center = 2 * max(mnn_idx, 0)
            best_d = float("inf")
            best_name = None
            for delta in range(-30, 31):
                ci = center + delta
                if 0 <= ci < len(onnx_names_ordered):
                    cand = onnx_names_ordered[ci]
                    if tuple(onnx_data[cand].shape) == mnn_shape:
                        onnx_name = cand
                        break
            if onnx_name is None:
                results.append((mnn_name, None, {"shape_mismatch": True,
                                                  "mnn_shape": list(mnn_shape),
                                                  "mnn_idx": mnn_idx}))
                continue
        d = diff_stats(mnn_arr, onnx_data[onnx_name])
        d["mnn_idx"] = mnn_idx
        d["onnx_idx"] = list(onnx_data.keys()).index(onnx_name)
        results.append((mnn_name, onnx_name, d))
    results.sort(key=lambda r: -r[2].get("max_abs", 0)
                  if not r[2].get("size_mismatch") and not r[2].get("shape_mismatch")
                  else 0)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/m2fd/bisect_v2.json")
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

    print("== matching (name + shape) ==")
    t0 = time.time()
    results = match_mnn_to_onnx(mnn_data, onnx_data)
    matched = [r for r in results if r[1] is not None]
    print(f"   {len(matched)} matched, {len(results)-len(matched)} unmatched, {time.time()-t0:.1f}s")

    # Top 20 worst
    print()
    print(f"{'rank':>4s}  {'MNN op':40s}  {'ONNX name':30s}  {'mnn_sh':25s}  {'max':>10s}  {'mean':>10s}  {'%>0.1':>6s}")
    print("-" * 140)
    worst = []
    for i, (mnn_n, onnx_n, d) in enumerate(results[:20]):
        if d.get("shape_mismatch"):
            print(f"  {i:2d}  {mnn_n[:39]:40s}  (no match, mnn={d.get('mnn_shape')})")
            continue
        if d.get("size_mismatch"):
            print(f"  {i:2d}  {mnn_n[:39]:40s}  (size mismatch)")
            continue
        maxv = d.get("max_abs", float("nan"))
        meanv = d.get("mean_abs", float("nan"))
        pct = d.get("pct_gt_01", float("nan"))
        mnn_sh = d.get("mnn_shape", list(mnn_data[mnn_n].shape))
        print(f"  {i:2d}  {mnn_n[:39]:40s}  {onnx_n[:29]:30s}  {str(mnn_sh)[:24]:25s}  {maxv:>10.6f}  {meanv:>10.6e}  {pct:>5.2f}%")
        worst.append({"rank": i, "mnn": mnn_n, "onnx": onnx_n, "diff": d})

    # First divergence in execution order
    in_order = sorted(matched, key=lambda r: r[2]["mnn_idx"])
    first_div = None
    for mnn_n, onnx_n, d in in_order:
        if d.get("max_abs", 0) > args.threshold:
            first_div = (mnn_n, onnx_n, d)
            break

    print()
    print(f"== first MNN-vs-ONNX divergence in execution order (max > {args.threshold}) ==")
    if first_div:
        mnn_n, onnx_n, d = first_div
        print(f"   MNN op: {mnn_n} (mnn_idx={d.get('mnn_idx')})")
        print(f"   ONNX name: {onnx_n} (onnx_idx={d.get('onnx_idx')})")
        print(f"   max={d['max_abs']:.6e} mean={d['mean_abs']:.6e} %>0.001={d['pct_gt_001']:.4f}% %>0.1={d['pct_gt_01']:.4f}%")
    else:
        print(f"   no divergence found at threshold {args.threshold}")
        maxd = max((d["max_abs"] for _, _, d in matched), default=0)
        print(f"   max diff over all matched ops: {maxd:.6e}")

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
