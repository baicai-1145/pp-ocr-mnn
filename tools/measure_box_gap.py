#!/usr/bin/env python3
"""M2-DET-FINAL step 1: per-image box-gap measurement.

For each zh/en image we run our CLI with --det-only and compare
the resulting det_polys against the baseline det_polys in
/root/ppocr_reference/.../ocr_results.json.

For each image we compute a greedy 1-to-1 matching by IoU:
  - sort baseline and our boxes by area descending
  - walk through the larger set; for each box, take the
    unmatched counterpart with the highest IoU as the match
  - report the IoU distribution of matched pairs
  - report the unmatched count in each direction

Output stats:
  - per-image: n_base, n_pred, n_matched, IoU min/median/mean
  - global:    overall IoU distribution across the 20 images
  - categorical: how many boxes have IoU < 0.5, 0.5-0.8, 0.8-0.9,
    0.9-0.95, >= 0.95 (the "near-perfect" bin).

We use OpenCV's contourFromPoints + contourArea for poly area,
matching pyclipper/PaddleCR conventions (signed poly area for
CW/CCW determination is not relevant here since both polylines
are CW or both CCW after Paddle sort_min_area_rect).
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

ENV = {"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu", "PATH": "/usr/bin:/bin"}
# Configurable via env var; defaults to the standard ws/m2-pipe layout.
CLI = os.environ.get("PPOCR_CLI", "/root/pp-ocr-mnn/build-main/ppocr_cli")
DET_CFG = os.environ.get("PPOCR_DET_CFG", "configs/PP-OCRv6_tiny_det.json")
MODELS = os.environ.get("PPOCR_MODELS", "/root/pp-ocr-mnn/models")
CWD = os.environ.get("PPOCR_CWD", "/root/pp-ocr-mnn")
REF = Path(os.environ.get("PPOCR_REF",
    "/root/ppocr_reference/PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec"))
JSON_OUT = os.environ.get("PPOCR_JSON_OUT", "/tmp/m2df_pred.json")
LANG_DIR = "/root/ocr_test_imgs"


def poly_area(poly):
    """PaddleCR poly = 4 points (8 ints). Polygon area (signed)."""
    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]; y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def iou_poly(p1, p2):
    """Compute IoU between two quads via cv2 contour intersection."""
    a1 = np.asarray(p1, dtype=np.float32).reshape(4, 2)
    a2 = np.asarray(p2, dtype=np.float32).reshape(4, 2)
    inter_area, _ = cv2.intersectConvexConvex(a1, a2, handleNested=True)
    if inter_area <= 0:
        return 0.0
    union = poly_area(p1) + poly_area(p2) - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area) / float(union)


def match_iou(base_polys, pred_polys):
    """Greedy 1-to-1 matching by IoU.

    Returns:
      matched: list of (b_idx, p_idx, iou)
      unmatched_base: list of base indices
      unmatched_pred: list of pred indices
    """
    n_b, n_p = len(base_polys), len(pred_polys)
    ious = np.zeros((n_b, n_p), dtype=np.float32)
    for i, b in enumerate(base_polys):
        for j, p in enumerate(pred_polys):
            ious[i, j] = iou_poly(b, p)
    matched = []
    used_b = set()
    used_p = set()
    # Greedy: take all (i, j) pairs with iou > 0, sort by iou desc,
    # walk down and add matches while endpoints are unused.
    pairs = [(ious[i, j], i, j) for i in range(n_b) for j in range(n_p) if ious[i, j] > 0]
    pairs.sort(reverse=True)
    for iou, i, j in pairs:
        if i in used_b or j in used_p:
            continue
        if iou < 0.05:  # below reasonable threshold
            continue
        matched.append((i, j, float(iou)))
        used_b.add(i); used_p.add(j)
    unmatched_base = [i for i in range(n_b) if i not in used_b]
    unmatched_pred = [j for j in range(n_p) if j not in used_p]
    return matched, unmatched_base, unmatched_pred


def run_cli(img_path):
    """Run our CLI --det-only --json and return the parsed det_polys."""
    r = subprocess.run(
        [CLI, "--image", img_path, "--det-config", DET_CFG,
         "--backend", "cpu", "--model-dir", MODELS,
         "--det-only", "--json", JSON_OUT],
        capture_output=True, text=True, cwd=CWD, env=ENV)
    if r.returncode != 0:
        return None
    try:
        d = json.loads(Path(JSON_OUT).read_text())
        polys = []
        for line in d.get("lines", []):
            poly = line.get("poly", [])
            if len(poly) == 8:
                polys.append(poly)
        return polys
    except Exception:
        return None


def main():
    out = {}
    for lang in ["zh", "en"]:
        base = json.load(open(REF / lang / "ocr_results.json"))
        per_image = []
        all_ious = []
        all_unmatched_base = 0
        all_unmatched_pred = 0
        for item in base:
            img = item["image_path"]
            base_polys = item["det_polys"]
            pred_polys = run_cli(img)
            if pred_polys is None:
                print(f"  [{lang}] {Path(img).name}: CLI FAIL")
                continue
            matched, ub, up = match_iou(base_polys, pred_polys)
            ious = [m[2] for m in matched]
            entry = {
                "image": Path(img).name,
                "n_base": len(base_polys),
                "n_pred": len(pred_polys),
                "n_matched": len(matched),
                "iou_min": min(ious) if ious else 0.0,
                "iou_median": float(np.median(ious)) if ious else 0.0,
                "iou_mean": float(np.mean(ious)) if ious else 0.0,
                "n_unmatched_base": len(ub),
                "n_unmatched_pred": len(up),
                "matched": matched,
                "unmatched_base_idx": ub,
                "unmatched_pred_idx": up,
            }
            per_image.append(entry)
            all_ious.extend(ious)
            all_unmatched_base += len(ub)
            all_unmatched_pred += len(up)
        out[lang] = {
            "n_images": len(per_image),
            "total_base_polys": sum(p["n_base"] for p in per_image),
            "total_pred_polys": sum(p["n_pred"] for p in per_image),
            "total_matched": sum(p["n_matched"] for p in per_image),
            "iou_min": min(p["iou_min"] for p in per_image) if per_image else 0,
            "iou_median_global": float(np.median(all_ious)) if all_ious else 0,
            "iou_mean_global": float(np.mean(all_ious)) if all_ious else 0,
            "total_unmatched_base": all_unmatched_base,
            "total_unmatched_pred": all_unmatched_pred,
            "per_image": per_image,
        }
    Path("/tmp/m2df_box_gap.json").write_text(json.dumps(out, indent=2))

    # Summary
    print("=" * 60)
    print("M2-DET-FINAL box-gap measurement")
    print("=" * 60)
    for lang in ["zh", "en"]:
        d = out[lang]
        print(f"\n[{lang}] n_images={d['n_images']} "
              f"base={d['total_base_polys']} pred={d['total_pred_polys']} "
              f"matched={d['total_matched']} "
              f"unmatched_base={d['total_unmatched_base']} "
              f"unmatched_pred={d['total_unmatched_pred']}")
        print(f"  IoU min over images: {d['iou_min']:.4f}")
        print(f"  IoU median (all matched pairs): {d['iou_median_global']:.4f}")
        print(f"  IoU mean (all matched pairs):   {d['iou_mean_global']:.4f}")
        # IoU histogram
        ious = []
        for p in d['per_image']:
            ious.extend([m[2] for m in p['matched']])
        if ious:
            bins = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 1.0]
            hist, _ = np.histogram(ious, bins=bins)
            print("  IoU histogram (all matched pairs across all images):")
            for i in range(len(bins) - 1):
                bar = '#' * int(50 * hist[i] / max(hist.max(), 1))
                print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:>5d}  {bar}")
        # Worst per-image offenders
        worst = sorted(d['per_image'], key=lambda x: x['iou_min'])[:5]
        print("  Worst per-image (lowest min IoU):")
        for p in worst:
            print(f"    {p['image']:>12s}  "
                  f"base={p['n_base']} pred={p['n_pred']} matched={p['n_matched']}  "
                  f"iou min={p['iou_min']:.3f} med={p['iou_median']:.3f} "
                  f"unmatched_b={p['n_unmatched_base']} unmatched_p={p['n_unmatched_pred']}")
    print()
    print("Report: /tmp/m2df_box_gap.json")


if __name__ == "__main__":
    main()
