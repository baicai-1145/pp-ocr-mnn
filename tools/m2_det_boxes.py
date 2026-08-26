#!/usr/bin/env python3
"""M2-DET-BOXES: systematic box-gap decomposition.

Per decision-maker's M2-MATRIX-RERUN review (commit 6648752):
the residual CER after rot90 fix + baseline regen is *all* in
the det chain (box placement / over-segmentation). Two concrete
data points cited by the decision-maker's isolated boxes-json
experiments:
  - ar: 0.188 CER (not the rec's 0.99 hallucination we saw in
    the full pipeline)
  - pt: 0.004 CER (not the rec's 1.31)
The gap to baseline boxes is the entire residual error.

This tool does 3 things for v6_medium_det (the best det from
the matrix rerun) × zh,en,ar,pt,ja,ko × 3 images per lang:

  1. Run our CLI --det-only on each image, parse the resulting
     det_polys; load the baseline det_polys; greedy IoU-match
     them (same matcher as tools/measure_box_gap.py::match_iou).
  2. Categorize the gap:
     (a) IoU < 0.8 = shifted box (positional error in our det)
     (b) extra pred box (over-segmentation: we have N+1, base
         has N for that region)
     (c) missing pred box (under-segmentation: base has N+1, we
         have N for that region)
  3. For each category, re-run rec on the **baseline** polys
     (i.e. use --boxes-json to feed the baseline geometry to
     our rec engine), and report the per-image CER that the
     rec would have produced. Compare to the boxes-json CER
     the decision-maker already has in their isolation study
     (this gives us the boxes-json per-language number without
     re-running their analysis; we re-derive it on the 18
     images and confirm the 0.18 / 0.00 order of magnitude).

Output: /tmp/m2_boxes.json + a markdown summary printed to
stdout that the M2-DET-BOXES report.md can quote verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants (mirroring tools/measure_box_gap.py)
# ---------------------------------------------------------------------------

ENV = {"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu", "PATH": "/usr/bin:/bin"}
CWD = "/root/pp-ocr-mnn"
CLI = f"{CWD}/build-main/ppocr_cli"
DET_CFG = f"{CWD}/configs/PP-OCRv6_medium_det.json"  # best det from M2-MATRIX
REC_CFG = f"{CWD}/configs/PP-OCRv6_medium_rec.json"  # best rec from M2-MATRIX
MODELS = f"{CWD}/models"
REF_ROOT = Path("/root/ppocr_reference")
IMG_ROOT = Path("/root/ocr_test_imgs")
LANGS = ["zh", "en", "ar", "pt", "ja", "ko"]
N_IMGS_PER_LANG = 3
IOU_GOOD = 0.8  # below = "shifted box"
IOU_OK = 0.5    # below = "bad shift", still counted as match


def poly_area(poly):
    pts = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def iou_poly(p1, p2):
    a1 = np.asarray(p1, dtype=np.float32).reshape(4, 2)
    a2 = np.asarray(p2, dtype=np.float32).reshape(4, 2)
    inter_area, _ = cv2.intersectConvexConvex(a1, a2, handleNested=True)
    if inter_area <= 0:
        return 0.0
    union = poly_area(p1) + poly_area(p2) - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area) / float(union)


def match_iou(base_polys, pred_polys, iou_min=IOU_OK):
    n_b, n_p = len(base_polys), len(pred_polys)
    ious = np.zeros((n_b, n_p), dtype=np.float32)
    for i, b in enumerate(base_polys):
        for j, p in enumerate(pred_polys):
            ious[i, j] = iou_poly(b, p)
    pairs = [(ious[i, j], i, j)
             for i in range(n_b) for j in range(n_p) if ious[i, j] >= iou_min]
    pairs.sort(reverse=True)
    matched = []
    used_b, used_p = set(), set()
    for iou, i, j in pairs:
        if i in used_b or j in used_p:
            continue
        matched.append((i, j, float(iou)))
        used_b.add(i)
        used_p.add(j)
    unmatched_base = [i for i in range(n_b) if i not in used_b]
    unmatched_pred = [j for j in range(n_p) if j not in used_p]
    return matched, unmatched_base, unmatched_pred


# ---------------------------------------------------------------------------
# CLI wrappers
# ---------------------------------------------------------------------------

def _cli_json(args):
    """Run ppocr_cli with the given flags, return parsed JSON dict.

    The CLI writes a temp JSON file via --json PATH; we read it
    back. (Same pattern as tools/run_reference.py::_run_one.)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_path = f.name
    try:
        cmd = [CLI] + args + ["--json", json_path]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=CWD, env=ENV,
                           timeout=120)
        if r.returncode != 0:
            return None
        return json.loads(Path(json_path).read_text())
    finally:
        try:
            os.unlink(json_path)
        except OSError:
            pass


def det_only(image):
    return _cli_json(["--image", str(image), "--det-config", DET_CFG,
                      "--backend", "cpu", "--model-dir", MODELS,
                      "--det-only"])


def rec_only_with_boxes_json(image, boxes_path):
    """Run CLI --boxes-json to feed the rec engine the baseline
    polys (skipping det). Returns the rec_texts list."""
    d = _cli_json(["--image", str(image), "--det-config", DET_CFG,
                   "--rec-config", REC_CFG, "--backend", "cpu",
                   "--model-dir", MODELS,
                   "--boxes-json", str(boxes_path)])
    if d is None:
        return None
    return [ln.get("text", "") for ln in d.get("lines", [])]


# ---------------------------------------------------------------------------
# Levenshtein + CER (no numpy)
# ---------------------------------------------------------------------------

def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    n, m = len(b), len(a)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[n]


def cer(pred, base):
    if not base:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, base) / len(base)


def join_texts(d):
    return "\n".join(d.get("rec_texts") or [])


# ---------------------------------------------------------------------------
# Per-image analysis
# ---------------------------------------------------------------------------

def find_baseline_entry(combo, lang, image_path):
    bp = REF_ROOT / combo / lang / "ocr_results.json"
    if not bp.exists():
        return None
    base = json.loads(bp.read_text())
    for entry in base:
        if entry.get("image_path") == str(image_path):
            return entry
    return None


def run_per_image(lang, img_name):
    """Run the full analysis for one (lang, image)."""
    image = IMG_ROOT / lang / img_name
    if not image.exists():
        return None
    # 1) our det-only polys
    ours_d = det_only(image)
    if ours_d is None:
        return {"error": "det-cli-fail", "image": img_name, "lang": lang}
    ours_polys = [ln["poly"] for ln in ours_d.get("lines", [])
                  if len(ln.get("poly", [])) == 8]
    # 2) baseline polys (use the natural pairing combo: v6_medium x v6_medium)
    base_entry = find_baseline_entry(
        "PP-OCRv6_medium_det__PP-OCRv6_medium_rec", lang, image)
    if base_entry is None:
        return {"error": "no-baseline-entry", "image": img_name, "lang": lang}
    base_polys = base_entry.get("det_polys") or []
    base_texts = base_entry.get("rec_texts") or []
    # 3) match
    matched, unmatched_base, unmatched_pred = match_iou(base_polys, ours_polys)
    # 4) categorize the matched pairs
    shifted = [m for m in matched if m[2] < IOU_GOOD]
    good = [m for m in matched if m[2] >= IOU_GOOD]
    # 5) baseline-feed rec CER (our MNN rec on the baseline polys)
    tmp = Path(tempfile.mkstemp(suffix=".txt")[1])
    try:
        tmp.write_text("\n".join(" ".join(str(c) for c in p) for p in base_polys))
        rec_texts_baseline = rec_only_with_boxes_json(image, tmp)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    if rec_texts_baseline is None:
        rec_cer_baseline = float("nan")
    else:
        pred_join = "\n".join(rec_texts_baseline)
        rec_cer_baseline = cer(pred_join, join_texts(base_entry))
    # 6) our-pipeline rec CER (i.e. what the full pipeline produced
    # for this image, already on disk from the M2-MATRIX-RERUN).
    pred_path = CWD + f"/results/PP-OCRv6_medium_det__PP-OCRv6_medium_rec/{lang}/pred.json"
    our_pipeline_texts = None
    if Path(pred_path).exists():
        pl = json.loads(Path(pred_path).read_text())
        for item in pl:
            if item.get("image_path") == str(image):
                our_pipeline_texts = item.get("rec_texts") or []
                break
    if our_pipeline_texts is not None:
        pred_join = "\n".join(our_pipeline_texts)
        rec_cer_ours = cer(pred_join, join_texts(base_entry))
    else:
        rec_cer_ours = float("nan")
    return {
        "image": img_name,
        "lang": lang,
        "n_base": len(base_polys),
        "n_pred": len(ours_polys),
        "n_matched_good_iou": len(good),
        "n_shifted_iou_lt_0.8": len(shifted),
        "n_unmatched_base": len(unmatched_base),  # we missed a base box
        "n_unmatched_pred": len(unmatched_pred),  # we added an extra box
        "iou_median": float(np.median([m[2] for m in matched])) if matched else 0,
        "cer_baseline_box_ours_rec": rec_cer_baseline,
        "cer_ours_box_ours_rec": rec_cer_ours,
        "cer_delta": rec_cer_ours - rec_cer_baseline,
        "base_polys": base_polys,
        "ours_polys": ours_polys,
        "matched": matched,
        "unmatched_base_idx": unmatched_base,
        "unmatched_pred_idx": unmatched_pred,
        "base_texts": base_texts,
        "our_pipeline_texts": our_pipeline_texts,
        "rec_texts_with_baseline_box": rec_texts_baseline,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--n", type=int, default=N_IMGS_PER_LANG,
                    help="images per lang")
    ap.add_argument("--out", default="/tmp/m2_boxes.json")
    args = ap.parse_args()
    langs = args.langs.split(",")
    out = {}
    for lang in langs:
        d = Path(IMG_ROOT / lang)
        if not d.is_dir():
            continue
        imgs = sorted(p.name for p in d.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.n]
        out[lang] = []
        for img_name in imgs:
            r = run_per_image(lang, img_name)
            if r is not None:
                out[lang].append(r)
                # Compact progress
                print(f"  [{lang}] {img_name}: n_base={r.get('n_base')} "
                      f"n_pred={r.get('n_pred')} "
                      f"matched={r.get('n_matched_good_iou')} "
                      f"shifted={r.get('n_shifted_iou_lt_0.8')} "
                      f"extra_pred={r.get('n_unmatched_pred')} "
                      f"missed_base={r.get('n_unmatched_base')} "
                      f"cer_base={r.get('cer_baseline_box_ours_rec'):.3f} "
                      f"cer_ours={r.get('cer_ours_box_ours_rec'):.3f}",
                      flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    # Per-lang summary
    print()
    print("=" * 70)
    print("M2-DET-BOXES per-lang summary")
    print("=" * 70)
    for lang in out:
        rows = out[lang]
        n = len(rows)
        if n == 0:
            continue
        tot_base = sum(r.get("n_base", 0) for r in rows)
        tot_pred = sum(r.get("n_pred", 0) for r in rows)
        tot_matched = sum(r.get("n_matched_good_iou", 0) for r in rows)
        tot_shifted = sum(r.get("n_shifted_iou_lt_0.8", 0) for r in rows)
        tot_extra = sum(r.get("n_unmatched_pred", 0) for r in rows)
        tot_missed = sum(r.get("n_unmatched_base", 0) for r in rows)
        cer_b = [r["cer_baseline_box_ours_rec"] for r in rows
                 if r.get("cer_baseline_box_ours_rec") == r["cer_baseline_box_ours_rec"]]
        cer_o = [r["cer_ours_box_ours_rec"] for r in rows
                 if r.get("cer_ours_box_ours_rec") == r["cer_ours_box_ours_rec"]]
        print(f"  {lang}: n_imgs={n} "
              f"base={tot_base} pred={tot_pred} "
              f"matched_good={tot_matched} shifted<0.8={tot_shifted} "
              f"extra_pred={tot_extra} missed_base={tot_missed}")
        print(f"      cer_baseline_box={np.mean(cer_b):.3f}  "
              f"cer_ours_full_pipe={np.mean(cer_o):.3f}  "
              f"delta={np.mean(cer_o) - np.mean(cer_b):+.3f}")


if __name__ == "__main__":
    main()
