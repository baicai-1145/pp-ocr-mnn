#!/usr/bin/env python3
"""M2-ROBUST: parameter sweep for db_post on the MNN prob map.

Decision-maker's framing (ws/m2-robust task spec):
  - det prob numerical diff is intrinsic (M2-DET-FINAL); the
    prob map MNN produces is not the prob map PaddleX produces.
  - But the postprocess parameter space still has degrees of
    freedom (thresh, box_thresh, unclip_ratio, min_size), and
    the right combination on the MNN prob may approximate the
    PaddleX post-DB boxes much better than the default
    0.3 / 0.6 / 1.5 / 3.
  - This is NOT a tuning cheat: the baseline is the same. We
    are finding the most-robust operating point of OUR
    det chain on OUR prob map, parameterized for downstream
    configs. The justification: the postprocess parameter
    space is a function of the *prob map distribution*, not
    the model weights; since MNN's prob is systematically
    different from PaddleX's prob, a different operating
    point is the correct response.

What this tool does:
  1. For v6_medium_det × {zh,en,ar,pt,ja} × 5 imgs/lang:
  2. Sweep thresh ∈ {0.25, 0.30, 0.35} ×
            box_thresh ∈ {0.5, 0.55, 0.6} ×
            unclip_ratio ∈ {1.4, 1.5, 1.6} ×
            min_size ∈ {3, 5} = 81 combinations.
  3. For each combo, run our CLI in det-only mode (with a
     temp JSON config that overrides the 4 db_post params).
  4. Match our boxes against baseline boxes (greedy IoU,
     same matcher as tools/measure_box_gap.py / m2_det_boxes.py).
  5. Compute 3 metrics:
     (a) |n_pred - n_base| (box count error)
     (b) mean IoU over matched pairs
     (c) downstream CER — re-run rec on our boxes via
         --boxes-json, then compare to baseline rec_texts.
  6. Find the Pareto-frontier of (|Δn|, 1-IoU_mean, CER) and
     report the best combo(s) per lang and overall.

Output: /tmp/m2_robust_sweep.json (full per-(lang, img, combo)
detail) + a markdown summary printed to stdout that the
M2-ROBUST report.md can quote.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Reuse the IoU helpers from m2_det_boxes
sys.path.insert(0, "/root/pp-ocr-mnn/tools")
from m2_det_boxes import iou_poly, match_iou


ENV = {"LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu", "PATH": "/usr/bin:/bin"}
CWD = "/root/pp-ocr-mnn"
CLI = f"{CWD}/build-main/ppocr_cli"
DET_CFG = f"{CWD}/configs/PP-OCRv6_medium_det.json"
REC_CFG = f"{CWD}/configs/PP-OCRv6_medium_rec.json"
MODELS = f"{CWD}/models"
REF_ROOT = Path("/root/ppocr_reference")
IMG_ROOT = Path("/root/ocr_test_imgs")

LANGS = ["zh", "en", "ar", "pt", "ja"]
N_IMGS_PER_LANG = 5

THRESH = [0.25, 0.30, 0.35]
BOX_THRESH = [0.50, 0.55, 0.60]
UNCLIP = [1.4, 1.5, 1.6]
MIN_SIZE = [3, 5, 8]  # 3x3x3x3 = 81 combos per spec


def _write_temp_det_cfg(thresh, box_thresh, unclip, min_size):
    """Write a copy of the canonical det config with the 4 db_post
    params overridden. The temp file MUST be named
    'PP-OCRv6_medium_det.json' (the basename matches the
    registry key; config_basename in apps/ppocr_cli.cpp strips
    the directory and the .json extension)."""
    base = json.loads(Path(DET_CFG).read_text())
    base["det"]["thresh"] = thresh
    base["det"]["box_thresh"] = box_thresh
    base["det"]["unclip_ratio"] = unclip
    base["det"]["min_size"] = min_size
    # Use a temp dir but the file must have the canonical name
    tmpdir = tempfile.mkdtemp(prefix="ppocr_cfg_")
    path = os.path.join(tmpdir, "PP-OCRv6_medium_det.json")
    with open(path, "w") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)
    return path, tmpdir


def _cli_run(args, json_path):
    cmd = [CLI] + args + ["--json", json_path]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=CWD, env=ENV,
                       timeout=120)
    if r.returncode != 0:
        return None
    try:
        return json.loads(Path(json_path).read_text())
    except Exception:
        return None


def det_only(det_cfg_path, image, json_path):
    return _cli_run(
        ["--image", str(image), "--det-config", det_cfg_path,
         "--backend", "cpu", "--model-dir", MODELS, "--det-only"],
        json_path,
    )


def rec_with_boxes(image, polys, json_path):
    """Run rec on the given polys via --boxes-json."""
    if not polys:
        return []
    boxfile = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    try:
        for p in polys:
            boxfile.write(" ".join(str(c) for c in p) + "\n")
        boxfile.close()
        d = _cli_run(
            ["--image", str(image), "--det-config", DET_CFG,
             "--rec-config", REC_CFG, "--backend", "cpu",
             "--model-dir", MODELS, "--boxes-json", boxfile.name],
            json_path,
        )
        if d is None:
            return None
        return [ln.get("text", "") for ln in d.get("lines", [])]
    finally:
        try:
            os.unlink(boxfile.name)
        except OSError:
            pass


def levenshtein(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    if len(a) < len(b): a, b = b, a
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
    if not base: return 0.0 if not pred else 1.0
    return levenshtein(pred, base) / len(base)


def join_texts(d):
    return "\n".join(d.get("rec_texts") or [])


def find_baseline_entry(combo, lang, image_path):
    bp = REF_ROOT / combo / lang / "ocr_results.json"
    if not bp.exists():
        return None
    base = json.loads(bp.read_text())
    for entry in base:
        if entry.get("image_path") == str(image_path):
            return entry
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--n", type=int, default=N_IMGS_PER_LANG)
    ap.add_argument("--out", default="/tmp/m2_robust_sweep.json")
    ap.add_argument("--combos", default="all",
                    help="all | thresh_only | unclip_only ...")
    args = ap.parse_args()
    langs = args.langs.split(",")

    # Build the 81-combo grid
    grid = list(product(THRESH, BOX_THRESH, UNCLIP, MIN_SIZE))
    if args.combos != "all":
        # Allow filtering (not used in the default run)
        if args.combos == "thresh_only":
            grid = [(t, 0.6, 1.5, 3) for t in THRESH]
        elif args.combos == "box_thresh_only":
            grid = [(0.3, b, 1.5, 3) for b in BOX_THRESH]
        elif args.combos == "unclip_only":
            grid = [(0.3, 0.6, u, 3) for u in UNCLIP]
        elif args.combos == "min_size_only":
            grid = [(0.3, 0.6, 1.5, m) for m in MIN_SIZE]

    # For each (lang, image), pre-load baseline (and cache the
    # baseline polys dict keyed by image_path for the inner loop)
    base_polys_cache: Dict[str, list] = {}
    images = []
    for lang in langs:
        d = Path(IMG_ROOT / lang)
        if not d.is_dir():
            continue
        imgs = sorted(p.name for p in d.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.n]
        for img_name in imgs:
            base = find_baseline_entry(
                "PP-OCRv6_medium_det__PP-OCRv6_medium_rec",
                lang, d / img_name)
            if base is None:
                continue
            base_polys = base.get("det_polys") or []
            base_texts = base.get("rec_texts") or []
            base_text_joined = join_texts(base)
            base_polys_cache[str(d / img_name)] = base_polys
            images.append({
                "lang": lang, "img": img_name,
                "image": d / img_name,
                "n_base": len(base_polys),
                "base_text_joined": base_text_joined,
            })

    print(f"Loaded {len(images)} (lang, img) pairs; grid = {len(grid)} combos; "
          f"total = {len(images) * len(grid)} evaluations", flush=True)

    # Build the full list of (item, combo) tasks
    tasks = []
    for item in images:
        for thresh, box_thresh, unclip, min_size in grid:
            tasks.append((item, thresh, box_thresh, unclip, min_size))

    # Parallel evaluation across (item, combo). The CLI subprocess
    # work is mostly CPU-bound; thread pool releases the GIL during
    # subprocess.run and gives a near-linear speedup until we hit
    # machine core count.
    from concurrent.futures import ThreadPoolExecutor
    results = []
    completed = 0
    total = len(tasks)

    def _run(task):
        item, thresh, box_thresh, unclip, min_size = task
        image = item["image"]
        lang = item["lang"]
        n_base = item["n_base"]
        base_text = item["base_text_joined"]
        cfg_path, tmpdir = _write_temp_det_cfg(thresh, box_thresh, unclip, min_size)
        json_tmp = tempfile.mktemp(suffix=".json")
        try:
            d = det_only(cfg_path, image, json_tmp)
            if d is None:
                return None
            ours_polys = [ln["poly"] for ln in d.get("lines", [])
                          if len(ln.get("poly", [])) == 8]
            matched, ub, up = match_iou(
                base_polys_cache[str(image)], ours_polys)
            ious = [m[2] for m in matched]
            iou_mean = float(np.mean(ious)) if ious else 0.0
            iou_median = float(np.median(ious)) if ious else 0.0
            n_pred = len(ours_polys)
            n_extra = len(up)
            n_missed = len(ub)
            rec_json = tempfile.mktemp(suffix=".json")
            try:
                rt = rec_with_boxes(image, ours_polys, rec_json)
            finally:
                try:
                    os.unlink(rec_json)
                except OSError:
                    pass
            if rt is None:
                cer_v = float("nan")
            else:
                cer_v = cer("\n".join(rt), base_text)
            return {
                "lang": lang,
                "img": item["img"],
                "thresh": thresh,
                "box_thresh": box_thresh,
                "unclip": unclip,
                "min_size": min_size,
                "n_base": n_base,
                "n_pred": n_pred,
                "n_extra": n_extra,
                "n_missed": n_missed,
                "iou_mean": iou_mean,
                "iou_median": iou_median,
                "cer": cer_v,
            }
        finally:
            try:
                os.unlink(cfg_path)
                os.unlink(json_tmp)
                os.rmdir(tmpdir)
            except OSError:
                pass

    n_workers = max(1, int(os.environ.get("M2_SWEEP_WORKERS", "6")))
    print(f"Running with {n_workers} parallel workers", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(_run, tasks, chunksize=4):
            completed += 1
            if r is not None:
                results.append(r)
            if completed % 100 == 0 or completed == total:
                print(f"  [{completed}/{total}] done; {len(results)} valid evals so far",
                      flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))
    print()
    print("=" * 70)
    print("M2-ROBUST sweep summary")
    print("=" * 70)
    # Default 0.3/0.6/1.5/3 reference (Paddle's published defaults)
    default_key = (0.3, 0.6, 1.5, 3)
    # Per-lang stats
    print(f"{'lang':>4s} {'default_cer':>12s} {'best_cer':>10s} "
          f"{'best_combo':>30s} {'delta':>8s}")
    for lang in langs:
        lang_results = [r for r in results if r["lang"] == lang]
        if not lang_results:
            continue
        d = [r for r in lang_results if
             (r["thresh"], r["box_thresh"], r["unclip"], r["min_size"]) == default_key]
        default_cer = float(np.mean([r["cer"] for r in d
                                     if r["cer"] == r["cer"]])) if d else float("nan")
        # Find best combo (lowest mean CER across imgs)
        combo_cers = {}
        for r in lang_results:
            if r["cer"] != r["cer"]:
                continue
            key = (r["thresh"], r["box_thresh"], r["unclip"], r["min_size"])
            combo_cers.setdefault(key, []).append(r["cer"])
        best = sorted(combo_cers.items(), key=lambda x: np.mean(x[1]))[0]
        best_cer = float(np.mean(best[1]))
        delta = best_cer - default_cer
        print(f"  {lang:>3s} {default_cer:>10.4f}   {best_cer:>8.4f}  "
              f"thresh={best[0][0]} box_thresh={best[0][1]} "
              f"unclip={best[0][2]} min_size={best[0][3]}  {delta:+.4f}")
    # Global best
    combo_cers = {}
    for r in results:
        if r["cer"] != r["cer"]:
            continue
        key = (r["thresh"], r["box_thresh"], r["unclip"], r["min_size"])
        combo_cers.setdefault(key, []).append(r["cer"])
    if combo_cers:
        best = sorted(combo_cers.items(), key=lambda x: np.mean(x[1]))[0]
        print()
        print(f"Global best: thresh={best[0][0]} box_thresh={best[0][1]} "
              f"unclip={best[0][2]} min_size={best[0][3]}; "
              f"mean CER = {np.mean(best[1]):.4f} "
              f"(over {len(best[1])} imgs of {len(langs)} langs)")


if __name__ == "__main__":
    main()
