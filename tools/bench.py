#!/usr/bin/env python3
"""pp-ocr-mnn M3-PERF1: per-stage latency benchmark.

Runs the ppocr_cli with --profile over a fixed image list for each
(det, rec, backend, threads) cell, then aggregates per-stage
mean/std. Produces a JSON blob (--out) that PERF_BASELINE.md tables
are generated from.

Cells:
  models : PP-OCRv6_tiny (min), PP-OCRv6_medium (max), PP-OCRv4_mobile (legacy)
  backend: cpu x {1,4,8} threads, cuda
  images : /root/ocr_test_imgs/{zh,en}/00..09.jpg (20 images: 10 langs
           x 2 languages — representative for stage mix; each is
           street-scene style with 5-10 text lines)
  repeat : 8 warm runs per image (first run reported separately as
           cold start)

Usage:
  python3 tools/bench.py --build /root/pp-ws/post/build-cuda \
      --models-dir /root/pp-ocr-mnn/models --out /tmp/perf_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

STAGES = ["decode_ms", "det_prep_ms", "det_run_ms", "db_post_ms",
          "crop_warp_ms", "rec_prep_ms", "rec_run_ms", "ctc_decode_ms",
          "cls_ms", "e2e_ms"]

MODELS = {
    "v6_tiny":    ("PP-OCRv6_tiny_det",    "PP-OCRv6_tiny_rec"),
    "v6_medium":  ("PP-OCRv6_medium_det",  "PP-OCRv6_medium_rec"),
    "v4_mobile":  ("PP-OCRv4_mobile_det",  "PP-OCRv4_mobile_rec"),
}

IMG_ROOT = Path("/root/ocr_test_imgs")
LANGS = ["zh", "en"]


def image_list(n_per_lang: int) -> list[Path]:
    imgs = []
    for lang in LANGS:
        files = sorted((IMG_ROOT / lang).glob("*.jpg"))[:n_per_lang]
        imgs.extend(files)
    return imgs


def run_cell(cli: Path, models_dir: str, det: str, rec: str, backend: str,
             threads: int, imgs: list[Path], warm: int,
             ld_prepend: str = "") -> dict:
    env = dict(os.environ, LD_LIBRARY_PATH=
               (ld_prepend + ":" if ld_prepend else "")
               + "/usr/lib/x86_64-linux-gnu")
    per_img = []          # list of profile dicts (warm runs)
    raw = []              # per-(img, rep) records incl. image name
    cold = None
    for i, img in enumerate(imgs):
        for rep in range(warm + 1):
            out_json = f"/tmp/ppocr_bench_{os.getpid()}.json"
            t0 = time.perf_counter()
            p = subprocess.run(
                [str(cli), "--image", str(img),
                 "--det-config", str(CONFIGS / f"{det}.json"),
                 "--rec-config", str(CONFIGS / f"{rec}.json"),
                 "--model-dir", models_dir,
                 "--backend", backend, "--threads", str(threads),
                 "--profile", "--json", out_json],
                env=env, capture_output=True, text=True)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if p.returncode != 0:
                print(f"  FAIL rc={p.returncode} {img.name} rep={rep}\n"
                      f"{p.stderr[-400:]}", file=sys.stderr)
                continue
            d = json.load(open(out_json))
            prof = d["profile"]
            prof["wall_ms"] = wall_ms     # process-level wall clock
            prof["image"] = img.name
            if rep == 0:
                cold = prof               # first run = cold (fresh process)
            else:
                per_img.append(prof)
            raw.append(dict(prof))
    if not per_img:
        return {"error": "all runs failed"}
    agg = {}
    for s in STAGES + ["wall_ms"]:
        vals = [p[s] for p in per_img]
        agg[s] = {"mean": statistics.mean(vals),
                  "std": statistics.pstdev(vals),
                  "min": min(vals), "max": max(vals)}
    agg["n_runs"] = len(per_img)
    agg["cold"] = cold
    agg["cold_wall_ms"] = None
    agg["raw"] = raw
    # Per-image medians: robust cell figure immune to the dense-image skew
    # (one 302-box false-positive image otherwise dominates the mean).
    img_names = sorted({r["image"] for r in raw})
    per_img_med = {}
    for n in img_names:
        e2es = sorted(r["e2e_ms"] for r in raw if r["image"] == n)
        per_img_med[n] = e2es[len(e2es) // 2]
    med_vals = sorted(per_img_med.values())
    agg["img_median_e2e_ms"] = med_vals[len(med_vals) // 2]
    agg["per_image_median_e2e"] = per_img_med
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default=str(ROOT / "build-cuda"),
                    help="build dir containing ppocr_cli (cuda-linked)")
    ap.add_argument("--cli", default=None,
                    help="direct path to ppocr_cli (overrides --build)")
    ap.add_argument("--models-dir", default="/root/pp-ocr-mnn/models")
    ap.add_argument("--backend", default="all",
                    choices=["all", "cpu", "cuda"])
    ap.add_argument("--models", default="all",
                    help="comma list from: v6_tiny,v6_medium,v4_mobile")
    ap.add_argument("--threads", default="1,4,8",
                    help="cpu thread counts to sweep")
    ap.add_argument("--imgs-per-lang", type=int, default=10)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--out", default="/tmp/perf_baseline.json")
    args = ap.parse_args()

    cli = Path(args.cli) if args.cli else Path(args.build) / "ppocr_cli"
    if not cli.exists():
        print(f"ppocr_cli not found at {cli}", file=sys.stderr)
        return 2

    # The CUDA-linked cli needs build_cuda on LD_LIBRARY_PATH even for
    # CPU runs (the .so is linked but the CUDA backend is only entered
    # when requested).
    ld = str(ROOT / "third_party" / "MNN" / "build_cuda")

    model_filter = (None if args.models == "all"
                    else set(args.models.split(",")))
    backends = (["cpu", "cuda"] if args.backend == "all"
                else [args.backend])
    threads = [int(t) for t in args.threads.split(",")]

    imgs = image_list(args.imgs_per_lang)
    print(f"images: {len(imgs)}  warm runs/img: {args.warm}", flush=True)

    result = {
        "meta": {
            "cli": str(cli),
            "models_dir": args.models_dir,
            "images": [str(i) for i in imgs],
            "warm_runs": args.warm,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": os.uname().nodename,
        },
        "cells": {},
    }

    for mname, (det, rec) in MODELS.items():
        if model_filter and mname not in model_filter:
            continue
        for backend in backends:
            thread_list = threads if backend == "cpu" else [4]
            for th in thread_list:
                cell = f"{mname}/{backend}/t{th}"
                print(f"== {cell}", flush=True)
                t0 = time.perf_counter()
                agg = run_cell(cli, args.models_dir, det, rec, backend,
                               th, imgs, args.warm, ld_prepend=ld)
                dt = time.perf_counter() - t0
                if "error" in agg:
                    print(f"   {agg['error']}", flush=True)
                    continue
                e2e = agg["e2e_ms"]["mean"]
                fps = 1000.0 / e2e if e2e > 0 else 0.0
                print(f"   e2e {e2e:7.1f} ms  FPS {fps:6.2f}  "
                      f"(det_run {agg['det_run_ms']['mean']:6.1f}, "
                      f"rec_run {agg['rec_run_ms']['mean']:5.1f}, "
                      f"wall {agg['wall_ms']['mean']:6.1f})  "
                      f"[{dt:.0f}s]", flush=True)
                result["cells"][cell] = agg

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
