#!/usr/bin/env python3
"""run_reference.py — drive ppocr_cli across the PP-OCR reference matrix.

Per docs/CONTRACT.md:
  * walks /root/ppocr_reference/*__*/ (skipping `strip__*`, `manifest*`,
    `*seal_det` (no __) dirs);
  * maps `<det>__<rec>` to configs (det name = dir name before __; rec name =
    after __; examples: `PP-OCRv5_mobile_det__korean_PP-OCRv5_mobile_rec` →
    det=`PP-OCRv5_mobile_det`, rec=`korean_PP-OCRv5_mobile_rec`);
  * for each language under /root/ocr_test_imgs/<lang>/0*.jpg, runs
    `ppocr_cli --image IMG --det-config ... --rec-config ... [--cls ...]`,
    parses its JSON output, and writes
      results/<combo>/<lang>/pred.json
    with the same schema as the baseline (minus GT fields):
      [{"image_path":..., "rec_texts":[...], "rec_scores":[...],
        "det_polys":[[8 floats],...]}]

  --cells REGEX filters combos (matched against combo name).
  --jobs N parallel processes.
  --cli PATH, --backend NAME, --cls-config PATH.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

REF_ROOT = Path("/root/ppocr_reference")
IMG_ROOT = Path("/root/ocr_test_imgs")


def discover_combos(cells_re: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """Return list of (combo_dir_name, det_name, rec_name) tuples."""
    out: List[Tuple[str, str, str]] = []
    for child in sorted(REF_ROOT.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if "__" not in name:
            continue
        if name.startswith("strip__"):
            continue
        det, rec = name.split("__", 1)
        if cells_re and not re.search(cells_re, name):
            continue
        out.append((name, det, rec))
    return out


def list_images(lang: str) -> List[Path]:
    d = IMG_ROOT / lang
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def resolve_configs(det: str, rec: str, configs_dir: Path) -> Tuple[Path, Path]:
    d = configs_dir / f"{det}.json"
    r = configs_dir / f"{rec}.json"
    if not d.exists():
        raise FileNotFoundError(f"missing det config: {d}")
    if not r.exists():
        raise FileNotFoundError(f"missing rec config: {r}")
    return d, r


# ---------------------------------------------------------------------------
# Per-image run
# ---------------------------------------------------------------------------

def _run_one(cli: str, *, image: Path, det_cfg: Path, rec_cfg: Path,
             cls_cfg: Optional[Path], backend: str, threads: int) -> Optional[dict]:
    cmd = [
        cli,
        "--image", str(image),
        "--det-config", str(det_cfg),
        "--rec-config", str(rec_cfg),
        "--backend", backend,
        "--threads", str(threads),
        "--json", "stdout",  # emit JSON to stdout (CLI contract permits this)
    ]
    if cls_cfg is not None:
        cmd[cmd.index("--image"):cmd.index("--image")+0] = ["--cls-config", str(cls_cfg)]
        # adjust insertion (simpler: rebuild)
    # Rebuild cleanly to avoid insertion bugs above
    cmd = [
        cli,
        "--image", str(image),
        "--det-config", str(det_cfg),
        "--rec-config", str(rec_cfg),
        "--backend", backend,
        "--threads", str(threads),
    ]
    if cls_cfg is not None:
        cmd += ["--cls-config", str(cls_cfg)]
    cmd += ["--json", "stdout"]

    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        sys.stderr.write(f"[run] rc={p.returncode} img={image} stderr={p.stderr[:200]}\n")
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[run] bad-json img={image} err={e} raw={p.stdout[:200]}\n")
        return None


# ---------------------------------------------------------------------------
# Result writers
# ---------------------------------------------------------------------------

def _poly_to_baseline(lines: list) -> list:
    """CLI emits `lines=[{"poly":[8 ints],"text":...,"score":...}]`.
    baseline uses `det_polys: [[x0,y0,...,x3,y3], ...]`. Pull each poly."""
    out = []
    for ln in lines or []:
        poly = ln.get("poly")
        if poly is None:
            continue
        # accept either 4 [x,y] pairs or 8 flat; keep flat ints.
        if isinstance(poly[0], (list, tuple)):
            flat = [int(round(v)) for pt in poly for v in pt[:2]]
        else:
            flat = [int(round(v)) for v in poly]
        if len(flat) != 8:
            continue
        out.append(flat)
    return out


def write_pred(out_path: Path, items: List[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Per-combo orchestration
# ---------------------------------------------------------------------------

def process_combo(combo: str, det: str, rec: str, *,
                  cli: str, results_dir: Path, configs_dir: Path,
                  cls_cfg: Optional[Path], backend: str, threads: int,
                  langs: Optional[List[str]] = None,
                  jobs: int = 1) -> dict:
    """Process one (det__rec) combo: for every language (or a subset) run all images."""
    det_cfg, rec_cfg = resolve_configs(det, rec, configs_dir)
    out: dict = {"combo": combo, "det": det, "rec": rec, "langs": {}}
    if langs is None:
        langs = sorted(p.name for p in IMG_ROOT.iterdir() if p.is_dir())
    for lang in langs:
        img_dir = IMG_ROOT / lang
        if not img_dir.is_dir():
            continue
        images = list_images(lang)
        if not images:
            continue
        items: List[dict] = []
        ok_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = {ex.submit(_run_one, cli, image=im, det_cfg=det_cfg,
                              rec_cfg=rec_cfg, cls_cfg=cls_cfg, backend=backend,
                              threads=threads): im for im in images}
            for fut in concurrent.futures.as_completed(futs):
                im = futs[fut]
                r = fut.result()
                if r is None:
                    items.append({"image_path": str(im), "rec_texts": [],
                                  "rec_scores": [], "det_polys": [],
                                  "error": "cli_failed"})
                    continue
                lines = r.get("lines", [])
                rec_texts = [ln.get("text", "") for ln in lines]
                rec_scores = [float(ln.get("score", 0.0)) for ln in lines]
                det_polys = _poly_to_baseline(lines)
                items.append({
                    "image_path": str(im),
                    "rec_texts": rec_texts,
                    "rec_scores": rec_scores,
                    "det_polys": det_polys,
                })
                ok_count += 1
        # Stable order by image_path
        items.sort(key=lambda x: x["image_path"])
        out_path = results_dir / combo / lang / "pred.json"
        write_pred(out_path, items)
        out["langs"][lang] = {"count": len(items), "ok": ok_count,
                              "pred_path": str(out_path)}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="run_reference")
    ap.add_argument("--cli", default="./build-tools/ppocr_cli",
                    help="path to ppocr_cli binary")
    ap.add_argument("--configs-dir", default=str(Path(__file__).resolve().parent.parent / "configs"))
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--cls-config", default=None,
                    help="optional cls config (textline orientation)")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4,
                    help="per-combo parallel images; outer combos run serially")
    ap.add_argument("--cells", default=None,
                    help="regex filter for combo dir names")
    ap.add_argument("--langs", default=None,
                    help="comma-separated language subset (default: all)")
    ap.add_argument("--only-combo", action="append", default=[],
                    help="restrict to combo name (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan, do not execute")
    args = ap.parse_args(argv)

    configs_dir = Path(args.configs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cls_cfg = Path(args.cls_config) if args.cls_config else None
    if cls_cfg and not cls_cfg.exists():
        print(f"ERROR: cls-config not found: {cls_cfg}", file=sys.stderr)
        return 2
    if not Path(args.cli).exists() and not args.dry_run:
        print(f"ERROR: cli not found: {args.cli}", file=sys.stderr)
        return 2

    combos = discover_combos(args.cells)
    if args.only_combo:
        wanted = set(args.only_combo)
        combos = [c for c in combos if c[0] in wanted]
    if not combos:
        print("no combos to run", file=sys.stderr)
        return 0
    langs = [s.strip() for s in args.langs.split(",")] if args.langs else None
    if args.dry_run:
        print(f"# combos: {len(combos)}; first 5:")
        for c in combos[:5]:
            print(" ", c)
        return 0

    summary: List[dict] = []
    t0 = time.time()
    for combo, det, rec in combos:
        try:
            s = process_combo(combo, det, rec,
                              cli=args.cli, results_dir=results_dir,
                              configs_dir=configs_dir, cls_cfg=cls_cfg,
                              backend=args.backend, threads=args.threads,
                              langs=langs, jobs=args.jobs)
            summary.append(s)
            ok = sum(1 for v in s["langs"].values() if v.get("ok", 0) > 0)
            total = sum(v["count"] for v in s["langs"].values())
            print(f"[OK]   {combo:60s}  langs={len(s['langs']):2d} imgs={total:4d} ok={ok}")
        except FileNotFoundError as e:
            print(f"[SKIP] {combo:60s}  {e}", file=sys.stderr)
            summary.append({"combo": combo, "error": str(e)})
    print(f"\nTotal: {len(summary)} combos in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
