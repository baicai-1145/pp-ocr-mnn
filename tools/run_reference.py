#!/usr/bin/env python3
"""run_reference.py — drive ppocr_cli across the PP-OCR reference matrix.

Per docs/CONTRACT.md (CLI section) and the live `apps/ppocr_cli.cpp` in
main, the binary supports these flags:

    --image IMG          (required)
    --det-config PATH    (required, even in det-only mode)
    --rec-config PATH    (optional; if absent → det-only)
    --cls-config PATH    (optional)
    --model-dir DIR      (default ./models)
    --registry PATH      (optional override of registry.json)
    --backend auto|cpu|cuda|opencl|vulkan
    --threads N
    --batch N
    --det-only           (no rec; the rec_name in ppocr_config becomes NULL)
    --max-side N
    --json OUT_PATH      (file path; omit to write JSON to stdout)
    --time               (extra timing to stderr)

We pass these flags verbatim. `--json stdout` is **not** a valid value
(verified by reading apps/ppocr_cli.cpp parse_args(): `--json` takes a
file path and writes to that file; no stdout pseudo-target). To get
JSON on stdout, we omit `--json` and the binary writes to stdout.

What this driver does:
  * walks /root/ppocr_reference/*__*/ (skipping strip__*, manifest*, and
    seal_*__* (no __) dirs);
  * maps `<det>__<rec>` to configs (det = dir before __, rec = after __);
  * for each language under /root/ocr_test_imgs/<lang>/0*.jpg, runs
    ppocr_cli and writes
        results/<combo>/<lang>/pred.json
    with the same schema as the baseline (minus GT fields):
        [{"image_path":..., "rec_texts":[...], "rec_scores":[...],
          "det_polys":[[8 floats],...]}]
  * --rec-only (planned for strip cells): NOT YET SUPPORTED by the CLI;
    see REC_ONLY.md in this dir for the contract gap and the patch
    request to m1.
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


def discover_combos(cells_re: Optional[str] = None,
                    include_strip: bool = False) -> List[Tuple[str, str, str]]:
    """Return list of (combo_dir_name, det_name, rec_name) tuples.

    Skips:
      * dirs without '__' (seal_det, manifest*.json are flat files anyway)
      * dirs starting with 'strip__' (unless include_strip=True)

    strip__* combos have an empty det name and a single rec name. They are
    processed only under --rec-only mode. The directory layout per
    /root/ppocr_reference is:
      strip__<rec>/<lang>/ocr_results.json   (lang ∈ {ta, te})
    """
    out: List[Tuple[str, str, str]] = []
    for child in sorted(REF_ROOT.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if "__" not in name:
            continue
        det, rec = name.split("__", 1)
        if det == "strip":
            if not include_strip:
                continue
            det_name = ""  # rec-only: no det
        else:
            det_name = det
        if cells_re and not re.search(cells_re, name):
            continue
        out.append((name, det_name, rec))
    return out


def list_images(lang: str, rec_only: bool = False) -> List[Path]:
    d = IMG_ROOT / lang
    if not d.is_dir():
        return []
    if rec_only:
        # Strip cells use ta/te/00_0.jpg style (no 0*.jpg prefix).
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return sorted(p for p in d.iterdir()
                  if p.is_file()
                  and p.suffix.lower() in (".jpg", ".jpeg", ".png"))


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


def resolve_rec_config(rec: str, configs_dir: Path) -> Path:
    r = configs_dir / f"{rec}.json"
    if not r.exists():
        raise FileNotFoundError(f"missing rec config: {r}")
    return r


# ---------------------------------------------------------------------------
# Per-image run
# ---------------------------------------------------------------------------

def _build_cli_cmd(*, cli: str, image: Path, det_cfg: Optional[Path],
                   rec_cfg: Optional[Path], cls_cfg: Optional[Path],
                   backend: str, threads: int, batch: int,
                   json_path: Optional[Path]) -> List[str]:
    """Build a ppocr_cli argv that matches apps/ppocr_cli.cpp exactly.

    Flag alignment (verified against parse_args() in apps/ppocr_cli.cpp):
      --image IMG          (required)
      --det-config PATH    (required)
      --rec-config PATH    (optional; omit to disable rec → det-only)
      --cls-config PATH    (optional)
      --backend NAME
      --threads N
      --batch N
      --json PATH          (file path; omit → stdout)
    """
    cmd: List[str] = [cli, "--image", str(image), "--backend", backend,
                      "--threads", str(threads)]
    if det_cfg is not None:
        cmd += ["--det-config", str(det_cfg)]
    if rec_cfg is not None:
        cmd += ["--rec-config", str(rec_cfg)]
    if cls_cfg is not None:
        cmd += ["--cls-config", str(cls_cfg)]
    if batch > 0:
        cmd += ["--batch", str(batch)]
    if json_path is not None:
        cmd += ["--json", str(json_path)]
    return cmd


def _run_one(cli: str, *, image: Path, det_cfg: Path, rec_cfg: Path,
             cls_cfg: Optional[Path], backend: str, threads: int,
             batch: int = 0) -> Optional[dict]:
    """Run the CLI on one image, return parsed JSON dict (or None on error).

    MNN-2.9.x's Interpreter constructor prints a 3-line device-support
    banner ("The device support i8sdot:0, support fp16:0, support i8mm: 0"
    + an "OpenCL init error, fallback ..." line + an "Error to use creator
    of 3, delete it" line) to **stdout** before the JSON when running on
    a CPU-only box that lacks OpenCL/Metal. Capturing the JSON from
    stdout therefore needs a parse tolerant of leading garbage. The
    CLI also supports `--json PATH` (a file) which writes a clean JSON
    file; we use that to avoid the banner entirely.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as f:
        json_path = Path(f.name)
    try:
        cmd = _build_cli_cmd(cli=cli, image=image, det_cfg=det_cfg,
                             rec_cfg=rec_cfg, cls_cfg=cls_cfg, backend=backend,
                             threads=threads, batch=batch, json_path=json_path)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[run] timeout img={image}\n")
            return None
        if p.returncode != 0:
            sys.stderr.write(f"[run] rc={p.returncode} img={image} stderr={p.stderr[:200]}\n")
            return None
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            sys.stderr.write(f"[run] bad-json img={image} err={e}\n")
            return None
    finally:
        try:
            json_path.unlink()
        except OSError:
            pass


def _run_one_rec(cli: str, *, image: Path, rec_cfg: Path,
                 backend: str, threads: int) -> Optional[dict]:
    """Rec-only mode. NOT YET SUPPORTED by the real CLI: there is no
    `--rec-only` flag and the engine always wants a det config (ppocr_config
    requires det_name). This stub invokes the CLI the same way the user
    asked, and falls back to writing a synthetic error entry so the
    pipeline stays testable end-to-end.

    See tools/REC_ONLY.md for the contract gap and the patch request.
    """
    cmd = [cli, "--image", str(image), "--rec-config", str(rec_cfg),
           "--backend", backend, "--threads", str(threads)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if p.returncode == 0:
        try:
            return json.loads(p.stdout)
        except json.JSONDecodeError:
            pass
    # Real CLI doesn't accept --rec-config without --det-config; emit a
    # well-formed error entry so the downstream score.py can still run.
    return {
        "error": "rec_only_unsupported",
        "detail": (f"ppocr_cli does not support --rec-only (rc={p.returncode}). "
                   f"See tools/REC_ONLY.md. stderr={p.stderr[:160]}"),
    }


# ---------------------------------------------------------------------------
# Result writers
# ---------------------------------------------------------------------------

def _poly_to_baseline(lines: list) -> list:
    """CLI emits `lines=[{"poly":[8 ints],"text":...,"score":...}]`.
    baseline uses `det_polys: [[x0,y0,...,x3,y3], ...]`. Flatten to 8 ints."""
    out = []
    for ln in lines or []:
        poly = ln.get("poly")
        if poly is None:
            continue
        if not poly:
            continue
        # Accept 4 [x,y] pairs or a flat list of 8.
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
                  batch: int = 0,
                  langs: Optional[List[str]] = None,
                  jobs: int = 1,
                  rec_only: bool = False) -> dict:
    """Process one (det__rec) combo: for every language (or a subset) run all images.

    If rec_only=True, the cli flag is --rec-only (not yet supported; emits
    a synthetic error entry per image; see _run_one_rec).
    """
    out: dict = {"combo": combo, "det": det, "rec": rec, "langs": {},
                 "rec_only": rec_only}
    if langs is None:
        # Default: skip seal/ta/te in normal mode; --rec-only restricts to ta+te
        all_dirs = sorted(p.name for p in IMG_ROOT.iterdir() if p.is_dir())
        if rec_only:
            langs = [d for d in all_dirs if d in ("ta", "te")]
        else:
            langs = [d for d in all_dirs if d not in ("seal", "ta", "te")]
    if rec_only:
        rec_cfg = resolve_rec_config(rec, configs_dir)
    else:
        det_cfg, rec_cfg = resolve_configs(det, rec, configs_dir)
    for lang in langs:
        img_dir = IMG_ROOT / lang
        if not img_dir.is_dir():
            continue
        images = list_images(lang, rec_only=rec_only)
        if not images:
            continue
        items: List[dict] = []
        ok_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            if rec_only:
                futs = {ex.submit(_run_one_rec, cli, image=im, rec_cfg=rec_cfg,
                                  backend=backend, threads=threads): im
                        for im in images}
            else:
                futs = {ex.submit(_run_one, cli, image=im, det_cfg=det_cfg,
                                  rec_cfg=rec_cfg, cls_cfg=cls_cfg,
                                  backend=backend, threads=threads,
                                  batch=batch): im for im in images}
            for fut in concurrent.futures.as_completed(futs):
                im = futs[fut]
                r = fut.result()
                if r is None:
                    items.append({"image_path": str(im), "rec_texts": [],
                                  "rec_scores": [], "det_polys": [],
                                  "error": "cli_failed"})
                    continue
                if "error" in r:
                    items.append({"image_path": str(im), "rec_texts": [],
                                  "rec_scores": [], "det_polys": [], **r})
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
# Dry-run summary
# ---------------------------------------------------------------------------

def _lang_image_counts(rec_only: bool = False) -> Dict[str, int]:
    """Return {lang: image_count} using the same rules as list_images().

    In normal mode we exclude `seal` (handled by M4) and `ta`/`te` (those
    are the strip cells, processed by --rec-only; they share a different
    image set). In --rec-only mode we *include* ta and te and exclude
    everything else.
    """
    out: Dict[str, int] = {}
    EXCLUDE_NORMAL = {"seal", "ta", "te"}
    for lang_dir in sorted(IMG_ROOT.iterdir()):
        if not lang_dir.is_dir():
            continue
        name = lang_dir.name
        if rec_only:
            if name not in ("ta", "te"):
                continue
        else:
            if name in EXCLUDE_NORMAL:
                continue
        out[name] = sum(
            1 for p in lang_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return out


def render_dry_run(combos: List[Tuple[str, str, str]], langs_filter: Optional[List[str]],
                   rec_only: bool) -> str:
    counts = _lang_image_counts(rec_only=rec_only)
    if langs_filter is None:
        langs = sorted(counts.keys())
    else:
        langs = langs_filter
    per_combo: List[Tuple[str, int]] = []
    for combo, det, rec in combos:
        try:
            if rec_only:
                resolve_rec_config(rec, Path("configs"))
            else:
                resolve_configs(det, rec, Path("configs"))
        except FileNotFoundError:
            per_combo.append((combo, -1))  # missing config
            continue
        per_combo.append((combo, sum(counts.get(l, 0) for l in langs)))
    total_imgs = sum(n for _, n in per_combo if n >= 0)
    out: List[str] = []
    out.append("# run_reference dry-run")
    out.append("")
    out.append(f"rec_only: {rec_only}")
    out.append(f"total combos: {len(combos)}  (of which --only-combo/dryrun filter)")
    out.append(f"total images to process: {total_imgs}")
    out.append("")
    # Group by combo kind
    STDLIB_DETS = {"PP-OCRv4_mobile_det", "PP-OCRv4_server_det",
                   "PP-OCRv5_mobile_det", "PP-OCRv5_server_det",
                   "PP-OCRv6_tiny_det", "PP-OCRv6_small_det",
                   "PP-OCRv6_medium_det"}
    STDLIB_RECS = {"PP-OCRv4_mobile_rec", "PP-OCRv4_server_rec",
                   "PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec",
                   "PP-OCRv6_tiny_rec", "PP-OCRv6_small_rec",
                   "PP-OCRv6_medium_rec"}
    strip = [c for c in combos if c[0].startswith("strip__")]
    core = [c for c in combos if c[1] in STDLIB_DETS and c[2] in STDLIB_RECS]
    doc = [c for c in combos if c[2] == "PP-OCRv4_server_rec_doc"]
    lang_rec = [c for c in combos if c not in core and c not in doc and c not in strip]
    out.append(f"combo breakdown:")
    out.append(f"  core (7x7 dets x 7 recs): {len(core)}")
    out.append(f"  doc-rec (PP-OCRv4_server_rec_doc): {len(doc)}")
    out.append(f"  lang-rec: {len(lang_rec)}")
    out.append(f"  strip (only under --rec-only): {len(strip)}")
    out.append(f"  TOTAL: {len(combos)}  (= 60 normal + 2 strip = 62 in /root/ppocr_reference)")
    out.append("")
    out.append("per-language image counts:")
    out.append("| lang | images |")
    out.append("|---|---|")
    for l in langs:
        out.append(f"| {l} | {counts.get(l, 0)} |")
    out.append("")
    out.append(f"per-combo image count (sum across selected langs):")
    out.append("| combo | imgs |")
    out.append("|---|---|")
    for name, n in per_combo:
        out.append(f"| {name} | {n if n >= 0 else 'MISSING CONFIG'} |")
    out.append("")
    out.append("CLI flag set that run_reference.py will use (verified against")
    out.append("apps/ppocr_cli.cpp parse_args()):")
    out.append("  --image PATH")
    out.append("  --det-config PATH        (required by CLI)")
    out.append("  --rec-config PATH        (omitted in --rec-only mode; CLI")
    out.append("                            rejects this without --det-config;")
    out.append("                            see tools/REC_ONLY.md)")
    out.append("  --cls-config PATH        (when --cls-config given)")
    out.append("  --backend auto|cpu|cuda|opencl|vulkan")
    out.append("  --threads N")
    out.append("  --batch N                (when --batch > 0)")
    out.append("  --json PATH              (omitted → stdout)")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="run_reference",
                                 description="drive ppocr_cli across the "
                                             "PP-OCR reference matrix")
    ap.add_argument("--cli", default="/root/pp-ocr-mnn/build-main/ppocr_cli",
                    help="path to ppocr_cli binary")
    ap.add_argument("--configs-dir",
                    default=str(Path(__file__).resolve().parent.parent / "configs"))
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--cls-config", default=None,
                    help="optional cls config (textline orientation)")
    ap.add_argument("--backend", default="cpu",
                    help="auto|cpu|cuda|opencl|vulkan (default: cpu)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=0,
                    help="rec batch size (0 = default 8 inside the engine)")
    ap.add_argument("--jobs", type=int, default=4,
                    help="per-combo parallel images; combos themselves run serially")
    ap.add_argument("--cells", default=None,
                    help="regex filter for combo dir names")
    ap.add_argument("--langs", default=None,
                    help="comma-separated language subset (default: all)")
    ap.add_argument("--only-combo", action="append", default=[],
                    help="restrict to combo name (repeatable)")
    ap.add_argument("--rec-only", action="store_true",
                    help="rec-only mode (no det); CLI does NOT support this "
                         "yet; see tools/REC_ONLY.md. Implies --include-strip.")
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

    combos = discover_combos(args.cells, include_strip=args.rec_only)
    # Under --rec-only, the normal det+rec combos do not apply (they need
    # det). Restrict to strip__ cells unless the user explicitly opted
    # in via --cells.
    if args.rec_only and not args.cells:
        combos = [c for c in combos if c[0].startswith("strip__")]
    if args.only_combo:
        wanted = set(args.only_combo)
        combos = [c for c in combos if c[0] in wanted]
    if not combos:
        print("no combos to run", file=sys.stderr)
        return 0
    langs = [s.strip() for s in args.langs.split(",")] if args.langs else None

    if args.dry_run:
        sys.stdout.write(render_dry_run(combos, langs, args.rec_only))
        return 0

    summary: List[dict] = []
    t0 = time.time()
    for combo, det, rec in combos:
        try:
            s = process_combo(combo, det, rec,
                              cli=args.cli, results_dir=results_dir,
                              configs_dir=configs_dir, cls_cfg=cls_cfg,
                              backend=args.backend, threads=args.threads,
                              batch=args.batch,
                              langs=langs, jobs=args.jobs,
                              rec_only=args.rec_only)
            summary.append(s)
            ok = sum(1 for v in s["langs"].values() if v.get("ok", 0) > 0)
            total = sum(v["count"] for v in s["langs"].values())
            tag = "REC-ONLY" if args.rec_only else "OK"
            print(f"[{tag:7s}] {combo:60s}  langs={len(s['langs']):2d} "
                  f"imgs={total:4d} ok={ok}")
        except FileNotFoundError as e:
            print(f"[SKIP] {combo:60s}  {e}", file=sys.stderr)
            summary.append({"combo": combo, "error": str(e)})
    print(f"\nTotal: {len(summary)} combos in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
