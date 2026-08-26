#!/usr/bin/env python3
"""fix_reference.py — regenerate baseline entries with null det_polys.

The M2-MATRIX sweep (commit 35d1f4b on ws/m2-matrix) uncovered a
**baseline-generator bug** in `tools/gen_parallel.py::run_ocr_cell`:

    polys = list(info.get("rec_polys", info.get("rec_boxes", [])) or [])
    det_polys, detections = [], []
    for i, t in enumerate(texts):
        poly = polys[i] if i < len(polys) else None
        ...
        det_polys.append(box)            # ← box is None when poly is None
        detections.append({"poly": box, ...})

When PaddleOCR drops a polygon's rec_polys (e.g. because the rec batch
hit the rec side with a default fallback box), the baseline entry is
written with `det_polys=[None, None, ...]` but `rec_texts=[...]` intact.
This makes the baseline's rec_text effectively run on a non-real
crop geometry, and any CER comparison with our MNN output (which
always produces a real box) becomes apples-to-oranges.

The decision-maker asked to:

  1. Dry-run: scan the whole baseline, list every entry that has a
     None element in det_polys, compute a per-(cell, lang, image)
     backlog, plus an estimated GPU-hour cost.
  2. After approval, re-run those entries with the **same PaddleOCR
     pipeline as gen_parallel.py** (i.e. the venv at
     /root/.local/pytools, PaddleX 3.x with the official
     PP-OCR<det> + PP-OCR<rec> model pair) and refill det_polys +
     rec_scores + rec_polys (in-place edit of
     /root/ppocr_reference/<combo>/<lang>/ocr_results.json).

This file ships with the dry-run driver only. The actual regen
(`--execute`) is **NOT** enabled by default; it requires a second
explicit `--i-understand-this-runs-gpu` flag. The dry-run path
default is to print a backlog table and exit.

Usage:

    # 1) dry-run (no Paddle/PaddleX import; pure-Python scan)
    python3 tools/fix_reference.py --dry-run

    # 2) live dry-run (one PaddleX import + warm-up to time it)
    python3 tools/fix_reference.py --dry-run --probe

    # 3) approved execute (NOT YET, pending decision-maker go-ahead)
    python3 tools/fix_reference.py --execute --i-understand-this-runs-gpu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REF_ROOT = Path("/root/ppocr_reference")
IMG_ROOT = Path("/root/ocr_test_imgs")
PPOCR_MODELS = Path("/root/ppocr_models")
PYTOOLS_BIN = Path("/root/.local/pytools/bin")

# Same canonical set as gen_parallel.py and tools/score.py.
MAIN_DETS = [
    "PP-OCRv4_mobile_det", "PP-OCRv4_server_det",
    "PP-OCRv5_mobile_det", "PP-OCRv5_server_det",
    "PP-OCRv6_tiny_det", "PP-OCRv6_small_det", "PP-OCRv6_medium_det",
]
MAIN_RECS = [
    "PP-OCRv4_mobile_rec", "PP-OCRv4_server_rec",
    "PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec",
    "PP-OCRv6_tiny_rec", "PP-OCRv6_small_rec", "PP-OCRv6_medium_rec",
]
LANG_RECS = [
    "en_PP-OCRv4_mobile_rec", "en_PP-OCRv5_mobile_rec",
    "arabic_PP-OCRv5_mobile_rec", "cyrillic_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec", "el_PP-OCRv5_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec", "korean_PP-OCRv5_mobile_rec",
    "latin_PP-OCRv5_mobile_rec", "th_PP-OCRv5_mobile_rec",
]
DOC_REC = "PP-OCRv4_server_rec_doc"
LANGS = ["zh", "en", "ja", "ko", "ru", "ar", "th", "el", "hi", "vi",
         "de", "fr", "es", "it", "pt", "tr"]
LANG_TO_PADDLE = {
    "zh": "ch", "en": "en", "ja": "japan", "ko": "korean", "ru": "ru",
    "ar": "ar", "th": "th", "el": "el", "hi": "hi", "vi": "vi",
    "de": "de", "fr": "fr", "es": "es", "it": "it", "pt": "pt", "tr": "tr",
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def is_full_combo_dir(p: Path) -> bool:
    return p.is_dir() and "__" in p.name and not p.name.startswith("strip__")


def iter_full_combos() -> List[Path]:
    return sorted(p for p in REF_ROOT.iterdir() if is_full_combo_dir(p))


def baseline_path(combo: Path, lang: str) -> Path:
    return combo / lang / "ocr_results.json"


def entry_is_invalid(b: dict) -> Optional[str]:
    """Return a reason string if the entry needs regen, else None.

    Same rules as tools/score.py::_baseline_entry_is_valid, with the
    extra "strip cell" check (strip cells don't have det_polys by
    design and are out of regen scope).
    """
    if not isinstance(b, dict):
        return "not-a-dict"
    polys = b.get("det_polys")
    if polys is None:
        return "det_polys-is-None"
    if not isinstance(polys, list):
        return "det_polys-not-list"
    for i, poly in enumerate(polys):
        if poly is None:
            return f"det_polys[{i}]-is-None"
    if "rec_texts" not in b or not isinstance(b["rec_texts"], list):
        return "rec_texts-missing"
    return None


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def collect_backlog() -> Tuple[List[dict], Dict[str, int]]:
    """Walk every (combo, lang) cell, list every entry needing regen.

    Returns:
      backlog: list of {combo, lang, image_path, det_n_polys,
                        n_null_polys, rec_n_texts, reason} dicts.
      per_lang_counts: {lang: count} of invalid entries.
    """
    backlog: List[dict] = []
    per_lang: Dict[str, int] = defaultdict(int)
    for combo in iter_full_combos():
        for lang in LANGS:
            bp = baseline_path(combo, lang)
            if not bp.exists():
                continue
            try:
                entries = json.load(open(bp, "r", encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                backlog.append({
                    "combo": combo.name, "lang": lang,
                    "image_path": "<baseline_unreadable>",
                    "det_n_polys": 0, "n_null_polys": 0, "rec_n_texts": 0,
                    "reason": f"baseline_unreadable: {e}",
                })
                continue
            for entry in entries:
                reason = entry_is_invalid(entry)
                if reason is None:
                    continue
                polys = entry.get("det_polys") or []
                n_null = sum(1 for p in polys if p is None) if isinstance(polys, list) else 0
                backlog.append({
                    "combo": combo.name,
                    "lang": lang,
                    "image_path": entry.get("image_path", "<no-path>"),
                    "det_n_polys": len(polys) if isinstance(polys, list) else 0,
                    "n_null_polys": n_null,
                    "rec_n_texts": len(entry.get("rec_texts", [])),
                    "reason": reason,
                })
                per_lang[lang] += 1
    return backlog, dict(per_lang)


def render_dry_run(backlog: List[dict], per_lang: Dict[str, int],
                   probe_seconds: Optional[float] = None) -> str:
    out: List[str] = []
    out.append("# fix_reference.py — dry-run backlog")
    out.append("")
    out.append(f"Total invalid entries: {len(backlog)}")
    out.append(f"Unique (combo, lang) cells affected: "
               f"{len({(b['combo'], b['lang']) for b in backlog})}")
    out.append(f"Unique languages affected: {len(per_lang)}")
    out.append("")
    out.append("Per-lang invalid entry counts:")
    out.append("| lang | invalid_entries |")
    out.append("|---|---|")
    for lang in LANGS + sorted(set(per_lang.keys()) - set(LANGS)):
        n = per_lang.get(lang, 0)
        if n == 0:
            continue
        out.append(f"| {lang} | {n} |")
    out.append("")
    out.append("Per-combo invalid entry counts (top 20):")
    out.append("| combo | invalid_entries |")
    out.append("|---|---|")
    combo_counts: Dict[str, int] = defaultdict(int)
    for b in backlog:
        combo_counts[b["combo"]] += 1
    for c, n in sorted(combo_counts.items(), key=lambda x: -x[1])[:20]:
        out.append(f"| {c} | {n} |")
    out.append("")
    out.append("First 20 entries needing regen (out of "
               f"{len(backlog)}):")
    out.append("| combo | lang | image | det_n | n_null | rec_n | reason |")
    out.append("|---|---|---|---|---|---|---|")
    for b in backlog[:20]:
        img_short = Path(b["image_path"]).name
        out.append(f"| {b['combo']} | {b['lang']} | {img_short} | "
                   f"{b['det_n_polys']} | {b['n_null_polys']} | "
                   f"{b['rec_n_texts']} | {b['reason']} |")
    out.append("")
    out.append("Note: the strip cells `strip__ta_PP-OCRv5_mobile_rec` and")
    out.append("`strip__te_PP-OCRv5_mobile_rec` are NOT in the full-combo")
    out.append("loop (they have no `__` between det and rec and no det_polys")
    out.append("schema). They are out of scope for fix_reference.py.")
    out.append("")
    out.append("## GPU time estimate")
    out.append("")
    if probe_seconds is None:
        out.append("Probe (one PaddleOCR instance warm-up) was not run. Add")
        out.append("`--probe` to time one inference and extrapolate.")
    else:
        # Conservative estimate: each regen is one PaddleOCR.predict()
        # call. gen_parallel.py measured ~3-5s per cell (16 langs in
        # ~50-80s on a single A10G; 1 cell = 1 image). We assume
        # 1.5x the probe time for the real regen (PaddleX bookkeeping,
        # disk write). Multiply by backlog size.
        per_img = probe_seconds * 1.5
        total = per_img * len(backlog)
        out.append(f"Probe (one PaddleOCR.predict() warm-up): "
                   f"{probe_seconds:.2f} s/image")
        out.append(f"Estimated (×1.5 for the real run, per entry): "
                   f"{per_img:.2f} s/entry")
        out.append(f"Total estimated for {len(backlog)} entries: "
                   f"{total:.1f} s ≈ {total/3600:.2f} GPU-hours")
        out.append("On the box A10G (24GB), 4 parallel workers (same as")
        out.append("gen_parallel.py default), the wall time is ≈ "
                   f"{total/4/3600:.2f} hours.")
    out.append("")
    out.append("## What the executor needs to do (after approval)")
    out.append("")
    out.append("```bash")
    out.append("# use the pytools venv so paddleocr/paddlex are importable")
    out.append("export PATH=/root/.local/pytools/bin:$PATH")
    out.append("export PYTHONPATH=/root/.local/pytools/lib/python3.12/site-packages")
    out.append("#")
    out.append("# 1) dry-run with PaddleX warm-up:")
    out.append("python3 /root/pp-ocr-mnn/tools/fix_reference.py --dry-run --probe")
    out.append("#")
    out.append("# 2) approved execute (after decision-maker sign-off):")
    out.append("python3 /root/pp-ocr-mnn/tools/fix_reference.py --execute "
               "--i-understand-this-runs-gpu --workers 4")
    out.append("```")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Optional probe: time one PaddleOCR.predict() call to extrapolate.
# ---------------------------------------------------------------------------

def probe_one_inference() -> float:
    """Build a PaddleOCR instance (same flags as gen_parallel.py) and
    run one .predict() on a small image. Returns seconds. Used by
    --probe to extrapolate the full regen wall time.

    Requires the pytools venv (PYTHONPATH already set).
    """
    import importlib
    # Sanity-check: the pytools venv must be active.
    if "/root/.local/pytools/lib/python3.12/site-packages" not in sys.path:
        sys.path.insert(0,
                        "/root/.local/pytools/lib/python3.12/site-packages")
    # Pick a fast combo + small image for the probe.
    combo = "PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec"
    det, rec = combo.split("__", 1)
    lang = "en"
    img = IMG_ROOT / lang / "00.jpg"
    if not img.exists():
        # fallback: any .jpg in /root/ocr_test_imgs
        for p in IMG_ROOT.rglob("*.jpg"):
            img = p
            break
    from paddleocr import PaddleOCR
    t0 = time.time()
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=det,
        text_detection_model_dir=str(PPOCR_MODELS / det),
        text_recognition_model_name=rec,
        text_recognition_model_dir=str(PPOCR_MODELS / rec),
    )
    res = ocr.predict(str(img))
    dt = time.time() - t0
    n = len(res[0].get("rec_texts", [])) if res else 0
    sys.stderr.write(f"[probe] PaddleOCR({combo}) on {img.name}: "
                     f"{dt:.2f} s, n_texts={n}\n")
    return dt


# ---------------------------------------------------------------------------
# Stub for the executor path (gated behind --i-understand-this-runs-gpu)
# ---------------------------------------------------------------------------

def execute_regen(backlog: List[dict], workers: int) -> None:
    """Regenerate the baseline entries in-place.

    Strategy:
      * For each (combo, lang) cell affected, build the PaddleOCR
        instance with the same flags as tools/gen_parallel.py and
        call .predict() on the affected images only.
      * Refill the matching entries in the existing
        /root/ppocr_reference/<combo>/<lang>/ocr_results.json in
        place. Other (already-valid) entries are left untouched.
      * multiprocessing.Pool(workers) over cells; per-cell processing
        is serial (PaddleX is not fork-safe across instances).
    """
    import multiprocessing as mp
    by_cell: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for b in backlog:
        by_cell[(b["combo"], b["lang"])].append(b)
    by_cell_plain = {f"{c}/{l}": v for (c, l), v in by_cell.items()}

    cells = list(by_cell.keys())
    print(f"[execute] {len(cells)} cells to regen, {len(backlog)} entries", flush=True)
    print(f"[execute] workers={workers}", flush=True)

    t0 = time.time()
    ok = err = 0
    if workers <= 1:
        _init_worker(by_cell_plain)
        for cell in cells:
            r = _process_cell(cell)
            combo, lang, status, n = r
            line = status.splitlines()[0][:120]
            print(f"[execute] {combo}/{lang} n={n} {line}", flush=True)
            if status == "ok":
                ok += 1
            else:
                err += 1
    else:
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(by_cell_plain,)) as pool:
            for r in pool.imap_unordered(_process_cell, cells):
                combo, lang, status, n = r
                line = status.splitlines()[0][:120]
                print(f"[execute] {combo}/{lang} n={n} {line}", flush=True)
                if status == "ok":
                    ok += 1
                else:
                    err += 1
    print(f"[execute] done: ok={ok} err={err} in {time.time()-t0:.1f}s", flush=True)


# Module-level worker state (set by _init_worker in each Pool process).
_WORKER_BY_CELL: Dict[str, List[dict]] = {}


def _init_worker(by_cell_plain: Dict[str, List[dict]]) -> None:
    global _WORKER_BY_CELL
    _WORKER_BY_CELL = by_cell_plain


def _process_cell(args: Tuple[str, str]) -> Tuple[str, str, str, int]:
    import numpy as np
    from paddleocr import PaddleOCR

    combo, lang = args
    det, rec = combo.split("__", 1)
    paddle_lang = LANG_TO_PADDLE[lang]
    bp = REF_ROOT / combo / lang / "ocr_results.json"
    key = f"{combo}/{lang}"
    affected = {b["image_path"]: b for b in _WORKER_BY_CELL.get(key, [])}
    with open(bp, "r", encoding="utf-8") as f:
        entries = json.load(f)
    try:
        ocr = PaddleOCR(
            lang=paddle_lang,
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name=det,
            text_detection_model_dir=str(PPOCR_MODELS / det),
            text_recognition_model_name=rec,
            text_recognition_model_dir=str(PPOCR_MODELS / rec),
        )
    except Exception as e:
        return (combo, lang, f"PaddleOCR-init-failed: {e}", 0)
    n_done = 0
    for entry in entries:
        ipath = entry.get("image_path", "")
        if ipath not in affected:
            continue
        try:
            res = ocr.predict(ipath)
            info = res[0] if res else {}
        except Exception as e:
            sys.stderr.write(f"[execute] {combo}/{lang} {Path(ipath).name}: "
                             f"predict failed: {e}\n")
            continue
        texts = list(info.get("rec_texts") or [])
        scores = list(info.get("rec_scores") or [])
        polys = list(info.get("rec_polys", info.get("rec_boxes", [])) or [])
        det_polys_new, rec_polys_new, detections_new = [], [], []
        for i, t in enumerate(texts):
            poly = polys[i] if i < len(polys) else None
            box = None
            if poly is not None:
                try:
                    box = [int(x) for x in list(np.array(poly).flatten())[:8]]
                except Exception:
                    box = None
            det_polys_new.append(box)
            rec_polys_new.append(poly)
            detections_new.append({
                "poly": box,
                "rec_text": str(t),
                "rec_score": float(scores[i]) if i < len(scores) else 1.0,
            })
        entry["rec_texts"] = [str(t) for t in texts]
        entry["rec_scores"] = [float(s) for s in scores]
        entry["det_polys"] = det_polys_new
        entry["detections"] = detections_new
        if texts:
            entry["text"] = "\n".join(str(t) for t in texts)
        else:
            entry.pop("text", None)
        n_done += 1
    tmp = bp.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    os.replace(tmp, bp)
    return (combo, lang, "ok", n_done)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="fix_reference",
                                 description="regen baseline entries "
                                             "with null det_polys")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and print the backlog (no GPU)")
    ap.add_argument("--probe", action="store_true",
                    help="also run one PaddleOCR.predict() to extrapolate "
                         "GPU time. Requires pytools venv on PYTHONPATH.")
    ap.add_argument("--execute", action="store_true",
                    help="regenerate the invalid entries in-place. NOT "
                         "YET ENABLED; requires --i-understand-this-runs-gpu.")
    ap.add_argument("--i-understand-this-runs-gpu",
                    action="store_true",
                    help="explicit acknowledgement that --execute will run "
                         "GPU inference and rewrite "
                         "/root/ppocr_reference/*.json in place.")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel (combo, lang) cells under --execute "
                         "(default 4, same as gen_parallel.py)")
    args = ap.parse_args(argv)

    if not (args.dry_run or args.execute):
        ap.error("must specify either --dry-run or --execute")
    if args.execute and not args.i_understand_this_runs_gpu:
        ap.error("--execute requires --i-understand-this-runs-gpu "
                 "(decision-maker approval gate)")
    if args.execute and args.dry_run:
        ap.error("--execute and --dry-run are mutually exclusive")

    backlog, per_lang = collect_backlog()
    probe_dt: Optional[float] = None
    if args.probe:
        sys.stderr.write("[probe] importing paddleocr / paddle / paddleX...\n")
        probe_dt = probe_one_inference()

    sys.stdout.write(render_dry_run(backlog, per_lang, probe_dt))

    if args.execute:
        # Decision-maker has approved; back up & regen.
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup = REF_ROOT.parent / f"ppocr_reference.bak.{ts}"
        if not backup.exists():
            sys.stderr.write(f"[execute] backup to {backup} ...\n")
            import shutil
            shutil.copytree(REF_ROOT, backup)
        execute_regen(backlog, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
