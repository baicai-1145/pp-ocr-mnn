#!/usr/bin/env python3
"""score.py — CER scoring for the PP-OCR reference matrix.

Per docs/CONTRACT.md:
  * For every (det__rec) cell: per image, CER = levenshtein(pred_join, base_join)
    / len(base_join) where join = "\\n".join(rec_texts); cell score = mean over
    images; PASS if cell_score ≤ 0.05.
  * --strip mode scores `strip__<rec>` cells against
    /root/ocr_test_imgs/strip_gt.json (per-image gt).
  * seal cells (PP-OCRv4_mobile_seal_det/, PP-OCRv4_server_seal_det/) are out
    of scope (M4) — reported as N/A.

Output (default `results/report.md`):
  * **Main matrix** (7×7): rows = det, cols = rec, cell = lang-averaged CER
    + PASS/FAIL. This is the canonical 49-cell 7×7 view the decision-maker
    needs at a glance.
  * **Lang-rec block**: per-cell CER for the 10 lang-specific rec models
    (each row = det__<lang>_PP-OCRv5_mobile_rec).
  * **Doc-rec block**: PP-OCRv4_server_rec_doc cells.
  * **Strip block**: 2 strip cells (ta, te).
  * **Seal block**: M4 stub, N/A.
  * Exit code: 0 if all PASS, 1 if any FAIL (NaN never trips the gate).

Levenshtein is a self-implemented O(m*n) DP (no numpy dep).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REF_ROOT = Path("/root/ppocr_reference")
STRIP_GT = Path("/root/ocr_test_imgs/strip_gt.json")

# Canonical dets / recs for the 7×7 main matrix (per AGENTS.md).
MAIN_DETS: List[str] = [
    "PP-OCRv4_mobile_det", "PP-OCRv4_server_det",
    "PP-OCRv5_mobile_det", "PP-OCRv5_server_det",
    "PP-OCRv6_tiny_det", "PP-OCRv6_small_det", "PP-OCRv6_medium_det",
]
MAIN_RECS: List[str] = [
    "PP-OCRv4_mobile_rec", "PP-OCRv4_server_rec",
    "PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec",
    "PP-OCRv6_tiny_rec", "PP-OCRv6_small_rec", "PP-OCRv6_medium_rec",
]
DOC_REC = "PP-OCRv4_server_rec_doc"
LANG_RECS: List[str] = [
    "en_PP-OCRv4_mobile_rec", "en_PP-OCRv5_mobile_rec",
    "arabic_PP-OCRv5_mobile_rec", "cyrillic_PP-OCRv5_mobile_rec",
    "devanagari_PP-OCRv5_mobile_rec", "el_PP-OCRv5_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec", "korean_PP-OCRv5_mobile_rec",
    "latin_PP-OCRv5_mobile_rec", "th_PP-OCRv5_mobile_rec",
]


# ---------------------------------------------------------------------------
# Levenshtein (no numpy)
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Standard Wagner–Fischer DP. O(m*n) time, O(min(m,n)) space."""
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
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,      # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, cur = cur, prev
    return prev[n]


def cer(pred: str, base: str) -> float:
    if not base:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, base) / len(base)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_pred(pred_path: Path) -> List[dict]:
    with open(pred_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline(base_path: Path) -> List[dict]:
    with open(base_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_strip_gt(gt_path: Path) -> Dict[str, str]:
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Strip schema validation
# ---------------------------------------------------------------------------

REQUIRED_STRIP_KEYS = ("image_path", "rec_texts", "rec_scores", "gt_text")
REQUIRED_FULL_KEYS = ("image_path", "rec_texts", "rec_scores", "det_polys")


def validate_strip_entry(entry: dict) -> Optional[str]:
    """Return an error string if the strip entry is malformed, else None.

    The strip baseline schema (per /root/ppocr_reference/strip__ta_PP-OCRv5_mobile_rec/ta/ocr_results.json)
    is:
        {image_path: str, rec_texts: [str, ...], rec_scores: [float, ...],
         gt_text: str}
    """
    for k in REQUIRED_STRIP_KEYS:
        if k not in entry:
            return f"missing key: {k}"
    if not isinstance(entry["rec_texts"], list):
        return "rec_texts not a list"
    if not isinstance(entry["rec_scores"], list):
        return "rec_scores not a list"
    if len(entry["rec_texts"]) != len(entry["rec_scores"]):
        return ("rec_texts/rec_scores length mismatch: "
                f"{len(entry['rec_texts'])} vs {len(entry['rec_scores'])}")
    if not isinstance(entry["gt_text"], str):
        return "gt_text not a string"
    return None


def validate_full_entry(entry: dict) -> Optional[str]:
    for k in REQUIRED_FULL_KEYS:
        if k not in entry:
            return f"missing key: {k}"
    return None


# ---------------------------------------------------------------------------
# Per-cell scoring
# ---------------------------------------------------------------------------

def _join_texts(d: dict) -> str:
    return "\n".join(d.get("rec_texts", []) or [])


def _baseline_entry_is_valid(b: dict) -> Tuple[bool, str]:
    """Decide whether a baseline entry is usable for CER scoring.

    A baseline entry is considered **invalid** (counted as N/A) if any
    element of `det_polys` is None, or if `det_polys` is missing entirely,
    or if `rec_texts` is missing or not a list. These three conditions all
    indicate that `gen_parallel.py::run_ocr_cell` (the PaddleX reference
    generator) wrote the entry but the PaddleOCR pipeline dropped one or
    more polygons — usually because the rec side did not get a valid crop
    geometry. Scoring against such an entry would be apples-to-oranges:
    our MNN run produces a rec_text per real box, the baseline was emitted
    from a default or fallback crop.

    The 1 103 entries with at least one null det_polys element
    (concentrated in ru 64 %, th 49 %, ar 27 %, en 18 %, hi 14 %, pt 13 %,
    plus all 80 strip cells) would inflate the CER noise floor. Marking
    them as N/A is the correct response until M2-BASELINE-REGEN regenerates
    them with a faithful Paddle re-run (see tools/M2_BASELINE_REGEN.md).
    """
    if not isinstance(b, dict):
        return False, "baseline entry is not a dict"
    polys = b.get("det_polys")
    if polys is None:
        return False, "det_polys is None"
    if not isinstance(polys, list):
        return False, "det_polys is not a list"
    if any(p is None for p in polys):
        return False, "det_polys contains None element(s)"
    if "rec_texts" not in b or not isinstance(b["rec_texts"], list):
        return False, "rec_texts missing or not a list"
    return True, ""


def score_image(pred_rec_texts: List[str], base: str) -> float:
    pred = "\n".join(pred_rec_texts or [])
    return cer(pred, base)


def score_full_cell(combo_dir: Path, lang: str, results_dir: Path
                    ) -> Tuple[float, int, int, int]:
    """Score one (combo, lang) cell.

    Returns (mean_cer, n_scored, n_invalid_baseline, n_missing_pred).
    - mean_cer: average CER across the images whose baseline entry is
      valid AND whose pred.json has a matching row. NaN if no images
      were scoreable.
    - n_scored: number of images that contributed to mean_cer.
    - n_invalid_baseline: number of baseline entries marked invalid
      (det_polys has a None element, or rec_texts missing, etc.). These
      are NOT counted in n_scored; they are the baseline-regen backlog
      and a per-cell diagnostic is emitted to stderr.
    - n_missing_pred: number of baseline entries that have a corresponding
      image_path but the pred.json row is missing (cli error, OOM, etc.).
    """
    base = load_baseline(combo_dir / lang / "ocr_results.json")
    pred_path = results_dir / combo_dir.name / lang / "pred.json"
    if not pred_path.exists():
        # Whole cell missing → report NaN, 0 scored, but keep the invalid
        # count so the decision-maker can still see the backlog.
        n_invalid = sum(1 for b in base if not _baseline_entry_is_valid(b)[0])
        return float("nan"), 0, n_invalid, 0
    pred = load_pred(pred_path)
    pred_by_path: Dict[str, List[str]] = {}
    for p in pred:
        pred_by_path[p["image_path"]] = p.get("rec_texts", [])
    scores: List[float] = []
    n_invalid = 0
    n_missing_pred = 0
    for b in base:
        valid, _reason = _baseline_entry_is_valid(b)
        if not valid:
            n_invalid += 1
            continue
        btext = _join_texts(b)
        ipath = b.get("image_path", "")
        if ipath not in pred_by_path:
            n_missing_pred += 1
            continue
        ptexts = pred_by_path[ipath]
        scores.append(score_image(ptexts, btext))
    if not scores:
        return float("nan"), 0, n_invalid, n_missing_pred
    return sum(scores) / len(scores), len(scores), n_invalid, n_missing_pred


def score_full_cell_lang_avg(combo_dir: Path, results_dir: Path
                             ) -> Tuple[float, int, int, int]:
    """Mean over languages. Each cell has one number; aggregates the
    invalid/missing counts across langs so a single diagnostic is enough.
    """
    scores: List[float] = []
    n_scored = 0
    n_invalid = 0
    n_missing = 0
    for lang_dir in sorted(combo_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        c, ns, ni, nm = score_full_cell(combo_dir, lang_dir.name, results_dir)
        if c == c:
            scores.append(c)
        n_scored += ns
        n_invalid += ni
        n_missing += nm
    if not scores:
        return float("nan"), 0, n_invalid, n_missing
    return sum(scores) / len(scores), n_scored, n_invalid, n_missing


def score_strip_cell(combo_dir: Path, lang: str, results_dir: Path,
                     strip_gt: Dict[str, str]) -> Tuple[float, int, List[str]]:
    """Per-lang score for a strip cell. Returns (cer, n, warnings)."""
    pred_path = results_dir / combo_dir.name / lang / "pred.json"
    if not pred_path.exists():
        return float("nan"), 0, [f"missing pred: {pred_path}"]
    pred = load_pred(pred_path)
    # Validate pred schema (must include image_path + rec_texts).
    warnings: List[str] = []
    for i, p in enumerate(pred):
        if "image_path" not in p or "rec_texts" not in p:
            warnings.append(f"pred[{i}] missing keys")
    base_path = combo_dir / lang / "ocr_results.json"
    base = load_baseline(base_path) if base_path.exists() else []
    base_by_path: Dict[str, dict] = {b["image_path"]: b for b in base}
    scores: List[float] = []
    for p in pred:
        ipath = p["image_path"]
        b = base_by_path.get(ipath)
        if b is not None:
            err = validate_strip_entry(b)
            if err is not None:
                warnings.append(f"baseline[{Path(ipath).name}]: {err}")
        gt = strip_gt.get(ipath, "")
        scores.append(score_image(p.get("rec_texts", []), gt))
    if not scores:
        return float("nan"), 0, warnings
    return sum(scores) / len(scores), len(scores), warnings


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "N/A"
    return f"{x:.4f}"


def _status(x: float, thr: float = 0.05) -> str:
    if x != x:
        return "N/A"
    return "PASS" if x <= thr else "FAIL"


def _matrix_7x7(det_to_rec_cer: Dict[Tuple[str, str], Tuple[float, int]],
                threshold: float) -> str:
    """Render a 7x7 main matrix: rows=det, cols=rec; cell=CER (status)."""
    lines: List[str] = []
    lines.append("| det \\\\ rec | " + " | ".join(MAIN_RECS) + " |")
    lines.append("|" + "|".join(["---"] * (len(MAIN_RECS) + 1)) + "|")
    for det in MAIN_DETS:
        row = [det]
        for rec in MAIN_RECS:
            cer, _n = det_to_rec_cer.get((det, rec), (float("nan"), 0))
            row.append(f"{_fmt(cer)} {_status(cer, threshold)}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _summary(rows: List[dict], threshold: float) -> Tuple[int, int, int]:
    pass_n = fail_n = nan_n = 0
    for r in rows:
        c = r.get("cer", float("nan"))
        if c != c:
            nan_n += 1
        elif c <= threshold:
            pass_n += 1
        else:
            fail_n += 1
    return pass_n, fail_n, nan_n


def render_report(*, det_to_rec_cer: Dict[Tuple[str, str], Tuple[float, int]],
                  lang_rec_rows: List[dict], doc_rows: List[dict],
                  strip_rows: List[dict], seal_rows: List[dict],
                  threshold: float, warnings: List[str]) -> Tuple[str, bool]:
    """Build the markdown report. Returns (md, has_fail)."""
    lines: List[str] = []
    lines.append("# PP-OCR CER report")
    lines.append("")
    lines.append(f"Threshold: CER ≤ {threshold}")
    lines.append("")

    # ---- Main 7x7 matrix -------------------------------------------------
    if det_to_rec_cer:
        lines.append("## Main matrix (7×7) — lang-averaged CER")
        lines.append("")
        lines.append("Rows = `det`, cols = `rec`. Cell = mean CER across the 16 "
                     "languages. Status: PASS / FAIL.")
        lines.append("")
        lines.append(_matrix_7x7(det_to_rec_cer, threshold))
        # Flatten for summary
        flat: List[dict] = []
        for (det, rec), (cer_, n) in det_to_rec_cer.items():
            flat.append({"combo": f"{det}__{rec}", "cer": cer_, "n": n})
        pass_n, fail_n, nan_n = _summary(flat, threshold)
        lines.append(f"**Main matrix (49 cells):** PASS={pass_n}  FAIL={fail_n}  N/A={nan_n}")
        lines.append("")
    else:
        flat = []

    # ---- Lang-rec block --------------------------------------------------
    if lang_rec_rows:
        lines.append("## Lang-rec block (per-cell CER)")
        lines.append("")
        lines.append("| combo | langs scored | mean CER | status |")
        lines.append("|---|---|---|---|")
        for r in lang_rec_rows:
            lines.append(f"| {r['combo']} | {r['n_langs']} | {_fmt(r['cer'])} | "
                         f"{_status(r['cer'], threshold)} |")
        flat += lang_rec_rows
        lines.append("")

    # ---- Doc block -------------------------------------------------------
    if doc_rows:
        lines.append("## Doc-rec block (PP-OCRv4_server_rec_doc)")
        lines.append("")
        lines.append("| combo | langs scored | mean CER | status |")
        lines.append("|---|---|---|---|")
        for r in doc_rows:
            lines.append(f"| {r['combo']} | {r['n_langs']} | {_fmt(r['cer'])} | "
                         f"{_status(r['cer'], threshold)} |")
        flat += doc_rows
        lines.append("")

    # ---- Strip block -----------------------------------------------------
    if strip_rows:
        lines.append("## Strip rec-only (ta/te)")
        lines.append("")
        lines.append("| combo | lang | imgs | CER | status |")
        lines.append("|---|---|---|---|---|")
        for r in strip_rows:
            lines.append(f"| {r['combo']} | {r['lang']} | {r['n']} | "
                         f"{_fmt(r['cer'])} | {_status(r['cer'], threshold)} |")
        flat += strip_rows
        lines.append("")

    # ---- Seal block ------------------------------------------------------
    if seal_rows:
        lines.append("## Seal det (M4 scope — N/A)")
        lines.append("")
        lines.append("| combo | lang | status |")
        lines.append("|---|---|---|")
        for r in seal_rows:
            lines.append(f"| {r['combo']} | {r.get('lang', '-')} | N/A |")
        flat += seal_rows
        lines.append("")

    # ---- Warnings --------------------------------------------------------
    if warnings:
        lines.append("## Schema warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    pass_n, fail_n, nan_n = _summary(flat, threshold)
    has_fail = fail_n > 0
    lines.append("---")
    lines.append(f"**Total:** PASS={pass_n}  FAIL={fail_n}  N/A={nan_n}  "
                 f"(cells={len(flat)})")
    return "\n".join(lines) + "\n", has_fail


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _iter_full_combos() -> List[Path]:
    out: List[Path] = []
    for c in sorted(REF_ROOT.iterdir()):
        if not c.is_dir():
            continue
        n = c.name
        if "__" not in n or n.startswith("strip__"):
            continue
        out.append(c)
    return out


def _iter_strip_combos() -> List[Path]:
    out: List[Path] = []
    for c in sorted(REF_ROOT.iterdir()):
        if not c.is_dir():
            continue
        if c.name.startswith("strip__"):
            out.append(c)
    return out


def _iter_seal() -> List[Path]:
    out: List[Path] = []
    for c in sorted(REF_ROOT.iterdir()):
        if not c.is_dir():
            continue
        if c.name.endswith("seal_det") and "__" not in c.name:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Top-level score functions (return rows + warnings)
# ---------------------------------------------------------------------------

def collect_main_matrix(results_dir: Path) -> Tuple[Dict[Tuple[str, str], Tuple[float, int]],
                                                    List[str]]:
    """Build the 7×7 (det, rec) → (cer, n_imgs) map using lang-averaged CER."""
    det_to_rec: Dict[Tuple[str, str], Tuple[float, int]] = {}
    warnings: List[str] = []
    for combo_dir in _iter_full_combos():
        n = combo_dir.name
        if "__" not in n:
            continue
        det, rec = n.split("__", 1)
        if det not in MAIN_DETS or rec not in MAIN_RECS:
            continue
        c, n_imgs, n_invalid, n_missing = score_full_cell_lang_avg(
            combo_dir, results_dir)
        det_to_rec[(det, rec)] = (c, n_imgs)
        if n_invalid or n_missing:
            warnings.append(
                f"{n}: {n_invalid} invalid-baseline + {n_missing} missing-pred "
                f"(backlog for M2-BASELINE-REGEN; see tools/M2_BASELINE_REGEN.md)"
            )
    return det_to_rec, warnings


def _count_langs_in_cell(combo_dir: Path) -> int:
    """Number of language subdirs under a combo (zh, en, ja, ...)."""
    return sum(1 for p in combo_dir.iterdir() if p.is_dir())


def collect_lang_rec_block(results_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for combo_dir in _iter_full_combos():
        n = combo_dir.name
        if "__" not in n:
            continue
        det, rec = n.split("__", 1)
        if rec not in LANG_RECS:
            continue
        c, n_imgs, _n_invalid, _n_missing = score_full_cell_lang_avg(
            combo_dir, results_dir)
        n_langs = _count_langs_in_cell(combo_dir)
        rows.append({"combo": n, "cer": c, "n_langs": n_langs})
    return rows


def collect_doc_block(results_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for combo_dir in _iter_full_combos():
        n = combo_dir.name
        if "__" not in n:
            continue
        det, rec = n.split("__", 1)
        if rec != DOC_REC:
            continue
        c, n_imgs, _n_invalid, _n_missing = score_full_cell_lang_avg(
            combo_dir, results_dir)
        n_langs = _count_langs_in_cell(combo_dir)
        rows.append({"combo": n, "cer": c, "n_langs": n_langs})
    return rows


def collect_strip_block(results_dir: Path) -> Tuple[List[dict], List[str]]:
    if not STRIP_GT.exists():
        return [], [f"strip GT not found at {STRIP_GT}"]
    strip_gt = load_strip_gt(STRIP_GT)
    rows: List[dict] = []
    warnings: List[str] = []
    for combo_dir in _iter_strip_combos():
        for lang_dir in sorted(combo_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            c, n, ws = score_strip_cell(combo_dir, lang, results_dir, strip_gt)
            rows.append({"combo": combo_dir.name, "lang": lang, "cer": c, "n": n})
            warnings.extend(ws)
    return rows, warnings


def collect_seal_block() -> List[dict]:
    rows: List[dict] = []
    for s in _iter_seal():
        rows.append({"combo": s.name, "lang": "seal", "cer": float("nan"), "n": 0})
    return rows


# ---------------------------------------------------------------------------
# Main entry points (CLI handlers)
# ---------------------------------------------------------------------------

def main_full(results_dir: Path, *, threshold: float, report_path: Path) -> int:
    """Score the 7×7 main matrix + lang-rec + doc blocks.

    The user-facing default is the 7×7 (visible in the report header) plus
    a lang-rec/doc block; the 'all' flag adds strip and seal. This matches
    the contract: '7×7 main matrix + lang-rec/doc single column'.
    """
    det_to_rec, warnings = collect_main_matrix(results_dir)
    lang_rec = collect_lang_rec_block(results_dir)
    doc = collect_doc_block(results_dir)
    md, has_fail = render_report(
        det_to_rec_cer=det_to_rec, lang_rec_rows=lang_rec, doc_rows=doc,
        strip_rows=[], seal_rows=[], threshold=threshold,
        warnings=warnings,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return 1 if has_fail else 0


def main_strip(results_dir: Path, *, threshold: float, report_path: Path) -> int:
    """Strip-only mode: just the strip block, no main matrix.

    (Use --all to combine everything.)"""
    rows, warnings = collect_strip_block(results_dir)
    md, has_fail = render_report(
        det_to_rec_cer={}, lang_rec_rows=[], doc_rows=[],
        strip_rows=rows, seal_rows=[], threshold=threshold, warnings=warnings,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return 1 if has_fail else 0


def main_seal(*, report_path: Path) -> int:
    """Seal-only mode: just the M4 stub."""
    rows = collect_seal_block()
    md, _ = render_report(
        det_to_rec_cer={}, lang_rec_rows=[], doc_rows=[],
        strip_rows=[], seal_rows=rows, threshold=0.05, warnings=[],
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return 0


def main_all(results_dir: Path, *, threshold: float, report_path: Path) -> int:
    det_to_rec, _ = collect_main_matrix(results_dir)
    lang_rec = collect_lang_rec_block(results_dir)
    doc = collect_doc_block(results_dir)
    strip, warnings = collect_strip_block(results_dir)
    seal = collect_seal_block()
    md, has_fail = render_report(
        det_to_rec_cer=det_to_rec, lang_rec_rows=lang_rec, doc_rows=doc,
        strip_rows=strip, seal_rows=seal, threshold=threshold,
        warnings=warnings,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return 1 if has_fail else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="score")
    ap.add_argument("--results-dir",
                    default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--report", default=None,
                    help="path to write report.md (default: <results-dir>/report.md)")
    ap.add_argument("--report-md", dest="report_md", default=None,
                    help="alias of --report")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--strip", action="store_true", help="score strip cells")
    ap.add_argument("--seal", action="store_true", help="emit seal N/A stub")
    ap.add_argument("--all", action="store_true",
                    help="write full + lang-rec + doc + strip + seal sections")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        return 2
    report_path = Path(args.report_md or args.report or
                       (results_dir / "report.md"))

    if args.all:
        return main_all(results_dir, threshold=args.threshold, report_path=report_path)
    if args.strip:
        return main_strip(results_dir, threshold=args.threshold, report_path=report_path)
    if args.seal:
        return main_seal(report_path=report_path)
    return main_full(results_dir, threshold=args.threshold, report_path=report_path)


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
