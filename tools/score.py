#!/usr/bin/env python3
"""score.py — CER scoring for the PP-OCR reference matrix.

Per docs/CONTRACT.md:
  * For every (det__rec) cell: per image, CER = levenshtein(pred_join, base_join)
    / len(base_join) where join = "\n".join(rec_texts); cell score = mean over
    images; PASS if cell_score ≤ 0.05.
  * --strip mode scores `strip__<rec>` cells against
    /root/ocr_test_imgs/strip_gt.json (per-image gt).
  * seal cells (PP-OCRv4_mobile_seal_det/, PP-OCRv4_server_seal_det/) are out
    of scope (M4) — reported as N/A.
  * Output: results/report.md matrix + exit code (0 = all PASS, 1 = any FAIL).

Levenshtein is a self-implemented O(m*n) DP (no numpy dep). Acceptable for
matrix-scale inputs; cell text is short.
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
    # ensure b is shorter
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
# Loading
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
# Per-cell scoring
# ---------------------------------------------------------------------------

def _join_texts(d: dict) -> str:
    return "\n".join(d.get("rec_texts", []) or [])


def _join_texts_strip(d: dict) -> str:
    """For strip cells, baseline uses `gt_text` (single line). For symmetry
    we accept both list and scalar."""
    gt = d.get("gt_text", "")
    if isinstance(gt, list):
        return "\n".join(gt)
    return gt


def score_image(pred_rec_texts: List[str], base: str) -> float:
    pred = "\n".join(pred_rec_texts or [])
    return cer(pred, base)


def score_full_cell(combo_dir: Path, lang: str, results_dir: Path) -> Tuple[float, int]:
    base = load_baseline(combo_dir / lang / "ocr_results.json")
    pred_path = results_dir / combo_dir.name / lang / "pred.json"
    if not pred_path.exists():
        return float("nan"), 0
    pred = load_pred(pred_path)
    # build map by image_path
    pred_by_path: Dict[str, List[str]] = {}
    for p in pred:
        pred_by_path[p["image_path"]] = p.get("rec_texts", [])
    scores: List[float] = []
    for b in base:
        btext = _join_texts(b)
        ipath = b.get("image_path", "")
        ptexts = pred_by_path.get(ipath, [])
        scores.append(score_image(ptexts, btext))
    if not scores:
        return float("nan"), 0
    return sum(scores) / len(scores), len(scores)


def score_strip_cell(combo_dir: Path, results_dir: Path,
                     strip_gt: Dict[str, str]) -> Tuple[float, int]:
    pred_path = results_dir / combo_dir.name / "pred.json"
    if not pred_path.exists():
        return float("nan"), 0
    pred = load_pred(pred_path)
    scores: List[float] = []
    for p in pred:
        ipath = p["image_path"]
        gt = strip_gt.get(ipath, "")
        scores.append(score_image(p.get("rec_texts", []), gt))
    if not scores:
        return float("nan"), 0
    return sum(scores) / len(scores), len(scores)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "N/A"
    return f"{x:.4f}"


def _status(x: float, thr: float = 0.05) -> str:
    if x != x:
        return "N/A"
    return "PASS" if x <= thr else "FAIL"


def write_report(report_path: Path, rows: List[dict], *,
                 title: str, threshold: float) -> None:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Threshold: CER ≤ {threshold}")
    lines.append("")
    # Decide columns
    cols = ["combo", "lang", "images", "cer", "status"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    pass_n = fail_n = nan_n = 0
    for r in rows:
        c = r["cer"]
        if c != c:
            nan_n += 1
        else:
            if c <= threshold:
                pass_n += 1
            else:
                fail_n += 1
        lines.append("| {combo} | {lang} | {n} | {cer} | {st} |".format(
            combo=r["combo"], lang=r.get("lang", "-"), n=r.get("n", 0),
            cer=_fmt(c), st=_status(c, threshold),
        ))
    lines.append("")
    lines.append(f"**Summary:** PASS={pass_n}  FAIL={fail_n}  N/A={nan_n}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _iter_full_combos() -> List[Path]:
    out: List[Path] = []
    for c in sorted(REF_ROOT.iterdir()):
        if not c.is_dir():
            continue
        n = c.name
        if "__" not in n or n.startswith("strip__"):
            continue
        # skip seal dirs (they have no __)
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


def main_full(results_dir: Path, *, threshold: float, report_path: Path) -> int:
    rows: List[dict] = []
    for combo_dir in _iter_full_combos():
        for lang_dir in sorted(combo_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            c, n = score_full_cell(combo_dir, lang, results_dir)
            rows.append({"combo": combo_dir.name, "lang": lang, "cer": c, "n": n})
    write_report(report_path, rows, title="PP-OCR Full OCR matrix — CER", threshold=threshold)
    has_fail = any((r["cer"] == r["cer"] and r["cer"] > threshold) for r in rows)
    return 1 if has_fail else 0


def main_strip(results_dir: Path, *, threshold: float, report_path: Path) -> int:
    if not STRIP_GT.exists():
        print(f"ERROR: strip GT not found at {STRIP_GT}", file=sys.stderr)
        return 2
    strip_gt = load_strip_gt(STRIP_GT)
    rows: List[dict] = []
    for combo_dir in _iter_strip_combos():
        # strip cells have a flat ocr_results.json per language subdir
        # (we read pred.json from results/<combo>/<lang>/pred.json, so iterate langs)
        any_lang = False
        for lang_dir in sorted(combo_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name
            # The strip baseline is per-language; but the GT we use is strip_gt
            # which is keyed by image path. So per-language cer works.
            pred_path = results_dir / combo_dir.name / lang / "pred.json"
            if not pred_path.exists():
                continue
            pred = load_pred(pred_path)
            scores: List[float] = []
            for p in pred:
                gt = strip_gt.get(p["image_path"], "")
                scores.append(score_image(p.get("rec_texts", []), gt))
            c = sum(scores) / len(scores) if scores else float("nan")
            rows.append({"combo": combo_dir.name, "lang": lang, "cer": c, "n": len(scores)})
            any_lang = True
        if not any_lang:
            rows.append({"combo": combo_dir.name, "lang": "-", "cer": float("nan"), "n": 0})
    write_report(report_path, rows, title="PP-OCR Strip rec-only — CER",
                 threshold=threshold)
    has_fail = any((r["cer"] == r["cer"] and r["cer"] > threshold) for r in rows)
    return 1 if has_fail else 0


def main_seal(*, report_path: Path) -> int:
    """M4 scope: report seal dirs as N/A, no scoring."""
    rows: List[dict] = []
    for s in _iter_seal():
        rows.append({"combo": s.name, "lang": "seal", "cer": float("nan"), "n": 0})
    write_report(report_path, rows, title="PP-OCR Seal det — N/A (M4 scope)",
                 threshold=0.05)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="score")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--report", default=None,
                    help="path to write report.md (default: results/report.md)")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--strip", action="store_true", help="score strip cells instead of full")
    ap.add_argument("--seal", action="store_true", help="emit seal N/A stub")
    ap.add_argument("--all", action="store_true", help="write full + strip + seal sections")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        return 2
    default_report = results_dir / "report.md"
    report_path = Path(args.report) if args.report else default_report

    if args.all:
        # combined
        rows: List[dict] = []
        for combo_dir in _iter_full_combos():
            for lang_dir in sorted(combo_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                c, n = score_full_cell(combo_dir, lang_dir.name, results_dir)
                rows.append({"combo": combo_dir.name, "lang": lang_dir.name,
                             "cer": c, "n": n})
        strip_gt = load_strip_gt(STRIP_GT) if STRIP_GT.exists() else {}
        for combo_dir in _iter_strip_combos():
            for lang_dir in sorted(combo_dir.iterdir()):
                if not lang_dir.is_dir():
                    continue
                pred_path = results_dir / combo_dir.name / lang_dir.name / "pred.json"
                if not pred_path.exists():
                    rows.append({"combo": combo_dir.name, "lang": lang_dir.name,
                                 "cer": float("nan"), "n": 0})
                    continue
                pred = load_pred(pred_path)
                scores: List[float] = []
                for p in pred:
                    gt = strip_gt.get(p["image_path"], "")
                    scores.append(score_image(p.get("rec_texts", []), gt))
                c = sum(scores) / len(scores) if scores else float("nan")
                rows.append({"combo": combo_dir.name, "lang": lang_dir.name,
                             "cer": c, "n": len(scores)})
        for s in _iter_seal():
            rows.append({"combo": s.name, "lang": "seal", "cer": float("nan"), "n": 0})
        write_report(report_path, rows, title="PP-OCR full matrix (full+strip+seal)",
                     threshold=args.threshold)
        has_fail = any((r["cer"] == r["cer"] and r["cer"] > args.threshold) for r in rows)
        return 1 if has_fail else 0

    if args.strip:
        return main_strip(results_dir, threshold=args.threshold, report_path=report_path)
    if args.seal:
        return main_seal(report_path=report_path)
    return main_full(results_dir, threshold=args.threshold, report_path=report_path)


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
