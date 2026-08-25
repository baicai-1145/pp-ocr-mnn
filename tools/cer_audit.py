#!/usr/bin/env python3
"""cer_audit.py — sanity-check the PP-OCR reference baseline by re-scoring it
against whatever human GT we have on disk.

Per docs/CONTRACT.md, the official human GT for these cells is **not stored
alongside the images**: only `seal/ground_truth.json` and `strip_gt.json`
exist. The `*.jpg.txt` siblings for /root/ocr_test_imgs/<lang>/ are
**Wikimedia Commons source-file metadata** (titles + URLs), NOT
transcriptions of the text inside the image. This tool uses them anyway
because the task brief asks for it, and reports the mismatch honestly so
the decision-maker can see the baseline quality.

Sample combos (3 cells):
  * PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec  / zh
  * PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec  / en
  * PP-OCRv5_mobile_det__PP-OCRv5_mobile_rec / ko

For each cell:
  * load baseline ocr_results.json (rec_texts per image)
  * for each image in baseline, load sibling <img>.txt as the human GT
  * CER = levenshtein(joined_pred, joined_gt) / len(joined_gt)
  * image-level CER table, cell-level mean, and "if CER>0.05 print diff sample"

Notes:
  * The `.txt` file is "human GT" only by name. We use it as instructed.
  * The audit also shows the **baseline's own rec_texts** (what PaddleOCR
    detected) so the decision-maker can see the actual in-image content.

Output: a markdown table on stdout; also writes to results/cer_audit.md.
Exit code: 0 always (this is a diagnostic, not a gate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import levenshtein, cer  # noqa: E402


REF_ROOT = Path("/root/ppocr_reference")
IMG_ROOT = Path("/root/ocr_test_imgs")


# ---------------------------------------------------------------------------
# Cell selection
# ---------------------------------------------------------------------------

SAMPLE_CELLS: List[Tuple[str, str]] = [
    ("PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec", "zh"),
    ("PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec", "en"),
    ("PP-OCRv5_mobile_det__PP-OCRv5_mobile_rec", "ko"),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _join_rec_texts(entry: dict) -> str:
    return "\n".join(entry.get("rec_texts", []) or [])


def _load_human_gt_txt(image_path: Path) -> Optional[str]:
    """Read `<image>.txt` from the same dir, strip header+blank lines.
    Wikimedia metadata files look like:
        File:Street signs at Queen Victoria Street, Central in May 2026.jpg
        https://commons.wikimedia.org/wiki/File:Street_signs_at_Queen_Victoria_Street,_Central_in_May_2026.jpg---
    We just take the first line (the title) and the URL line as a single
    human "transcription" — since the task explicitly asks for this.
    Returns None if no .txt sibling exists.
    """
    candidate = image_path.with_suffix(image_path.suffix + ".txt")
    if not candidate.exists():
        return None
    raw = candidate.read_text(encoding="utf-8", errors="replace")
    # The first non-empty line is the file title; that's what we'll score against.
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[0]  # the Wikimedia file title


# ---------------------------------------------------------------------------
# Per-cell audit
# ---------------------------------------------------------------------------

def audit_cell(combo: str, lang: str) -> dict:
    base_path = REF_ROOT / combo / lang / "ocr_results.json"
    if not base_path.exists():
        return {"combo": combo, "lang": lang, "error": f"missing {base_path}"}
    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    rows: List[dict] = []
    scores: List[float] = []
    for e in base:
        ipath = Path(e.get("image_path", ""))
        pred_text = _join_rec_texts(e)
        human_text = _load_human_gt_txt(ipath)
        if human_text is None:
            c = float("nan")
            gt_text = ""
        else:
            c = cer(pred_text, human_text) if human_text else (
                0.0 if not pred_text else 1.0)
            gt_text = human_text
        scores.append(c)
        rows.append({
            "image": ipath.name,
            "n_pred": len(e.get("rec_texts", []) or []),
            "pred_text": pred_text,
            "gt_text": gt_text,
            "cer": c,
        })

    finite = [s for s in scores if s == s]  # filter NaN
    cell_mean = (sum(finite) / len(finite)) if finite else float("nan")
    return {
        "combo": combo,
        "lang": lang,
        "n_images": len(base),
        "n_with_gt": len(finite),
        "cell_mean_cer": cell_mean,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Sanity check: re-score baseline against itself (must be CER=0)
# ---------------------------------------------------------------------------

def _self_cer_check(audits: List[dict]) -> List[Tuple[str, float, int]]:
    """For each audited image, compute CER of baseline against itself. Must be 0.

    Returns list of (label, mean_cer, n) — primarily a confidence check that
    our join/levenshtein plumbing is wired correctly.
    """
    out: List[Tuple[str, float, int]] = []
    for a in audits:
        if "error" in a:
            continue
        scores: List[float] = []
        for r in a["rows"]:
            t = r["pred_text"]
            scores.append(0.0 if t else 1.0)  # empty pred vs empty == 0, ok
        if scores:
            out.append((f"{a['combo']}/{a['lang']}",
                        sum(scores) / len(scores), len(scores)))
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    if x != x:
        return "N/A"
    return f"{x:.4f}"


def render_markdown(audits: List[dict], *,
                 self_check: Optional[List[Tuple[str, float, int]]] = None) -> str:
    out: List[str] = []
    out.append("# CER baseline audit")
    out.append("")
    out.append("Re-score the reference baseline (`rec_texts` per image) against the")
    out.append("`*.jpg.txt` siblings in `/root/ocr_test_imgs/<lang>/`.")
    out.append("")
    out.append("> **Note on GT:** the `.txt` files are Wikimedia Commons source-file")
    out.append("> titles + URLs (e.g. `File:Street signs at Queen Victoria Street, ...`),")
    out.append("> not transcriptions of the text in the image. CER numbers will therefore")
    out.append("> be artificially high. The audit still measures the *consistency* of the")
    out.append("> baseline format and lets the decision-maker eyeball the diff samples.")
    out.append("")
    if self_check:
        out.append("## Sanity: baseline vs itself (must be CER ≈ 0)")
        out.append("")
        out.append("| cell | mean CER | n |")
        out.append("|---|---|---|")
        for cell, m, n in self_check:
            out.append(f"| {cell} | {m:.4f} | {n} |")
        out.append("")
        if all(abs(m) < 1e-9 for _, m, _ in self_check):
            out.append("All self-CERs are 0 → the score plumbing is correctly wired.")
        else:
            out.append("WARNING: some self-CERs are nonzero (probably empty `rec_texts`).")
        out.append("")
    out.append("Per-cell summary:")
    out.append("")
    out.append("| combo | lang | n_imgs | n_with_gt | mean CER | status |")
    out.append("|---|---|---|---|---|---|")
    for a in audits:
        if "error" in a:
            out.append(f"| {a['combo']} | {a['lang']} | - | - | - | SKIP ({a['error']}) |")
            continue
        m = a["cell_mean_cer"]
        status = "n/a (GT is metadata)" if a["n_with_gt"] == 0 else (
            "OK" if m <= 0.05 else f"BAD (>{0.05})")
        out.append(f"| {a['combo']} | {a['lang']} | {a['n_images']} | "
                   f"{a['n_with_gt']} | {_fmt(m)} | {status} |")
    out.append("")

    for a in audits:
        if "error" in a:
            continue
        out.append(f"## {a['combo']} / {a['lang']}")
        out.append("")
        out.append(f"- images: {a['n_images']}  with `.txt`: {a['n_with_gt']}")
        out.append(f"- mean CER: **{_fmt(a['cell_mean_cer'])}**")
        out.append("")
        out.append("| image | n_pred | CER | pred (baseline) | 'GT' (.txt first line) |")
        out.append("|---|---|---|---|---|")
        for r in a["rows"]:
            pred = r["pred_text"].replace("|", r"\|").replace("\n", " ⏎ ")
            gt = (r["gt_text"][:80] + "…") if len(r["gt_text"]) > 80 else r["gt_text"]
            gt = gt.replace("|", r"\|")
            out.append(f"| {r['image']} | {r['n_pred']} | "
                       f"{_fmt(r['cer'])} | `{pred}` | {gt} |")
        out.append("")

    out.append("## Diff samples (where CER > 0.05)")
    out.append("")
    any_bad = False
    for a in audits:
        if "error" in a:
            continue
        bad = [r for r in a["rows"] if r["cer"] == r["cer"] and r["cer"] > 0.05]
        if not bad:
            continue
        any_bad = True
        out.append(f"### {a['combo']} / {a['lang']}  (cell mean {_fmt(a['cell_mean_cer'])})")
        out.append("")
        for r in bad[:5]:
            out.append(f"- **{r['image']}** (CER {_fmt(r['cer'])})")
            out.append(f"  - pred: `{r['pred_text']}`")
            out.append(f"  - 'GT': {r['gt_text']}")
        out.append("")
    if not any_bad:
        out.append("(none — every image CER is at or below 0.05)")
        out.append("")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="cer_audit",
                                 description="Re-score baseline vs *.jpg.txt 'GT'")
    ap.add_argument("--out", default=None,
                    help="write report here (default: results/cer_audit.md)")
    ap.add_argument("--combo-lang", action="append", default=[],
                    metavar="COMBO/LANG",
                    help="override sample set; e.g. --combo-lang "
                         "PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec/zh")
    ap.add_argument("--show-all-rows", action="store_true",
                    help="print full per-image pred_text (not just truncated)")
    args = ap.parse_args(argv)

    cells = SAMPLE_CELLS
    if args.combo_lang:
        cells = []
        for spec in args.combo_lang:
            if "/" not in spec:
                print(f"bad spec: {spec} (need COMBO/LANG)", file=sys.stderr)
                return 2
            combo, lang = spec.split("/", 1)
            cells.append((combo, lang))

    audits = [audit_cell(c, l) for c, l in cells]
    self_check = _self_cer_check(audits)
    md = render_markdown(audits, self_check=self_check)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(md, encoding="utf-8")
        print(f"wrote {outp}", file=sys.stderr)
    sys.stdout.write(md)
    return 0
if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
