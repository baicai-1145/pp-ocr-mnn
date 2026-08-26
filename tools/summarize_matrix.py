#!/usr/bin/env python3
"""M2-MATRIX: per-det / per-rec failure summary.

Reads tools/score.py's report.md (default results/report.md) and
emits:
  - per-det (row of 7×7 main matrix) stats: how many cells PASS, mean
    CER across all 7 recs, worst rec.
  - per-rec (col) stats: how many PASS, mean CER, worst det.
  - all-FAIL cells (CER > 0.05) ranked by CER desc (top 20).
  - per-language stats (mean CER across the 49 main cells).

This is a *measurement-only* script: it reads report.md, does not
re-run any inference. It assumes the 7×7 main matrix is in
the format produced by tools/score.py.
"""
import json
import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_report(report_path: Path) -> dict:
    """Parse the score.py report.md back into structured data.

    The report has a 7×7 markdown table. We re-parse it.
    """
    txt = report_path.read_text(encoding="utf-8")

    # Pull the 'Main matrix (49 cells): PASS=... FAIL=... N/A=...'
    # summary line, if present.
    main_summary = re.search(
        r"Main matrix \(49 cells\):\s*PASS=(\d+)\s+FAIL=(\d+)\s+N/A=(\d+)",
        txt)
    if not main_summary:
        print(f"WARNING: no main matrix summary line in {report_path}",
              file=sys.stderr)
    p, f, na = (int(x) for x in main_summary.groups()) if main_summary \
        else (0, 0, 0)

    # Find the main matrix table.
    table_re = re.compile(
        r"\|\s*PP-OCRv4_mobile_rec\s*\|.*?\n\s*\n", re.DOTALL)
    tables = table_re.findall(txt)
    # The first one is the 7×7 main matrix (rows = det, cols = rec).
    if not tables:
        # alt pattern: with 7 recs the .* is greedy; try a more
        # conservative end-anchor: the line that starts with '**'
        # (summary line).
        table_re = re.compile(
            r"(\| det \\\\ rec \|[^\n]+\n(?:\|[^\n]+\n)+)", re.MULTILINE)
        tables = table_re.findall(txt)

    if not tables:
        print(f"WARNING: no 7×7 main matrix table in {report_path}",
              file=sys.stderr)
        return {"main": {}, "summary": (p, f, na)}

    table = tables[0]
    lines = [l for l in table.splitlines() if l.strip().startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    recs = [c for c in header[1:] if c]  # 7 rec names
    main = {}
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) < 8:
            continue
        det = cells[0]
        if not det.startswith("PP-OCR"):
            continue
        main[det] = {}
        for i, rec in enumerate(recs):
            v = cells[1 + i]
            # Examples: "0.0429 PASS" / "0.1130 FAIL" / "N/A N/A"
            tok = v.split()
            cer_str = tok[0]
            status = tok[1] if len(tok) > 1 else "N/A"
            if cer_str == "N/A":
                cer = None
            else:
                try:
                    cer = float(cer_str)
                except ValueError:
                    cer = None
            main[det][rec] = {"cer": cer, "status": status}
    return {"main": main, "summary": (p, f, na)}


def main():
    report_path = Path(sys.argv[1] if len(sys.argv) > 1
                       else "results/report.md")
    if not report_path.exists():
        print(f"ERROR: {report_path} not found", file=sys.stderr)
        sys.exit(1)
    data = parse_report(report_path)
    main = data["main"]
    p, f, na = data["summary"]
    if not main:
        print("no main matrix data; nothing to summarize")
        return

    print("=" * 60)
    print(f"M2-MATRIX summary  ({report_path})")
    print("=" * 60)
    print(f"Main matrix totals: PASS={p}  FAIL={f}  N/A={na}")
    print()

    # Per-det (row) stats
    print("--- per-det (row of 7×7) ---")
    det_stats = []
    for det, row in main.items():
        cers = [r["cer"] for r in row.values() if r["cer"] is not None]
        if not cers:
            continue
        passes = sum(1 for r in row.values() if r["status"] == "PASS")
        fails = sum(1 for r in row.values() if r["status"] == "FAIL")
        # Worst (max) cell
        worst = max(row.items(), key=lambda kv: kv[1]["cer"] or -1)
        det_stats.append({
            "det": det,
            "mean_cer": sum(cers) / len(cers),
            "max_cer": max(cers),
            "pass": passes,
            "fail": fails,
            "n": len(cers),
            "worst_rec": worst[0],
            "worst_cer": worst[1]["cer"],
        })
    det_stats.sort(key=lambda x: x["mean_cer"])
    for d in det_stats:
        print(f"  {d['det']:30s}  mean={d['mean_cer']:.4f}  max={d['max_cer']:.4f}  "
              f"pass={d['pass']}/{d['n']}  worst={d['worst_rec']} ({d['worst_cer']:.4f})")
    print()

    # Per-rec (col) stats
    print("--- per-rec (col of 7×7) ---")
    rec_stats = []
    all_recs = sorted({r for row in main.values() for r in row})
    for rec in all_recs:
        cers = [main[d][rec]["cer"] for d in main if main[d][rec]["cer"] is not None]
        if not cers:
            continue
        passes = sum(1 for d in main if main[d][rec]["status"] == "PASS")
        fails = sum(1 for d in main if main[d][rec]["status"] == "FAIL")
        worst = max(((d, main[d][rec]["cer"]) for d in main
                     if main[d][rec]["cer"] is not None), key=lambda x: x[1])
        rec_stats.append({
            "rec": rec, "mean_cer": sum(cers) / len(cers),
            "max_cer": max(cers), "pass": passes, "fail": fails,
            "n": len(cers), "worst_det": worst[0],
            "worst_cer": worst[1],
        })
    rec_stats.sort(key=lambda x: x["mean_cer"])
    for r in rec_stats:
        print(f"  {r['rec']:30s}  mean={r['mean_cer']:.4f}  max={r['max_cer']:.4f}  "
              f"pass={r['pass']}/{r['n']}  worst={r['worst_det']} ({r['worst_cer']:.4f})")
    print()

    # All-FAIL cells ranked
    print("--- all-FAIL cells (CER > 0.05) ranked by CER desc ---")
    fails = []
    for det, row in main.items():
        for rec, r in row.items():
            if r["status"] == "FAIL" and r["cer"] is not None:
                fails.append({"det": det, "rec": rec, "cer": r["cer"]})
    fails.sort(key=lambda x: x["cer"], reverse=True)
    for f in fails[:20]:
        print(f"  {f['det']:30s} x {f['rec']:30s}  CER={f['cer']:.4f}")
    print()
    print(f"Total FAIL: {len(fails)}")

    # 7x7 small grid showing CER color-coded
    print()
    print("--- 7×7 CER heatmap (top-3 det best, top-3 rec best) ---")
    best_dets = sorted(det_stats, key=lambda x: x["mean_cer"])[:3]
    best_recs = sorted(rec_stats, key=lambda x: x["mean_cer"])[:3]
    print("Top-3 best dets (by row mean):", [d["det"] for d in best_dets])
    print("Top-3 best recs (by col mean):", [r["rec"] for r in best_recs])

    # "Pipeline-robust det" classification: a det that PASSes against
    # all 7 recs is a numerical-robust candidate.
    print()
    print("--- 'pipeline-robust' dets (all 7 recs PASS) ---")
    robust = [d["det"] for d in det_stats if d["pass"] == d["n"] and d["n"] == 7]
    print(" ", robust)
    print("--- dets with 0 PASS ---")
    zero = [d["det"] for d in det_stats if d["pass"] == 0]
    print(" ", zero)


if __name__ == "__main__":
    main()
