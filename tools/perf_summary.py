#!/usr/bin/env python3
"""Pretty-print / summarize a perf baseline JSON from bench.py.

Usage:
  python3 tools/perf_summary.py /tmp/perf_baseline.json
"""
from __future__ import annotations

import json
import sys

STAGES = ["decode_ms", "det_prep_ms", "det_run_ms", "db_post_ms",
          "crop_warp_ms", "rec_prep_ms", "rec_run_ms", "ctc_decode_ms",
          "cls_ms"]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = json.load(open(sys.argv[1]))
    cells = d["cells"]
    print(f"pp-ocr-mnn perf baseline — {d['meta']['date']} on "
          f"{d['meta']['hostname']}  ({len(d['meta']['images'])} imgs x "
          f"{d['meta']['warm_runs']} warm runs)\n")

    # ---- main table: e2e + top stages ------------------------------
    hdr = (f"{'cell':22s} {'e2e mean':>9s} {'img med':>8s} {'±':>6s} {'FPS':>6s} "
           f"{'det_run':>8s} {'rec_run':>8s} {'db_post':>8s} {'cold ms':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(cells):
        c = cells[name]
        e2e = c["e2e_ms"]
        fps = 1000.0 / e2e["mean"] if e2e["mean"] > 0 else 0.0
        cold = c["cold"]["e2e_ms"] + c["cold"]["create_ms"] if c.get("cold") else 0.0
        med = c.get("img_median_e2e_ms", 0.0)
        print(f"{name:22s} {e2e['mean']:9.1f} {med:8.1f} {e2e['std']:6.1f} {fps:6.2f} "
              f"{c['det_run_ms']['mean']:8.1f} {c['rec_run_ms']['mean']:8.1f} "
              f"{c['db_post_ms']['mean']:8.1f} {cold:8.0f}")

    # ---- stage share per cell --------------------------------------
    print("\nStage share (% of e2e, warm mean):")
    for name in sorted(cells):
        c = cells[name]
        total = c["e2e_ms"]["mean"]
        parts = "  ".join(
            f"{s.replace('_ms','')}={100.0 * c[s]['mean'] / total:4.1f}%"
            for s in STAGES if c[s]["mean"] > 0.05)
        print(f"  {name:22s} {parts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
