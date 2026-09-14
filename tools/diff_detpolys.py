#!/usr/bin/env python3
"""diff_detpolys.py — compare det polygon output between two prediction trees.

The hard gate for the perf-post db_post rewrite is that the detected polygons do
not change. This tool diffs `det_polys` (and `rec_texts`) between two directories
produced by `tools/run_reference.py` and reports, per image:

    box count        same / different
    same-as-multiset  every polygon present in both (order-insensitive)
    same-as-sequence  polygons identical in the same order (the strong claim)

That distinction matters: cv2's contour emission order drives candidate order,
and the old CCL extractor sorted components by pixel count, so a rewrite could
legitimately agree on the box SET while disagreeing on ORDER. Reporting both
makes the difference visible instead of hiding it behind a single number.

Usage:
    python3 tools/diff_detpolys.py OLD_DIR NEW_DIR [--max-show N]

OLD_DIR / NEW_DIR are run_reference --results-dir trees, i.e. they contain
<det>__<rec>/<lang>/pred.json. Exit code 0 iff every image matches as a
sequence. Use --set-only to accept multiset equality instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_images(root: Path) -> Dict[str, dict]:
    """Map '<cell>/<lang>/<image>' -> prediction record."""
    out: Dict[str, dict] = {}
    for pj in sorted(root.glob("*/*/pred.json")):
        cell, lang = pj.parts[-3], pj.parts[-2]
        try:
            recs = json.load(open(pj))
        except (OSError, json.JSONDecodeError) as e:
            sys.stderr.write("cannot read %s: %s\n" % (pj, e))
            continue
        for r in recs:
            name = str(r.get("image_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
            out["%s/%s/%s" % (cell, lang, name)] = r
    return out


def polys_of(rec: dict) -> List[tuple]:
    return [tuple(int(v) for v in poly) for poly in rec.get("det_polys", [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--max-show", type=int, default=5,
                    help="how many differing images to detail (default 5)")
    ap.add_argument("--set-only", action="store_true",
                    help="accept multiset equality; do not require same order")
    args = ap.parse_args()

    old = load_images(Path(args.old))
    new = load_images(Path(args.new))
    if not old:
        sys.stderr.write("no pred.json found under %s\n" % args.old)
        return 2
    if not new:
        sys.stderr.write("no pred.json found under %s\n" % args.new)
        return 2

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    for k in only_old[:5]:
        print("only in OLD: %s" % k)
    for k in only_new[:5]:
        print("only in NEW: %s" % k)

    n_seq = n_set = n_diff = 0
    shown = 0
    for key in sorted(set(old) & set(new)):
        po, pn = polys_of(old[key]), polys_of(new[key])
        if po == pn:
            n_seq += 1
            n_set += 1
            continue
        if sorted(po) == sorted(pn):
            n_set += 1
            n_diff += 1
            if shown < args.max_show:
                shown += 1
                print("ORDER  %s  %d boxes, same set, different order" % (key, len(po)))
                for i, (a, b) in enumerate(zip(po, pn)):
                    if a != b:
                        print("        [%d] old=%s new=%s" % (i, a, b))
            continue
        n_diff += 1
        if shown < args.max_show:
            shown += 1
            so, sn = set(po), set(pn)
            print("DIFF   %s  old=%d boxes new=%d boxes" % (key, len(po), len(pn)))
            for p in sorted(so - sn)[:3]:
                print("        only OLD %s" % (p,))
            for p in sorted(sn - so)[:3]:
                print("        only NEW %s" % (p,))
            to = old[key].get("rec_texts", [])
            tn = new[key].get("rec_texts", [])
            if to != tn:
                print("        rec_texts differ: old=%d new=%d lines" % (len(to), len(tn)))

    n = n_seq + n_diff
    print("\nimages compared : %d" % n)
    print("poly sequence   : %d identical, %d differ" % (n_seq, n_diff))
    print("poly multiset   : %d identical, %d differ" % (n_set, n - n_set))
    if only_old or only_new:
        print("missing images  : %d only-old, %d only-new" % (len(only_old), len(only_new)))
    passed = (n_diff == 0 if not args.set_only else (n - n_set) == 0)
    print("RESULT          : %s" % ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
