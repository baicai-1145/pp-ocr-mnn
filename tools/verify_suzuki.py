#!/usr/bin/env python3
"""verify_suzuki.py — prove src/postprocess/suzuki.cpp is byte-exact with cv2.

`db_postprocess` feeds the contour points straight into `minAreaRect`, so any
difference in point ORDER or CONTENT versus OpenCV changes the emitted boxes.
This script is the standing evidence for that claim: it compiles the standalone
`tests/test_suzuki` driver, runs it over a set of masks, and compares the
contour list with `cv2.findContours(mask, RETR_LIST, CHAIN_APPROX_NONE)`.

Two input sources:
  * `--prob FILE H W THRESH`  binarize a raw float32 prob map (the real input
    the det head produces), repeated per flag — this is the interesting case;
  * `--synthetic`  the built-in shape zoo (rings, combs, diagonal chains,
    edge-touching blobs, random blobs, a 480x640 many-box map). Needs only
    numpy + cv2.

Usage:
    python3 tools/verify_suzuki.py --synthetic
    python3 tools/verify_suzuki.py --prob .tmp/probs/ja_09.f32 960 1280 0.3

Exit code 0 iff every mask matches cv2 exactly (contour count, order, every
point, in order).

NOTE on scope: PaddleX calls `cv2.findContours(bitmap, cv2.RETR_LIST,
cv2.CHAIN_APPROX_NONE)` and consumes the raw contour. We reproduce cv2 4.x,
which is what the reference runtime shipped. OpenCV 5.x adds a `TRUE` fast path
for the no-hierarchy RETR_LIST case; both 4.10 and 5.0 were checked to give the
same output as this implementation for every mask tested.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("verify_suzuki.py needs numpy and opencv-python-headless\n")
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "third_party" / "_suzuki_driver"


def build_driver() -> Path:
    """Compile tests/test_suzuki.cpp + src/postprocess/suzuki.cpp standalone."""
    DRIVER.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["c++", "-std=c++17", "-O2", "-I", str(ROOT / "include"),
           str(ROOT / "tests" / "test_suzuki.cpp"),
           str(ROOT / "src" / "postprocess" / "suzuki.cpp"),
           "-o", str(DRIVER)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(2)
    return DRIVER


def contours_from_cv2(mask: "np.ndarray") -> list:
    cs, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [[(int(p[0]), int(p[1])) for p in c.reshape(-1, 2)] for c in cs]


def contours_from_cxx(driver: Path, mask: "np.ndarray") -> list:
    h, w = mask.shape
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        mask.astype(np.uint8).tofile(tf.name)
        raw = tf.name
    try:
        out = subprocess.run([str(driver), raw, str(h), str(w)],
                             capture_output=True, text=True)
        lines = [l for l in out.stdout.splitlines() if l.strip()]
    finally:
        os.unlink(raw)
    res = []
    for line in lines:
        res.append([tuple(int(v) for v in tok.split(",")) for tok in line.split()])
    return res


def compare(name: str, mask: "np.ndarray", driver: Path) -> bool:
    mine = contours_from_cxx(driver, mask)
    ref = contours_from_cv2(mask)
    if mine == ref:
        print("  OK   %-16s %4d contours, %6d points" %
              (name, len(ref), sum(len(c) for c in ref)))
        return True
    print("  FAIL %-16s" % name)
    print("       mine n=%d sizes=%s" % (len(mine), [len(c) for c in mine]))
    print("       cv2  n=%d sizes=%s" % (len(ref), [len(c) for c in ref]))
    if [len(c) for c in mine] == [len(c) for c in ref]:
        for i, (a, b) in enumerate(zip(mine, ref)):
            if a != b:
                first = next((k for k in range(min(len(a), len(b)))
                              if a[k] != b[k]), 0)
                print("       contour %d first differs at point %d: mine=%s cv2=%s"
                      % (i, first, a[first], b[first]))
                break
    return False


def synthetic_masks():
    """Shape zoo covering the cases the old CCL+flood tracer got wrong."""
    W, H = 128, 96
    def blank(w=W, h=H):
        return np.zeros((h, w), np.uint8)

    def rect(m, x0, y0, x1, y1, v=1):
        m[y0:y1 + 1, x0:x1 + 1] = v
        return m

    out = {}
    for k in range(6):
        out["rect%d" % k] = rect(blank(), 5 + k, 5, 40 + k, 30)
    out["diag_touch"] = rect(rect(blank(), 10, 10, 30, 30), 31, 31, 50, 50)
    out["diag_apart"] = rect(rect(blank(), 10, 10, 30, 30), 32, 32, 50, 50)
    out["corner_overlap"] = rect(rect(blank(), 10, 10, 30, 30), 30, 30, 50, 50)
    out["near_touch"] = rect(rect(blank(), 10, 10, 30, 30), 29, 31, 50, 50)
    out["ring1"] = rect(blank(), 10, 10, 60, 60); out["ring1"][20:50, 20:50] = 0
    out["thin_ring"] = rect(blank(), 10, 10, 80, 70); out["thin_ring"][15:65, 15:75] = 0
    n = rect(blank(), 5, 5, 90, 80)
    n[10:75, 10:85] = 0; n[13:72, 13:82] = 1; n[20:65, 20:75] = 0
    out["nested"] = n
    c = rect(blank(), 2, 2, 120, 80)
    for x in range(10, 110, 10):
        c[20:60, x] = 0
    out["comb"] = c
    sh = rect(blank(), 10, 10, 60, 60); sh[25:45, 25:45] = 0
    sh[25:45, 45] = 1
    out["split_hole"] = sh
    d = rect(blank(), 10, 10, 50, 50); d[30, 30] = 0
    out["diag_two"] = rect(d, 31, 31, 60, 60)
    cl = blank(); cl[50, 5:120] = 1; cl[5:90, 60] = 1
    out["cross_line"] = cl
    out["edge_tl"] = rect(blank(), 0, 0, 40, 40)
    out["edge_bl"] = rect(blank(), 0, 50, 40, 95)
    out["edge_r"] = rect(blank(), 100, 0, 127, 95)
    out["solid"] = rect(blank(), 0, 0, 127, 95)
    fr = rect(blank(), 0, 0, 127, 95); fr[5:90, 5:120] = 0
    out["frame"] = fr
    ch = blank()
    for i in range(0, 60):
        ch[10 + i, 10 + i] = 1
        ch[10 + i, 11 + i] = 1
    out["diag_chain"] = ch
    td = rect(blank(), 10, 10, 40, 40); td[20:30, 20:30] = 0
    td = rect(td, 41, 41, 70, 70); td[50:60, 50:60] = 0
    out["two_donuts_diag"] = td
    rng = np.random.default_rng(7)
    for k in range(12):
        m = blank()
        for _ in range(int(rng.integers(1, 7))):
            cx, cy = int(rng.integers(5, W - 6)), int(rng.integers(5, H - 6))
            rx, ry = int(rng.integers(2, 19)), int(rng.integers(2, 19))
            m[max(0, cy - ry):min(H, cy + ry + 1),
              max(0, cx - rx):min(W, cx + rx + 1)] ^= 1
        out["rand%d" % k] = m
    # max_candidates stress: 3700 borders (1850 boxes x2 for their holes) on a
    # 480x640 map, i.e. well past the configured max_candidates = 1000. Confirms
    # the tracer's emission order and content stay exact when the cap WOULD
    # filter, so the cap cannot silently change which boxes survive.
    grid = np.zeros((480, 640), np.uint8)
    n_borders = 0
    for gy in range(20, 480 - 20, 12):
        for gx in range(20, 640 - 20, 12):
            grid[gy:gy + 8, gx:gx + 8] = 1
            grid[gy + 3:gy + 5, gx + 3:gx + 5] = 0
            n_borders += 2
    assert n_borders == 3700, n_borders
    out["grid_3700"] = grid
    # larger map with many small boxes (perf + ordering stress)
    mb = np.zeros((480, 640), np.uint8)
    rng = np.random.default_rng(11)
    for _ in range(120):
        x0 = int(rng.integers(0, 640 - 60)); y0 = int(rng.integers(0, 480 - 30))
        mb[y0:min(480, y0 + int(rng.integers(6, 26))),
           x0:min(640, x0 + int(rng.integers(10, 51)))] = 1
    out["many_boxes"] = mb
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run the built-in shape zoo")
    ap.add_argument("--prob", nargs=4, action="append", default=[],
                    metavar=("FILE", "H", "W", "THRESH"),
                    help="raw float32 prob map to binarize and check")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.synthetic and not args.prob:
        ap.error("pass --synthetic and/or --prob")

    driver = build_driver()
    print("cv2 %s — comparing against src/postprocess/suzuki.cpp" % cv2.__version__)
    bad = 0
    if args.synthetic:
        print("synthetic masks:")
        for name, m in synthetic_masks().items():
            if not compare(name, m, driver):
                bad += 1
    for spec in args.prob:
        path, h, w, th = spec[0], int(spec[1]), int(spec[2]), float(spec[3])
        prob = np.fromfile(path, dtype=np.float32).reshape(h, w)
        mask = (prob > th).astype(np.uint8)
        print("prob map %s (%d x %d, thresh %g):" % (path, h, w, th))
        if not compare(Path(path).stem, mask, driver):
            bad += 1

    total = (len(synthetic_masks()) if args.synthetic else 0) + len(args.prob)
    print("\n%s — %d/%d masks byte-exact with cv2"
          % ("PASS" if bad == 0 else "FAIL", total - bad, total))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
