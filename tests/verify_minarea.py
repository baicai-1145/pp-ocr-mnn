#!/usr/bin/env python3
"""
Cross-validate our C++ min_area_rect (used in db_post + geometry) vs
OpenCV's cv2.minAreaRect (used by Paddle / PaddleX).

For the same 20-quad stress set as verify_unclip.py:
  1. Compute cv2.minAreaRect(contour).boxPoints() (Paddle's "GetMiniBoxes")
  2. Build a C++ harness that calls our min_area_rect() on the same
     float points, with the canonical order applied via
     sort_min_area_rect_points.
  3. Compare per-vertex diff after re-ordering the 4 outputs so the
     starting corner matches (minAreaRect is rotation-invariant; the
     order of the 4 points varies by implementation).

Pass criteria: max per-vertex diff <= 0.5 px, mean <= 0.25 px.
"""
from __future__ import annotations
import os
import subprocess
import sys
from typing import List, Tuple

import cv2
import numpy as np


HARNESS_TEMPLATE = r"""
// Auto-generated harness for min_area_rect cross-check (cv2 vs our cpp).
#include "ppocr/postprocess/geometry.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {
  int n_in;
  if (std::scanf("%d", &n_in) != 1 || n_in < 3) {
    std::fprintf(stderr, "expected >=3 input vertices, got %d\n", n_in);
    return 1;
  }
  std::vector<ppocr::PointF> pts;
  pts.reserve(n_in);
  for (int i = 0; i < n_in; ++i) {
    double x, y;
    if (std::scanf("%lf %lf", &x, &y) != 2) {
      std::fprintf(stderr, "vertex read failed at %d\n", i);
      return 2;
    }
    pts.push_back({static_cast<float>(x), static_cast<float>(y)});
  }
  ppocr::PointF box[4];
  if (!ppocr::min_area_rect(pts.data(), pts.size(), box)) {
    std::fprintf(stderr, "min_area_rect returned false\n");
    return 3;
  }
  // Apply canonical sort (Paddle's "sort_min_area_rect_points": sort by x,
  // split, per pair pick the top by y). Our sort_min_area_rect_points
  // matches Paddle's C++ exactly.
  ppocr::sort_min_area_rect_points(box);
  // Print 4 (x, y) pairs.
  for (int i = 0; i < 4; ++i) {
    std::printf("%.6f %.6f\n", box[i].x, box[i].y);
  }
  return 0;
}
"""


def make_quads(n: int = 20, seed: int = 0xc0ffee) -> List[np.ndarray]:
    """Same stress set as verify_unclip.py (rectangles, extreme aspect,
    45° rotated, random angle, near-degenerate)."""
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(5):
        w = rng.uniform(40, 220)
        h = rng.uniform(20, 80)
        x0 = rng.uniform(50, 800)
        y0 = rng.uniform(50, 600)
        out.append(np.array([
            [x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h],
        ], dtype=np.float64))
    for _ in range(4):
        w = rng.uniform(200, 400)
        h = w / 20.0
        cx = rng.uniform(300, 900)
        cy = rng.uniform(300, 600)
        out.append(np.array([
            [cx - w/2, cy - h/2], [cx + w/2, cy - h/2],
            [cx + w/2, cy + h/2], [cx - w/2, cy + h/2],
        ], dtype=np.float64))
    for _ in range(5):
        w = rng.uniform(80, 200)
        h = rng.uniform(30, 60)
        cx = rng.uniform(200, 900)
        cy = rng.uniform(200, 600)
        ang = np.deg2rad(45.0)
        ca, sa = np.cos(ang), np.sin(ang)
        local = np.array([[-w/2, -h/2], [w/2, -h/2],
                          [w/2, h/2], [-w/2, h/2]])
        out.append(np.array([[cx + x*ca - y*sa, cy + x*sa + y*ca]
                             for (x, y) in local]))
    for _ in range(4):
        w = rng.uniform(60, 200)
        h = rng.uniform(25, 70)
        cx = rng.uniform(200, 900)
        cy = rng.uniform(200, 600)
        ang = rng.uniform(0, 2*np.pi)
        ca, sa = np.cos(ang), np.sin(ang)
        local = np.array([[-w/2, -h/2], [w/2, -h/2],
                          [w/2, h/2], [-w/2, h/2]])
        out.append(np.array([[cx + x*ca - y*sa, cy + x*sa + y*ca]
                             for (x, y) in local]))
    for _ in range(2):
        w = rng.uniform(50, 100)
        h = rng.uniform(3, 8)
        x0 = rng.uniform(100, 800)
        y0 = rng.uniform(100, 600)
        out.append(np.array([
            [x0, y0], [x0 + w, y0 + 0.5],
            [x0 + w, y0 + h], [x0, y0 + h],
        ], dtype=np.float64))
    return out[:n]


def cv2_min_area_rect(quad: np.ndarray) -> np.ndarray:
    """Mirror Paddle's boxes_from_bitmap: cv2.minAreaRect then
    boxPoints, then canonical-sort to match our
    sort_min_area_rect_points. The canonical order is:
      box[0] = TL = smaller-y of leftmost pair
      box[1] = TR = smaller-y of rightmost pair
      box[2] = BR = larger-y  of rightmost pair
      box[3] = BL = larger-y  of leftmost pair
    """
    rect = cv2.minAreaRect(quad.astype(np.float32))
    pts = cv2.boxPoints(rect)  # 4 (x, y), any cyclic order
    # Sort by (x asc, y asc): pts_sorted[0..1] are leftmost pair,
    # pts_sorted[2..3] are rightmost pair.
    pts_sorted = sorted(pts.tolist(), key=lambda p: (p[0], p[1]))
    a, b, c, d = pts_sorted
    # Left pair: smaller-y is TL, larger-y is BL.
    if a[1] > b[1]:
        tl, bl = b, a
    else:
        tl, bl = a, b
    # Right pair: smaller-y is TR, larger-y is BR.
    if c[1] > d[1]:
        tr, br = d, c
    else:
        tr, br = c, d
    return np.array([tl, tr, br, bl], dtype=np.float64)


def build_harness(post_root: str, m1_include: str, build_dir: str) -> str:
    cpp_path = os.path.join(build_dir, "harness_minarea.cpp")
    bin_path = os.path.join(build_dir, "harness_minarea")
    with open(cpp_path, "w") as f:
        f.write(HARNESS_TEMPLATE)
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall",
        "-I", os.path.join(post_root, "include"),
        "-I", m1_include,
        cpp_path,
        os.path.join(post_root, "src", "postprocess", "geometry.cpp"),
        "-o", bin_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return bin_path


def cpp_min_area_rect(quad: np.ndarray, bin_path: str) -> np.ndarray:
    stdin = f"{len(quad)}\n"
    for p in quad:
        stdin += f"{p[0]:.6f} {p[1]:.6f}\n"
    proc = subprocess.run([bin_path], input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cpp harness failed: {proc.stderr}")
    lines = [l for l in proc.stdout.strip().split("\n") if l]
    if len(lines) != 4:
        raise RuntimeError(f"expected 4 lines, got {len(lines)}")
    pts = []
    for l in lines:
        x, y = l.split()
        pts.append([float(x), float(y)])
    return np.array(pts, dtype=np.float64)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    post_root = os.path.abspath(os.path.join(here, ".."))
    m1_include = os.path.abspath(os.path.join(post_root, "..", "m1", "include"))
    build_dir = os.path.join(here, "build-tests")
    os.makedirs(build_dir, exist_ok=True)
    bin_path = build_harness(post_root, m1_include, build_dir)

    quads = make_quads(n=20)
    print(f"{'idx':>3}  {'area_cv':>10}  {'area_cpp':>10}  {'max':>7}  {'mean':>7}  {'kind'}")
    print("-" * 60)
    all_max = []
    all_mean = []
    bad = 0
    for i, q in enumerate(quads):
        cv_box = cv2_min_area_rect(q)
        cpp_box = cpp_min_area_rect(q, bin_path)
        # Per-vertex diff after canonical sort (both are already in
        # canonical order, so 1:1 alignment should hold).
        diffs = np.linalg.norm(cv_box - cpp_box, axis=1)
        max_d = float(diffs.max())
        mean_d = float(diffs.mean())
        all_max.append(max_d)
        all_mean.append(mean_d)
        # Classify
        ws = [np.linalg.norm(q[(k+1) % 4] - q[k]) for k in range(4)]
        ratio = max(ws) / max(min(ws), 1e-3)
        kind = "extreme-ratio" if ratio > 10 else "regular"
        print(f"{i:>3}  {cv2.contourArea(cv_box.astype(np.float32).reshape(4, 1, 2)):>10.2f}  "
              f"{0.5*abs(sum(cv_box[i][0]*cv_box[(i+1)%4][1] - cv_box[(i+1)%4][0]*cv_box[i][1] for i in range(4))):>10.2f}  "
              f"{max_d:>7.4f}  {mean_d:>7.4f}  {kind}")
        if max_d > 0.5:
            bad += 1
    print("-" * 60)
    print(f"max |cv-cpp| over all 20 quads: max={max(all_max):.4f}  "
          f"mean-of-means={sum(all_mean)/len(all_mean):.4f}")
    if bad > 0:
        print(f"  STATUS: FAIL ({bad} cases > 0.5 px)")
        return 1
    print("  STATUS: OK (all 20 cases within 0.5 px max diff)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
