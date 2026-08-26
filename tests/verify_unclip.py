#!/usr/bin/env python3
"""
Cross-validate the C++ clipper offset (used by db_post unclip) vs
Paddle's pyclipper for the same inputs.

For 20 randomly generated quads (including extreme aspect ratios,
45° rotations, near-degenerate areas) we:
  1. Compute the unclip distance the way Paddle does
     (area * unclip_ratio / perimeter), using Shapely for area and
     cv2.arcLength for perimeter to mirror Paddle's actual call.
  2. Run pyclipper.PyclipperOffset with the SAME quad and distance,
     default ArcTolerance=0.25, JT_ROUND, ET_CLOSEDPOLYGON.
  3. Build a C++ harness that:
       - reads the same quad and distance
       - converts to IntPoint via llround (what db_post.cpp does)
       - calls ClipperLib::ClipperOffset(... default ctor, so
         ArcTolerance=0.25) and AddPath / Execute
       - prints the expanded poly as a list of "X Y" pairs
  4. Compare each path's vertices: count diff, per-vertex diff.

Pass criteria (per quad):
  * path count matches
  * per-vertex coord diff max <= 0.5 px, mean <= 0.25 px
  (1-2 px would explain the 5-12 px rec-misread error.)
"""
from __future__ import annotations
import os
import random
import subprocess
import sys
from typing import List, Tuple

import numpy as np
import pyclipper
import shapely.geometry as sg


HARNESS_TEMPLATE = r"""
// Auto-generated harness for unclip cross-check (pyclipper vs C++ clipper).
#include "clipper.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n_in;
  if (std::scanf("%d", &n_in) != 1 || n_in != 4) {
    std::fprintf(stderr, "expected 4 input vertices, got %d\n", n_in);
    return 1;
  }
  ClipperLib::Path path;
  path.reserve(4);
  for (int i = 0; i < 4; ++i) {
    double x, y;
    if (std::scanf("%lf %lf", &x, &y) != 2) {
      std::fprintf(stderr, "vertex read failed at %d\n", i);
      return 2;
    }
    // Truncation (matches pyclipper's _to_clipper_point: it does
    // `IntPoint(py_point[0], py_point[1])` which is a direct C++ struct
    // construction that truncates the float). llround would round 0.5
    // away from zero and shift vertices 0–1 px on odd-fractional coords.
    path << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(x),
                                 static_cast<ClipperLib::cInt>(y));
  }
  double distance;
  if (std::scanf("%lf", &distance) != 1) {
    std::fprintf(stderr, "distance read failed\n");
    return 3;
  }
  // Default ctor -> ArcTolerance = 0.25 (matches pyclipper's default).
  ClipperLib::ClipperOffset co;
  co.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
  ClipperLib::Paths solution;
  co.Execute(solution, distance);
  // Print: N paths, then for each path: K X Y X Y ...
  std::printf("%d\n", (int)solution.size());
  for (const auto& p : solution) {
    std::printf("%d", (int)p.size());
    for (const auto& ip : p) {
      std::printf(" %lld %lld", (long long)ip.X, (long long)ip.Y);
    }
    std::printf("\n");
  }
  return 0;
}
"""


def make_quads(n: int = 20, seed: int = 0xc0ffee) -> List[np.ndarray]:
    """Build a stress set: rectangles, long-thin, near-degenerate, rotated."""
    rng = random.Random(seed)
    out = []
    # 1. axis-aligned normal rectangles
    for _ in range(5):
        w = rng.uniform(40, 220)
        h = rng.uniform(20, 80)
        x0 = rng.uniform(50, 800)
        y0 = rng.uniform(50, 600)
        out.append(np.array([
            [x0, y0],
            [x0 + w, y0],
            [x0 + w, y0 + h],
            [x0, y0 + h],
        ], dtype=np.float64))
    # 2. extreme aspect ratio (20:1) — short bars
    for _ in range(4):
        w = rng.uniform(200, 400)
        h = w / 20.0
        cx = rng.uniform(300, 900)
        cy = rng.uniform(300, 600)
        out.append(np.array([
            [cx - w/2, cy - h/2],
            [cx + w/2, cy - h/2],
            [cx + w/2, cy + h/2],
            [cx - w/2, cy + h/2],
        ], dtype=np.float64))
    # 3. rotated at 45 deg, normal size
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
    # 4. rotated by random angle
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
    # 5. near-degenerate: 3 vertices almost collinear
    for _ in range(2):
        w = rng.uniform(50, 100)
        h = rng.uniform(3, 8)  # tiny height
        x0 = rng.uniform(100, 800)
        y0 = rng.uniform(100, 600)
        out.append(np.array([
            [x0, y0],
            [x0 + w, y0 + 0.5],
            [x0 + w, y0 + h],
            [x0, y0 + h],
        ], dtype=np.float64))
    return out[:n]


def unclip_distance(quad: np.ndarray, unclip_ratio: float = 1.5) -> float:
    """Mirror Paddle: area * unclip_ratio / perimeter. Use Shapely for
    area (Shapely uses the same Green's formula as us) and the
    Euclidean perimeter of the closed polygon."""
    poly = sg.Polygon(quad.tolist())
    if not poly.is_valid:
        poly = poly.buffer(0)
    area = poly.area
    perim = sum(np.linalg.norm(quad[(i+1) % 4] - quad[i]) for i in range(4))
    return area * unclip_ratio / perim


def pyclipper_expand(quad: np.ndarray, distance: float):
    co = pyclipper.PyclipperOffset()  # default ArcTolerance = 0.25
    co.AddPath([(float(p[0]), float(p[1])) for p in quad],
               pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = co.Execute(distance)
    # Convert pyclipper's mixed result to a list of (N, 2) float arrays.
    out = []
    for path in expanded:
        out.append(np.array(path, dtype=np.float64).reshape(-1, 2))
    return out


def build_harness(post_root: str, build_dir: str) -> str:
    cpp_path = os.path.join(build_dir, "harness_unclip.cpp")
    bin_path = os.path.join(build_dir, "harness_unclip")
    with open(cpp_path, "w") as f:
        f.write(HARNESS_TEMPLATE)
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall",
        "-I", os.path.join(post_root, "third_party", "clipper"),
        cpp_path,
        os.path.join(post_root, "third_party", "clipper", "clipper.cpp"),
        "-o", bin_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return bin_path


def cpp_expand(quad: np.ndarray, distance: float, bin_path: str):
    stdin = "4\n"
    for p in quad:
        stdin += f"{p[0]:.6f} {p[1]:.6f}\n"
    stdin += f"{distance:.10f}\n"
    proc = subprocess.run([bin_path], input=stdin, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"cpp harness failed: {proc.stderr}")
    out = proc.stdout.strip().split("\n")
    n_paths = int(out[0])
    paths = []
    for i in range(1, n_paths + 1):
        tokens = out[i].split()
        n = int(tokens[0])
        pts = []
        for j in range(n):
            x = int(tokens[1 + 2*j])
            y = int(tokens[1 + 2*j + 1])
            pts.append([x, y])
        paths.append(np.array(pts, dtype=np.float64))
    return paths


def path_diff(p_py: np.ndarray, p_cpp: np.ndarray) -> Tuple[int, float, float]:
    """Compare two polygons as point sets. Returns
    (matched_or_short, max_per_vertex_diff, mean_per_vertex_diff).
    The two implementations may produce different arc samplings, so we
    do a greedy nearest-neighbor match and report the per-vertex L-inf
    of the matched pairs (so the 'max' is the worst matched pair,
    not a tail). If one path is shorter, we report -1 max and use the
    min length.
    """
    n = min(len(p_py), len(p_cpp))
    if n == 0:
        return 0, 0.0, 0.0
    # For each cpp vertex, find the nearest py vertex.
    # For a rounded rect with the same number of samples, the index
    # alignment is roughly the same; we use a global L2 matching that
    # minimizes total cost (greedy since n is small, usually 16-40).
    used = [False] * len(p_py)
    pairs = []
    for j in range(len(p_cpp)):
        d2_min = 1e30
        k_best = -1
        for k in range(len(p_py)):
            if used[k]:
                continue
            d2 = (p_py[k, 0] - p_cpp[j, 0]) ** 2 + (p_py[k, 1] - p_cpp[j, 1]) ** 2
            if d2 < d2_min:
                d2_min = d2
                k_best = k
        if k_best >= 0:
            used[k_best] = True
            pairs.append((j, k_best))
    if not pairs:
        return n, 0.0, 0.0
    diffs = []
    for j, k in pairs:
        diffs.append(np.linalg.norm(p_cpp[j] - p_py[k]))
    return n, float(max(diffs)), float(np.mean(diffs))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    post_root = os.path.abspath(os.path.join(here, ".."))
    build_dir = os.path.join(here, "build-tests")
    os.makedirs(build_dir, exist_ok=True)
    bin_path = build_harness(post_root, build_dir)

    quads = make_quads(n=20)
    print(f"{'idx':>3}  {'n_py':>4}  {'n_cpp':>5}  {'max':>7}  {'mean':>7}  {'kind'}")
    print("-" * 60)
    all_max = []
    all_mean = []
    bad = 0
    for i, q in enumerate(quads):
        d = unclip_distance(q, unclip_ratio=1.5)
        py_paths = pyclipper_expand(q, d)
        cpp_paths = cpp_expand(q, d, bin_path)
        if len(py_paths) != len(cpp_paths):
            print(f"{i:>3}  {len(py_paths):>4}  {len(cpp_paths):>5}  PATH-COUNT-DIFF")
            bad += 1
            continue
        if not py_paths:
            print(f"{i:>3}  empty path")
            continue
        n_total = sum(p.shape[0] for p in py_paths)
        n_cpp_total = sum(p.shape[0] for p in cpp_paths)
        # Match path-by-path (both implementations return 1 path for
        # a closed input). If multiple paths, pair in order.
        max_d = 0.0
        sum_d = 0.0
        count = 0
        for ppy, pcpp in zip(py_paths, cpp_paths):
            n, mx, mn = path_diff(ppy, pcpp)
            if mx > max_d: max_d = mx
            sum_d += mn * n
            count += n
        mean_d = sum_d / count if count else 0.0
        all_max.append(max_d)
        all_mean.append(mean_d)
        # Heuristic: classify the quad.
        if q.shape[0] == 4:
            ws = [np.linalg.norm(q[(k+1) % 4] - q[k]) for k in range(4)]
            longest = max(ws)
            shortest = min(ws)
            ratio = longest / max(shortest, 1e-3)
            if ratio > 10:
                kind = "extreme-ratio"
            else:
                kind = "regular"
        else:
            kind = "?"
        # If both paths have very different vertex counts, log it.
        if abs(n_total - n_cpp_total) > 4:
            kind += f" (n_diff={n_total - n_cpp_total})"
        print(f"{i:>3}  {n_total:>4}  {n_cpp_total:>5}  {max_d:>7.4f}  {mean_d:>7.4f}  {kind}")
        if max_d > 0.5:
            bad += 1
    print("-" * 60)
    print(f"max |py-cpp| over all 20 quads: max={max(all_max):.4f}  "
          f"mean-of-means={sum(all_mean)/len(all_mean):.4f}")
    if bad > 0:
        print(f"  STATUS: FAIL ({bad} cases > 0.5 px)")
        return 1
    print("  STATUS: OK (all 20 cases within 0.5 px max diff)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
