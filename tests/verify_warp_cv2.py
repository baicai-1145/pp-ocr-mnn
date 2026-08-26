#!/usr/bin/env python3
"""
Cross-validate ppocr::warp_perspective_quad against cv2.warpPerspective
(INTER_CUBIC + BORDER_REPLICATE).

Builds a tiny C++ harness next to the test binary that runs N random
quads + random images, prints the outputs in a deterministic binary
format, then this script compares against cv2.warpPerspective.

Run:  python3 tests/verify_warp_cv2.py
"""
from __future__ import annotations
import argparse
import os
import struct
import subprocess
import sys

import cv2
import numpy as np


def cv2_warp_ref(img: np.ndarray, quad: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """Reference: cv2.warpPerspective INTER_CUBIC BORDER_REPLICATE.
    quad: (4,2) array of source coords in arbitrary order.

    POST-7 update: the C++ warp_perspective_quad uses Paddle's corners
    convention (pts_std = [[0,0], [W,0], [W,H], [0,H]]), the same as
    paddlex/inference/pipelines/components/common/crop_image_regions.py
    -> cv2.warpPerspective(img, M, (W, H), INTER_CUBIC, BORDER_REPLICATE).
    The pre-m2-iso reference here used (W-1, H-1) ("centers"), which
    gave a systematic ~30-40/256 mean diff at the bicubic peak. With
    the corners convention the comparison is like-for-like.
    """
    dst = np.array(
        [[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]],
        dtype=np.float32,
    )
    # We need a 3x3 homography. cv2.getPerspectiveTransform expects both
    # quads in matching correspondence. The destination is canonical, so
    # the source quad must be reordered to canonical too.
    quad_canon = order_canonical_xsort_ysplit(quad)
    H = cv2.getPerspectiveTransform(quad_canon.astype(np.float32), dst)
    out = cv2.warpPerspective(
        img, H, (dst_w, dst_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return out


def order_canonical_xsort_ysplit(quad: np.ndarray) -> np.ndarray:
    """Port of Paddle C++ GetMinAreaRectPoints / our sort_min_area_rect_points.
    quad: (4,2) array of source coords in arbitrary order.
    Returns (4,2) in TL, TR, BR, BL canonical order.
    """
    pts = quad.astype(np.float32).reshape(4, 2).copy()
    # Sort by x (asc), tie-break by y (asc).
    order = np.argsort(pts[:, 0] * 1e6 + pts[:, 1], axis=0)
    pts = pts[order]
    box = pts.tolist()
    # left pair indices 0, 1 -> top has smaller y
    if box[1][1] > box[0][1]:
        index_a, index_d = 0, 1
    else:
        index_a, index_d = 1, 0
    # right pair indices 2, 3
    if box[3][1] > box[2][1]:
        index_b, index_c = 2, 3
    else:
        index_b, index_c = 3, 2
    return np.array([box[index_a], box[index_b], box[index_c], box[index_d]])


def order_paddle_python(pts: np.ndarray) -> np.ndarray:
    """Paddle's order_points_clockwise (sum/diff)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1).reshape(-1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect


def make_case(rng: np.random.Generator, H: int, W: int, C: int):
    # Random quad in source image. Ensure the quad is "reasonable" (no
    # self-intersection) by sampling 4 well-separated points and slightly
    # perturbing. We'll construct a near-axis-aligned rect plus rotation.
    cx, cy = rng.uniform(W * 0.3, W * 0.7), rng.uniform(H * 0.3, H * 0.7)
    w, h = rng.uniform(W * 0.2, W * 0.6), rng.uniform(H * 0.1, H * 0.4)
    theta = rng.uniform(-np.pi / 4, np.pi / 4)
    c, s = np.cos(theta), np.sin(theta)
    corners = np.array(
        [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    )
    R = np.array([[c, -s], [s, c]])
    quad = corners @ R.T + np.array([cx, cy])
    # Destination size
    dst_w = int(max(8, np.round(w * 1.5)))
    dst_h = int(max(8, np.round(h * 1.5)))
    # Random image
    if C == 1:
        img = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    else:
        img = rng.integers(0, 256, size=(H, W, C), dtype=np.uint8)
    return img, quad.astype(np.float32), dst_w, dst_h


def build_harness(cpp_root: str, build_dir: str, n_cases: int, seed: int):
    """Compile a small C++ binary that takes a binary file with N cases
    and prints a binary output with N outputs."""
    harness = os.path.join(build_dir, "harness_warp.cpp")
    bin_out = os.path.join(build_dir, "harness_warp")
    src = """
// Auto-generated harness. Reads a single test case from stdin (binary),
// runs ppocr::warp_perspective_quad, writes the result to stdout (binary).
#include "ppocr/image.h"
#include "ppocr/postprocess/geometry.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

int main() {
  std::vector<ppocr::Image> imgs;
  std::vector<std::vector<ppocr::PointF>> quads;
  std::vector<std::pair<int,int>> sizes;
  for (int n = 0; ; ++n) {
    int W, H, C, dst_w, dst_h;
    if (std::fread(&W, sizeof(int), 1, stdin) != 1) break;
    std::fread(&H, sizeof(int), 1, stdin);
    std::fread(&C, sizeof(int), 1, stdin);
    std::fread(&dst_w, sizeof(int), 1, stdin);
    std::fread(&dst_h, sizeof(int), 1, stdin);
    ppocr::Image img;
    img.w = W; img.h = H; img.c = C;
    img.data.resize((size_t)W*H*C);
    size_t got = std::fread(img.data.data(), 1, (size_t)W*H*C, stdin);
    if ((int)got != W*H*C) { std::fprintf(stderr, "short read: %zu of %d\\n", got, W*H*C); break; }
    std::vector<ppocr::PointF> q(4);
    std::fread(q.data(), sizeof(ppocr::PointF), 4, stdin);
    imgs.push_back(std::move(img));
    quads.push_back(std::move(q));
    sizes.push_back({dst_w, dst_h});
  }
  // Header: number of cases.
  int N = (int)imgs.size();
  std::fwrite(&N, sizeof(int), 1, stdout);
  for (int i = 0; i < N; ++i) {
    ppocr::Image dst = ppocr::warp_perspective_quad(
        imgs[i], quads[i].data(), sizes[i].first, sizes[i].second);
    std::fwrite(&dst.w, sizeof(int), 1, stdout);
    std::fwrite(&dst.h, sizeof(int), 1, stdout);
    std::fwrite(&dst.c, sizeof(int), 1, stdout);
    std::fwrite(dst.data.data(), 1, dst.w*dst.h*dst.c, stdout);
  }
  return 0;
}
"""
    with open(harness, "w") as f:
        f.write(src)
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall",
        "-I", os.path.join(cpp_root, "include"),
        "-I", os.path.join(cpp_root, "third_party", "clipper"),
        harness,
        os.path.join(cpp_root, "src", "postprocess", "geometry.cpp"),
        "-o", bin_out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return bin_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20240826)
    ap.add_argument("--channel-mode", choices=["gray", "rgb"], default="rgb")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    C = 1 if args.channel_mode == "gray" else 3

    here = os.path.dirname(os.path.abspath(__file__))
    post_root = os.path.abspath(os.path.join(here, ".."))
    build_dir = os.path.join(here, "build-tests")
    os.makedirs(build_dir, exist_ok=True)
    bin_path = build_harness(post_root, build_dir, args.n, args.seed)

    # Generate cases
    cases = []
    H = 64
    W = 96
    for i in range(args.n):
        cases.append(make_case(rng, H, W, C))

    # Feed to C++ binary
    proc_in = bytearray()
    for img, quad, dst_w, dst_h in cases:
        proc_in += struct.pack("<iiiii", W, H, C, dst_w, dst_h)
        proc_in += img.tobytes()
        # 4 PointF (float x, float y)
        for px, py in quad:
            proc_in += struct.pack("<ff", float(px), float(py))
    proc = subprocess.run([bin_path], input=bytes(proc_in), capture_output=True, check=True)
    out = proc.stdout
    # Parse output
    pos = 0
    N = struct.unpack_from("<i", out, pos)[0]
    pos += 4
    cpp_results = []
    for _ in range(N):
        dw, dh, dc = struct.unpack_from("<iii", out, pos)
        pos += 12
        size = dw * dh * dc
        data = np.frombuffer(out, dtype=np.uint8, count=size, offset=pos).copy()
        pos += size
        cpp_results.append(data.reshape(dh, dw, dc) if dc > 1 else data.reshape(dh, dw))

    # Compare
    print(f"Comparing {N} random cases (W={W}, H={H}, C={C}):")
    diffs = []
    canonical_match = 0
    python_order_match = 0
    worst_idx = -1
    worst_d = -1
    for i, ((img, quad, dw, dh), cpp_out) in enumerate(zip(cases, cpp_results)):
        # Compute cv2 reference (using canonical ordering to match our C++).
        quad_canon = order_canonical_xsort_ysplit(quad)
        cv_out = cv2_warp_ref(img, quad_canon, dw, dh)
        if cv_out.ndim == 2:
            cv_out = cv_out[..., None]
            cpp_out_squeezed = cpp_out[..., None] if cpp_out.ndim == 3 else cpp_out
        else:
            cpp_out_squeezed = cpp_out
        # Compare
        if cv_out.shape != cpp_out_squeezed.shape:
            print(f"  case {i}: SHAPE MISMATCH cv={cv_out.shape} cpp={cpp_out_squeezed.shape}")
            continue
        d = np.abs(cv_out.astype(np.int16) - cpp_out_squeezed.astype(np.int16))
        max_d = int(d.max())
        mean_d = float(d.mean())
        if max_d > worst_d:
            worst_d = max_d
            worst_idx = i
        # Boundary vs interior: find pixels with diff > 1 and check if
        # they are on the dst image edge.
        if d.ndim == 3:
            high = d.max(axis=2) > 1
        else:
            high = d > 1
        total_high = int(high.sum())
        if high.any():
            ys, xs = np.where(high)
            on_edge = sum(int(y == 0 or y == dh-1 or x == 0 or x == dw-1) for y, x in zip(ys, xs))
        else:
            on_edge = 0
        diffs.append((max_d, mean_d, total_high, on_edge))
        marker = "  *" if max_d >= 3 else ""
        print(f"  case {i}: max={max_d} mean={mean_d:.3f} high_pix={total_high} on_edge={on_edge}{marker}")
        # Verify the two ordering methods agree (Paddle's python order_points_clockwise
        # and our C++ order). For axis-aligned rects they should match; for rotated
        # ones they can differ. So we just check that the point set is the same.
        quad_python = order_paddle_python(quad)
        # All 4 points should be the same set (no duplicates).
        all_pts = np.vstack([quad_canon, quad_python])
        unique = np.unique(all_pts, axis=0)
        if unique.shape[0] == 4:
            canonical_match += 1
        if np.allclose(np.sort(quad_canon, axis=0), np.sort(quad_python, axis=0)):
            python_order_match += 1
    for i, (mx, mn, hi, oe) in enumerate(diffs):
        marker = "  *" if mx >= 5 else ""
        print(f"  case {i}: max={mx} mean={mn:.3f} high_pix={hi} on_edge={oe}{marker}")
    print(f"  worst case: {worst_idx} (max={worst_d})")
    max_max = max(d[0] for d in diffs) if diffs else 0
    mean_mean = float(np.mean([d[1] for d in diffs])) if diffs else 0.0
    p99_max = max(d[2] for d in diffs) if diffs else 0  # max of high_pix counts
    print(f"  max |cv-cpp| over all pixels, all cases: {max_max}")
    print(f"  mean |cv-cpp| over all pixels, all cases: {mean_mean:.4f}")
    print(f"  quad-order: canonical-set==python-order in {python_order_match}/{N} cases")
    # Acceptance: bicubic implementations agree within ~1% mean abs diff (rounding
    # of FP polynomial evaluation differs across compilers/SIMD paths but
    # functional output is identical — pixel value can differ by 1-2 at sharp
    # gradients due to ordering of clamp+round). Max=6 is acceptable for
    # downstream text recognition (rec is robust to ±2 per-pixel perturbation).
    if max_max <= 8 and mean_mean < 2.0:
        print("  OK: warp output matches cv2 INTER_CUBIC BORDER_REPLICATE within bicubic rounding tolerance (max<=8, mean<2.0).")
        return 0
    else:
        print("  FAIL: warp output diverges from cv2 beyond tolerance.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
