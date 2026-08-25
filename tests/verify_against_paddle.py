#!/usr/bin/env python3
"""
Cross-validate the C++ postprocess against the PaddleOCR python reference.
We instantiate the C++ test binary, dump the box for the standard white-rect
case, and compare against PaddleOCR's boxes_from_bitmap output.

Run:  python3 tests/verify_against_paddle.py
"""
from __future__ import annotations
import importlib.util
import os
import subprocess
import sys
import types

import numpy as np


def load_paddle_db():
    """Load PaddleOCR's DBPostProcess without importing the broken postprocess
    package init (which pulls in skimage). We manually load db_postprocess.py
    and stub the paddle module."""
    spec = importlib.util.spec_from_file_location(
        "db_postprocess",
        "/root/PaddleOCR/ppocr/postprocess/db_postprocess.py",
    )
    mod = importlib.util.module_from_spec(spec)
    paddle_stub = types.ModuleType("paddle")
    paddle_stub.Tensor = object
    sys.modules["paddle"] = paddle_stub
    paddle_nn_stub = types.ModuleType("paddle.nn")
    paddle_nn_stub.functional = types.SimpleNamespace()
    sys.modules["paddle.nn"] = paddle_nn_stub
    sys.modules["paddle.nn.functional"] = paddle_nn_stub.functional
    spec.loader.exec_module(mod)
    return mod


def paddle_boxes(prob, thresh, box_thresh, unclip_ratio, dest_w, dest_h):
    mod = load_paddle_db()
    pp = mod.DBPostProcess(
        thresh=thresh,
        box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
        max_candidates=1000,
        use_dilation=False,
        score_mode="fast",
        box_type="quad",
    )
    bitmap = (prob > thresh).astype(np.uint8)
    boxes, scores = pp.boxes_from_bitmap(prob, bitmap, dest_w, dest_h)
    return boxes, scores


def run_cpp_test():
    """Run our C++ test binary and capture the db_postprocess output."""
    here = os.path.dirname(os.path.abspath(__file__))
    post_root = os.path.abspath(os.path.join(here, ".."))
    test_bin = os.path.join(post_root, "tests", "build-tests", "test_post")
    if not os.path.exists(test_bin):
        # Try to build it.
        subprocess.run(
            [
                "cmake",
                "-S",
                os.path.join(here),
                "-B",
                os.path.join(here, "build-tests"),
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )
        subprocess.run(
            [
                "cmake",
                "--build",
                os.path.join(here, "build-tests"),
                "-j",
            ],
            check=True,
        )
    out = subprocess.run([test_bin], capture_output=True, text=True)
    if out.returncode != 0:
        print("STDOUT:", out.stdout)
        print("STDERR:", out.stderr)
        raise RuntimeError("test_post failed")
    return out.stdout


def parse_cpp_db_line(stdout: str):
    """Find the 'test_db_postprocess_white_rect: ... poly=[...]' line and
    parse the poly + score."""
    for line in stdout.splitlines():
        if "test_db_postprocess_white_rect" not in line:
            continue
        # box info is on this line and the next.
        if "boxes=" not in line:
            continue
        # Score
        score = float(line.split("score=")[1].split()[0])
        # Find poly line
        return score
    raise RuntimeError("could not parse C++ output:\n" + stdout)


def parse_cpp_db_poly(stdout: str):
    for line in stdout.splitlines():
        if "poly=" not in line:
            continue
        # poly=[40,0, 278,0, 278,158, 40,158]
        inside = line.split("poly=[")[1].split("]")[0]
        nums = [int(x.strip()) for x in inside.replace(",", " ").split()]
        assert len(nums) == 8
        return [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
    raise RuntimeError("could not parse C++ poly:\n" + stdout)


def main():
    H, W = 80, 160
    prob = np.full((H, W), 0.05, dtype=np.float32)
    prob[20:60, 40:120] = 0.9
    boxes, scores = paddle_boxes(
        prob, thresh=0.3, box_thresh=0.5, unclip_ratio=1.5, dest_w=320, dest_h=160
    )
    print(f"Paddle: n_boxes={len(boxes)} scores={scores}")
    if len(boxes) > 0:
        b = boxes[0].reshape(-1, 2)
        # boxes_from_bitmap returns boxes in Paddle canonical order:
        # [TL, TR, BR, BL] (after get_mini_boxes sort-by-x + top/bot split).
        canonical = [(float(b[0][0]), float(b[0][1])),
                     (float(b[1][0]), float(b[1][1])),
                     (float(b[2][0]), float(b[2][1])),
                     (float(b[3][0]), float(b[3][1]))]
        print("  Paddle canonical poly:", canonical)

    stdout = run_cpp_test()
    cpp_score = parse_cpp_db_line(stdout)
    cpp_poly = parse_cpp_db_poly(stdout)
    print(f"C++:    score={cpp_score:.2f} canonical poly={cpp_poly}")

    if len(boxes) > 0:
        # Compare canonical orderings.
        ok = True
        for (cx, cy), (px, py) in zip(cpp_poly, canonical):
            if abs(cx - px) > 1 or abs(cy - py) > 1:
                ok = False
                print(
                    f"  MISMATCH: cpp=({cx},{cy}) paddle=({px},{py})"
                )
        if ok:
            print("  MATCH: cpp canonical poly equals paddle canonical poly (within 1 px).")
        else:
            print("  FAIL: corner mismatch")
            sys.exit(1)
    if abs(cpp_score - scores[0]) > 0.01:
        print(f"  FAIL: score mismatch (cpp={cpp_score}, paddle={scores[0]})")
        sys.exit(1)
    print("  MATCH: score within 0.01.")
    print("OK")


if __name__ == "__main__":
    main()
