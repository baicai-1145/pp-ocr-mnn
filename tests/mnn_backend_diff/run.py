#!/usr/bin/env python3
# POST-6 driver runner.
#
# Builds and runs tests/mnn_backend_diff/driver.cpp on the m2-num det
# inputs and prints a Python-side diff table for cross-check.
#
# Usage:  python3 tests/mnn_backend_diff/run.py
#   (override with env vars DRIVER_IN / DRIVER_REF / DRIVER_MODEL)
"""
POST-6 backend-precision diff runner.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"
MNN_INCLUDE = THIRD_PARTY / "MNN" / "include"
MNN_SOURCE = THIRD_PARTY / "MNN" / "source"
MNN_LIB = THIRD_PARTY / "MNN" / "build" / "libMNN.a"

# Inputs / outputs (overridable via env)
DRIVER_IN = Path(os.environ.get("DRIVER_IN",
                                 "/tmp/m2num/det_input_paddle.npy"))
DRIVER_REF = Path(os.environ.get("DRIVER_REF",
                                  "/tmp/m2num/det_output_paddle.npy"))
DRIVER_MODEL = Path(os.environ.get("DRIVER_MODEL",
                                    "/root/pp-ocr-mnn/models/PP-OCRv6_tiny_det.mnn"))

# Where to put the binary and outputs
BIN = Path("/tmp/mnn_backend_driver")
OUT_PREFIX = Path("/tmp/mnn_diff")

COMBOS = ["cpu_normal", "cpu_high", "cpu_low", "cpu_low_bf16"]


def build_driver() -> None:
    if not MNN_LIB.exists():
        sys.exit(f"missing MNN library: {MNN_LIB}; build MNN first")
    src = Path(__file__).parent / "driver.cpp"
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-I", str(MNN_INCLUDE),
        "-I", str(MNN_SOURCE),
        str(src),
        str(MNN_LIB),
        "-lpthread", "-ldl", "-lm",
        "-o", str(BIN),
    ]
    print("building:", " ".join(cmd))
    subprocess.check_call(cmd)


def run_driver() -> None:
    cmd = [
        str(BIN),
        str(DRIVER_MODEL),
        str(DRIVER_IN),
        str(OUT_PREFIX),
        "4",  # num_threads (MNN default = 4 if 0; we set 4 explicitly)
        str(DRIVER_REF),
    ]
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)


def cross_check() -> None:
    ref = np.load(DRIVER_REF)
    print()
    print("=== Python-side cross-check vs", DRIVER_REF, "===")
    print(f"{'combo':14s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'%>0.01':>9s}  {'%>0.1':>9s}")
    for name in COMBOS:
        p = OUT_PREFIX.with_name(OUT_PREFIX.name + f"__{name}.npy")
        if not p.exists():
            print(f"{name:14s}  file not found: {p}")
            continue
        a = np.load(p)
        d = np.abs(a - ref)
        max_abs = float(d.max())
        mean_abs = float(d.mean())
        gt001 = float((d > 0.01).mean()) * 100.0
        gt01  = float((d > 0.10).mean()) * 100.0
        print(f"{name:14s}  {max_abs:10.6f}  {mean_abs:10.6f}  "
              f"{gt001:8.2f}%  {gt01:8.2f}%")
    # Also check: are the 4 outputs byte-identical?
    arrs = [np.load(OUT_PREFIX.with_name(OUT_PREFIX.name + f"__{n}.npy"))
            for n in COMBOS
            if (OUT_PREFIX.with_name(OUT_PREFIX.name + f"__{n}.npy")).exists()]
    if len(arrs) >= 2:
        ref2 = arrs[0]
        all_equal = all(np.array_equal(a, ref2) for a in arrs[1:])
        if all_equal:
            print()
            print(">> All 4 MNN precisions produce BYTE-IDENTICAL output.")
        else:
            print()
            print(">> Precisions diverge (the `precision` switch has effect).")


def main() -> int:
    build_driver()
    run_driver()
    cross_check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
