#!/usr/bin/env python3
# POST-6 / M3-CUDA backend-precision diff runner.
#
# Builds and runs tests/mnn_backend_diff/driver.cpp on the m2-num det
# inputs and prints a Python-side diff table for cross-check.
#
# Default: CPU 4-precision sweep via the static CPU-only libMNN.a.
# M3-CUDA: build a shared-lib MNN from `build_cuda/` (or `build_vulkan/`)
# and run with --forward 2 (CUDA) or --forward 7 (Vulkan).
#
# Usage:
#   python3 tests/mnn_backend_diff/run.py                # CPU sweep
#   python3 tests/mnn_backend_diff/run.py --cuda         # single CUDA combo
#   python3 tests/mnn_backend_diff/run.py --vulkan       # single Vulkan combo
#
# Env overrides:
#   DRIVER_IN       /tmp/m2num/det_input_paddle.npy
#   DRIVER_REF      /tmp/m2num/det_output_paddle.npy
#   DRIVER_MODEL    /root/pp-ocr-mnn/models/PP-OCRv6_tiny_det.mnn
"""
POST-6 / M3-CUDA backend-precision diff runner.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"
MNN_INCLUDE = THIRD_PARTY / "MNN" / "include"
MNN_SOURCE = THIRD_PARTY / "MNN" / "source"
MNN_BUILD_CPU = THIRD_PARTY / "MNN" / "build" / "libMNN.a"
MNN_BUILD_CUDA = THIRD_PARTY / "MNN" / "build_cuda" / "libMNN.so"
MNN_BUILD_CUDA_BACKEND = THIRD_PARTY / "MNN" / "build_cuda" / "source" / "backend" / "cuda" / "libMNN_Cuda_Main.so"
MNN_BUILD_VULKAN = THIRD_PARTY / "MNN" / "build_vulkan" / "libMNN.so"
MNN_BUILD_VULKAN_BACKEND = THIRD_PARTY / "MNN" / "build_vulkan" / "source" / "backend" / "vulkan" / "libMNN_Vulkan.so"

# Inputs / outputs (overridable via env)
DRIVER_IN = Path(os.environ.get("DRIVER_IN",
                                 "/tmp/m2num/det_input_paddle.npy"))
DRIVER_REF = Path(os.environ.get("DRIVER_REF",
                                  "/tmp/m2num/det_output_paddle.npy"))
DRIVER_MODEL = Path(os.environ.get("DRIVER_MODEL",
                                    "/root/pp-ocr-mnn/models/PP-OCRv6_tiny_det.mnn"))

# Where to put binaries and outputs
BIN_CPU = Path("/tmp/mnn_backend_driver")
BIN_CUDA = Path("/tmp/mnn_cuda_driver")
BIN_VULKAN = Path("/tmp/mnn_vulkan_driver")
OUT_PREFIX_CPU = Path("/tmp/mnn_diff")
OUT_PREFIX_CUDA = Path("/tmp/mnn_diff_cuda")
OUT_PREFIX_VULKAN = Path("/tmp/mnn_diff_vulkan")

CPU_COMBOS = ["cpu_normal", "cpu_high", "cpu_low", "cpu_low_bf16"]


def build_cpu_driver() -> None:
    if not MNN_BUILD_CPU.exists():
        sys.exit(f"missing MNN CPU library: {MNN_BUILD_CPU}; build MNN first")
    src = Path(__file__).parent / "driver.cpp"
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-I", str(MNN_INCLUDE),
        "-I", str(MNN_SOURCE),
        str(src),
        str(MNN_BUILD_CPU),
        "-lpthread", "-ldl", "-lm",
        "-o", str(BIN_CPU),
    ]
    print("building CPU driver:", " ".join(cmd))
    subprocess.check_call(cmd)


def build_shared_driver(lib_path: Path, backend_lib: Path, out_bin: Path,
                        mnn_lib_label: str) -> None:
    """Link a shared-lib MNN (CUDA / Vulkan) plus its backend .so."""
    if not lib_path.exists():
        sys.exit(f"missing MNN library: {lib_path}")
    if not backend_lib.exists():
        sys.exit(f"missing backend library: {backend_lib}")
    src = Path(__file__).parent / "driver.cpp"
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-I", str(MNN_INCLUDE),
        "-I", str(MNN_SOURCE),
        str(src),
        "-L", str(lib_path.parent),
        "-lMNN",
        f"-Wl,-rpath,{lib_path.parent}",
        f"-Wl,-rpath,{backend_lib.parent}",
        "-Wl,--no-as-needed", f"-l{backend_lib.name.split('lib')[-1].split('.so')[0]}",
        "-L", str(backend_lib.parent),
        "-lpthread", "-ldl", "-lm",
        "-o", str(out_bin),
    ]
    print(f"building {mnn_lib_label} driver:", " ".join(cmd))
    subprocess.check_call(cmd)


def run_driver(bin_path: Path, out_prefix: Path, combos: list[str],
               forward_type: int = 0, single: str | None = None,
               mnn_lib_label: str = "cpu") -> None:
    cmd = [
        str(bin_path),
        "--forward", str(forward_type),
    ]
    if single:
        cmd += ["--single", single]
    cmd += [
        "--mnn-lib", mnn_lib_label,
        str(DRIVER_MODEL),
        str(DRIVER_IN),
        str(out_prefix),
        "4",  # num_threads
        str(DRIVER_REF),
    ]
    print(f"running {mnn_lib_label} driver:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, end="")
    if res.returncode != 0:
        print(res.stderr, end="", file=sys.stderr)
        print(f"[{mnn_lib_label}] driver exited with rc={res.returncode}")


def cross_check(out_prefix: Path, combos: list[str]) -> None:
    if not DRIVER_REF.exists():
        return
    ref = np.load(DRIVER_REF)
    print()
    print(f"=== Python-side cross-check vs {DRIVER_REF} ===")
    print(f"{'combo':14s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'%>0.01':>9s}  {'%>0.1':>9s}")
    for name in combos:
        p = out_prefix.with_name(out_prefix.name + f"__{name}.npy")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true",
                    help="Run MNN CUDA backend (requires build_cuda/)")
    ap.add_argument("--vulkan", action="store_true",
                    help="Run MNN Vulkan backend (requires build_vulkan/)")
    ap.add_argument("--no-cpu", action="store_true",
                    help="Skip the CPU 4-precision sweep")
    args = ap.parse_args()

    if not args.no_cpu:
        build_cpu_driver()
        # Clean previous outputs
        for n in CPU_COMBOS:
            p = OUT_PREFIX_CPU.with_name(OUT_PREFIX_CPU.name + f"__{n}.npy")
            if p.exists():
                p.unlink()
        run_driver(BIN_CPU, OUT_PREFIX_CPU, CPU_COMBOS,
                   forward_type=0, mnn_lib_label="cpu")
        cross_check(OUT_PREFIX_CPU, CPU_COMBOS)

    if args.cuda:
        if not MNN_BUILD_CUDA.exists():
            print(f"ERROR: {MNN_BUILD_CUDA} missing; build MNN with -DMNN_CUDA=ON first")
            return 1
        build_shared_driver(MNN_BUILD_CUDA, MNN_BUILD_CUDA_BACKEND,
                            BIN_CUDA, "cuda")
        p = OUT_PREFIX_CUDA.with_name(OUT_PREFIX_CUDA.name + "__cuda.npy")
        if p.exists():
            p.unlink()
        run_driver(BIN_CUDA, OUT_PREFIX_CUDA, ["cuda"],
                   forward_type=2, single="cuda", mnn_lib_label="cuda")
        cross_check(OUT_PREFIX_CUDA, ["cuda"])

    if args.vulkan:
        if not MNN_BUILD_VULKAN.exists():
            print(f"ERROR: {MNN_BUILD_VULKAN} missing; build MNN with -DMNN_VULKAN=ON first")
            return 1
        build_shared_driver(MNN_BUILD_VULKAN, MNN_BUILD_VULKAN_BACKEND,
                            BIN_VULKAN, "vulkan")
        p = OUT_PREFIX_VULKAN.with_name(OUT_PREFIX_VULKAN.name + "__vulkan.npy")
        if p.exists():
            p.unlink()
        run_driver(BIN_VULKAN, OUT_PREFIX_VULKAN, ["vulkan"],
                   forward_type=7, single="vulkan", mnn_lib_label="vulkan")
        cross_check(OUT_PREFIX_VULKAN, ["vulkan"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
