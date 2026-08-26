#!/usr/bin/env python3
"""MNN variant build helper.

For each variant name + cmake args dict, run cmake with the existing
third_party/MNN sources into third_party/MNN/build_var/<name>/ and
build just the libMNN.a target (skip converter, test, demo, tools).

The variants are intentionally pure compile-time switches — we are
not modifying MNN sources, just changing MNN_USE_SSE / MNN_AVX512 /
MNN_SUPPORT_BF16 / MNN_OPENMP / MNN_USE_SPARSE_COMPUTE / etc.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MNN_ROOT = ROOT / "third_party" / "MNN"
BUILD_VAR_ROOT = MNN_ROOT / "build_var"


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.check_call(cmd, **kw)


def build_variant(name: str, extra_args: list[str], jobs: int = 8) -> Path:
    """Configure + build MNN with the given cmake flags into a
    variant build dir under third_party/MNN/build_var/<name>/.

    Returns the path to the resulting libMNN.a.
    """
    build_dir = BUILD_VAR_ROOT / name
    if (build_dir / "libMNN.a").exists():
        print(f"== variant '{name}': libMNN.a already exists at {build_dir/libMNN.a}, skipping build", flush=True)
        return build_dir / "libMNN.a"

    # Always start fresh for the variant dir — cache state from a
    # previous variant with different flags would silently apply.
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the existing build/ settings from the parent project's
    # CMakeCache but allow the variant to override specific flags.
    # We use a minimal flag set: CPU only, no converter, no test,
    # no demo, no benchmark, no codegen, no LLM, no diffusion.
    base = [
        "-DMNN_BUILD_SHARED_LIBS=OFF",
        "-DMNN_BUILD_CONVERTER=OFF",
        "-DMNN_BUILD_TOOLS=OFF",
        "-DMNN_BUILD_DEMO=OFF",
        "-DMNN_BUILD_BENCHMARK=OFF",
        "-DMNN_BUILD_TEST=OFF",
        "-DMNN_BUILD_CODEGEN=OFF",
        "-DMNN_BUILD_LLM=OFF",
        "-DMNN_BUILD_DIFFUSION=OFF",
        "-DMNN_BUILD_TRAIN=OFF",
        "-DMNN_BUILD_QUANTOOLS=OFF",
        "-DMNN_BUILD_MINI=OFF",
        "-DMNN_BUILD_PROTOBUFFER=OFF",
        "-DMNN_OPENCL=OFF",
        "-DMNN_OPENGL=OFF",
        "-DMNN_VULKAN=OFF",
        "-DMNN_CUDA=OFF",
        "-DMNN_TENSORRT=OFF",
        "-DMNN_COREML=OFF",
        "-DMNN_NNAPI=OFF",
        "-DMNN_ARM82=OFF",
        "-DMNN_METAL=OFF",
        "-DMNN_ONEDNN=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS=-w",  # suppress MNN's internal warnings to speed compile
        f"-DMNN_SEP_BUILD=OFF",
    ] + extra_args

    t0 = time.time()
    print(f"== variant '{name}': cmake ...")
    run(["cmake", str(MNN_ROOT)] + base, cwd=build_dir)

    t1 = time.time()
    print(f"== variant '{name}': cmake took {t1 - t0:.1f}s; building libMNN.a (j={jobs}) ...")
    run(["make", "-j", str(jobs), "MNN"], cwd=build_dir)

    t2 = time.time()
    print(f"== variant '{name}': make took {t2 - t1:.1f}s; total {t2 - t0:.1f}s")

    lib = build_dir / "libMNN.a"
    if not lib.exists():
        sys.exit(f"variant '{name}': libMNN.a not found at {lib}")
    print(f"== variant '{name}': OK, lib at {lib}, size {lib.stat().st_size // 1024} KB", flush=True)
    return lib


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="variant name (used as build dir)")
    # Allow passing the cmake flags as a single quoted string
    # (avoids argparse splitting on the equals sign):
    #   --flags "-DMNN_USE_SSE=OFF -DMNN_AVX512=ON"
    p.add_argument("--flags", required=True,
                   help="space-separated cmake -D... flags")
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    extra_args = args.flags.split()
    build_variant(args.name, extra_args, jobs=args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
