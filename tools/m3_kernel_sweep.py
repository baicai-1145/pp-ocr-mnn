#!/usr/bin/env python3
"""MNN kernel sweep runner.

For each (variant_name, cmake_flags) pair, build the variant, rebuild
the mnn_backend_diff driver linked to that libMNN.a, run it on
/tmp/m2num/det_input_paddle.npy with /tmp/m2num/det_output_paddle.npy
as the reference, and print a summary table.

Also rebuilds the main ppocr_cli linking the variant libMNN.a and
runs it on ja/00 to count boxes.

Variants to test (per spec):
  (a) no-sse:     MNN_USE_SSE=OFF   (naive kernel)
  (b) avx512:     MNN_AVX512=ON     (no-op on AMD EPYC 7502 but codegen differs)
  (c) bf16:       MNN_SUPPORT_BF16=ON
  (d) openmp:     MNN_OPENMP=ON
  (e) no-sparse:  MNN_USE_SPARSE_COMPUTE=OFF
  (f) low-mem:    MNN_LOW_MEMORY=ON
  (g) sse-fp16:   MNN_SSE_USE_FP16_INSTEAD=ON (requires BF16)

OPENBLAS has no MNN option (verified by grep of MNN/CMakeLists.txt:
no MNN_OPENBLAS variable; no third_party/openblas link). We document
this finding and skip (c) for OPENBLAS.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MNN_ROOT = ROOT / "third_party" / "MNN"
BUILD_VAR_ROOT = MNN_ROOT / "build_var"
MNN_INCLUDE = MNN_ROOT / "include"
MNN_SOURCE = MNN_ROOT / "source"
DRIVER_SRC = ROOT / "tests" / "mnn_backend_diff" / "driver.cpp"

# Common CPU-only cmake base
BASE_FLAGS = [
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
    "-DCMAKE_CXX_FLAGS=-w",
    "-DMNN_SEP_BUILD=OFF",
]

# Common defaults for all variants (then variant-specific flags
# override these)
COMMON_DEFAULTS = [
    "-DMNN_USE_SSE=ON",
    "-DMNN_AVX512=OFF",
    "-DMNN_SUPPORT_BF16=OFF",
    "-DMNN_OPENMP=OFF",
    "-DMNN_USE_THREAD_POOL=ON",
    "-DMNN_USE_SPARSE_COMPUTE=ON",
    "-DMNN_LOW_MEMORY=OFF",
    "-DMNN_SSE_USE_FP16_INSTEAD=OFF",
]


def build_variant(name: str, flags: list[str], jobs: int = 12) -> Path:
    """Build a variant. Returns the libMNN.a path."""
    build_dir = BUILD_VAR_ROOT / name
    lib = build_dir / "libMNN.a"
    if lib.exists():
        print(f"== variant '{name}': libMNN.a exists, skipping build", flush=True)
        return lib
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    extra = BASE_FLAGS + COMMON_DEFAULTS + flags
    t0 = time.time()
    print(f"== variant '{name}': flags = {flags}", flush=True)
    subprocess.check_call(
        ["cmake", str(MNN_ROOT)] + extra,
        cwd=build_dir,
    )
    t1 = time.time()
    print(f"== variant '{name}': cmake {t1-t0:.1f}s; make -j{jobs} MNN ...", flush=True)
    subprocess.check_call(["make", "-j", str(jobs), "MNN"], cwd=build_dir)
    t2 = time.time()
    if not lib.exists():
        sys.exit(f"variant '{name}': libMNN.a not built at {lib}")
    print(f"== variant '{name}': built {lib} ({lib.stat().st_size//1024} KB) in {t2-t0:.1f}s", flush=True)
    return lib


def build_driver(lib: Path, out_bin: Path) -> None:
    """Build the diff driver linked to the given libMNN.a."""
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-I", str(MNN_INCLUDE), "-I", str(MNN_SOURCE),
        str(DRIVER_SRC), str(lib),
        "-lpthread", "-ldl", "-lm",
        "-o", str(out_bin),
    ]
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd)


def run_driver(bin_path: Path, out_prefix: Path, num_threads: int = 4) -> dict:
    """Run the diff driver on the m2num det input; return diff stats
    for cpu_normal (the only precision that actually uses the
    non-trivial CPU backend on x86_64)."""
    cmd = [
        str(bin_path),
        str(ROOT / "models" / "PP-OCRv6_tiny_det.mnn"),
        "/tmp/m2num/det_input_paddle.npy",
        str(out_prefix),
        str(num_threads),
        "/tmp/m2num/det_output_paddle.npy",
    ]
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = r.stdout
    # parse the diff table line for cpu_normal
    for line in out.splitlines():
        if line.startswith("cpu_normal"):
            parts = line.split()
            # cpu_normal  max_abs  mean_abs  %>0.01  %>0.1
            return {
                "max_abs": float(parts[1]),
                "mean_abs": float(parts[2]),
                "pct_gt_001": float(parts[3].rstrip("%")),
                "pct_gt_01": float(parts[4].rstrip("%")),
            }
    print(f"WARNING: cpu_normal line not found in driver output:", file=sys.stderr)
    print(out, file=sys.stderr)
    return {}


def run_cli_boxcount(model_dir: Path, image: Path, lib: Path,
                     tmp_build: Path) -> int:
    """Rebuild the main ppocr_cli linking the given libMNN.a,
    run on the given image, return the line count."""
    if (tmp_build / "ppocr_cli").exists():
        shutil.rmtree(tmp_build, ignore_errors=True)
    tmp_build.mkdir(parents=True, exist_ok=True)
    # Use cmake to configure + build ppocr_cli with the variant lib.
    # We pass -DPPOCR_MNN_ROOT to a synthetic root so find_library
    # picks up the variant's libMNN.a, and we need to be careful:
    # the main CMake uses find_library which scans PATHS, so we
    # also need to give it PPOCR_MNN_ROOT that points to a directory
    # whose 'build/libMNN.a' exists.  We use a symlink farm.
    syn_root = tmp_build / "mnn_root"
    if syn_root.exists():
        shutil.rmtree(syn_root)
    syn_root.mkdir()
    # symlink: <syn_root>/include -> real MNN/include
    syn_root.joinpath("include").symlink_to(MNN_ROOT / "include")
    # symlink: <syn_root>/source -> real MNN/source
    syn_root.joinpath("source").symlink_to(MNN_ROOT / "source")
    # symlink: <syn_root>/build/libMNN.a -> real variant lib
    syn_build = syn_root / "build"
    syn_build.mkdir()
    syn_build.joinpath("libMNN.a").symlink_to(lib)

    cmd = [
        "cmake", str(ROOT),
        f"-DPPOCR_MNN_ROOT={syn_root}",
        "-DPPOCR_BUILD_TESTS=OFF",
        "-DPPOCR_BUILD_TOOLS=OFF",
        "-DPPOCR_BUILD_CLS=OFF",
    ]
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_build)
    if r.returncode != 0:
        print("cmake stdout:", r.stdout, file=sys.stderr)
        print("cmake stderr:", r.stderr, file=sys.stderr)
        raise RuntimeError(f"cmake failed for variant {lib.name}")

    cmd = ["make", "-j", "12", "ppocr_cli"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_build)
    if r.returncode != 0:
        print("make stdout:", r.stdout, file=sys.stderr)
        print("make stderr:", r.stderr, file=sys.stderr)
        raise RuntimeError(f"make failed for variant {lib.name}")

    cli = tmp_build / "ppocr_cli"
    if not cli.exists():
        raise RuntimeError(f"ppocr_cli not built at {cli}")
    # Run on the image, count lines.
    cmd = [
        str(cli),
        "--image", str(image),
        "--det-config", str(ROOT / "configs" / "PP-OCRv6_medium_det.json"),
        "--rec-config", str(ROOT / "configs" / "PP-OCRv6_medium_rec.json"),
        "--backend", "cpu",
        "--model-dir", str(model_dir),
        "--json", "/tmp/m3_var_out.json",
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    if r.returncode != 0:
        print("cli stderr:", r.stderr, file=sys.stderr)
        return -1
    out_json = Path("/tmp/m3_var_out.json")
    if not out_json.exists():
        return -1
    d = json.load(open(out_json))
    return len(d.get("lines", []))


# Variants to test. Each is (name, flag_list, description).
VARIANTS = [
    ("stock",     [],                                       "baseline (MNN_USE_SSE=ON, AVX512=OFF, BF16=OFF, OpenMP=OFF)"),
    ("no-sse",    ["-DMNN_USE_SSE=OFF"],                    "(a) naive CPU kernel, no SSE/AVX2"),
    ("avx512",    ["-DMNN_AVX512=ON"],                      "(b) AVX512 codegen (no-op on EPYC 7502)"),
    ("bf16",      ["-DMNN_SUPPORT_BF16=ON"],                "(c) BF16 accumulator in conv/fc"),
    ("openmp",    ["-DMNN_OPENMP=ON"],                      "(d) OpenMP threadpool instead of MNN's own"),
    ("no-sparse", ["-DMNN_USE_SPARSE_COMPUTE=OFF"],         "(e) disable sparse-compute codegen"),
    ("low-mem",   ["-DMNN_LOW_MEMORY=ON"],                  "(f) low-memory weight path"),
    ("sse-fp16",  ["-DMNN_SUPPORT_BF16=ON", "-DMNN_SSE_USE_FP16_INSTEAD=ON"],
                                                              "(g) FP16 instead of BF16 in x86 op"),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/m3_sweep.json", help="results json path")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--skip-cli", action="store_true",
                   help="skip ppocr_cli rebuild (saves time; per-variant box count omitted)")
    p.add_argument("--variants", nargs="*", default=None,
                   help="only run these variant names")
    args = p.parse_args()

    model_dir = ROOT / "models"
    image = Path("/root/ocr_test_imgs/ja/00.jpg")

    results = []
    selected = set(args.variants) if args.variants else None
    for name, flags, desc in VARIANTS:
        if selected and name not in selected:
            continue
        print()
        print("=" * 72)
        print(f"== variant '{name}': {desc}")
        print("=" * 72)
        t0 = time.time()
        lib = build_variant(name, flags, jobs=args.jobs)
        t_build = time.time() - t0

        # Build + run the diff driver
        driver_bin = Path(f"/tmp/m3_driver_{name}")
        build_driver(lib, driver_bin)
        out_prefix = Path(f"/tmp/m3_diff_{name}")
        diff = run_driver(driver_bin, out_prefix)
        print(f"   diff: {diff}", flush=True)

        # Build + run the CLI on ja/00 for box count
        n_boxes = None
        if not args.skip_cli:
            tmp_build = Path(f"/tmp/m3_build_{name}")
            try:
                n_boxes = run_cli_boxcount(model_dir, image, lib, tmp_build)
            except Exception as e:
                print(f"   CLI build/run error: {e}", flush=True)
                n_boxes = -1
            print(f"   ja/00 box count: {n_boxes}", flush=True)

        results.append({
            "variant": name,
            "description": desc,
            "flags": flags,
            "lib_path": str(lib),
            "lib_size_kb": lib.stat().st_size // 1024,
            "build_time_s": t_build,
            "diff": diff,
            "ja00_box_count": n_boxes,
        })

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2))
    print()
    print("=" * 72)
    print("== M3 kernel sweep summary")
    print("=" * 72)
    print(f"{'variant':12s}  {'max':>8s}  {'mean':>8s}  {'%>0.1':>7s}  {'ja/00':>6s}  build_s")
    print("-" * 72)
    baseline_mean = None
    for r in results:
        if r["variant"] == "stock":
            baseline_mean = r["diff"].get("mean_abs")
    for r in results:
        d = r["diff"]
        if not d:
            print(f"{r['variant']:12s}  {'-':>8s}  {'-':>8s}  {'-':>7s}  {'-':>6s}  {r['build_time_s']:.0f}")
            continue
        delta = ""
        if baseline_mean and r["variant"] != "stock" and d.get("mean_abs"):
            ratio = baseline_mean / d["mean_abs"] if d["mean_abs"] else 0
            delta = f" ({ratio:.1f}x)"
        print(f"{r['variant']:12s}  {d['max_abs']:>8.4f}  {d['mean_abs']:>8.6f}  {d['pct_gt_01']:>6.2f}%  "
              f"{r.get('ja00_box_count', -1):>6d}{delta}  {r['build_time_s']:.0f}")
    print()
    print(f"results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
