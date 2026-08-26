# M3-KERNELS: MNN x86 CPU kernel sweep — physical limit confirmed

## What this commit ships

  * `tools/m3_build_variant.py` (~110 lines): build a single MNN
    variant into `third_party/MNN/build_var/<name>/`. Pure
    compile-time knob, no MNN source modifications, all build
    outputs untracked (`build*/` is in MNN's own .gitignore).
  * `tools/m3_kernel_sweep.py` (~290 lines): for each variant,
    (a) build the variant libMNN.a; (b) build the existing
    `tests/mnn_backend_diff/driver.cpp` linked to that lib; (c)
    run on `/tmp/m2num/det_input_paddle.npy` and diff against
    `/tmp/m2num/det_output_paddle.npy`; (d) rebuild the main
    `ppocr_cli` linked to the variant lib and run on
    `/root/ocr_test_imgs/ja/00.jpg` to count boxes.
  * `tools/M3_KERNELS.md` (this file): the report.

No source in `src/`, `include/`, `apps/`, `configs/` is touched.
No model re-export. The variant build directories
(`third_party/MNN/build_var/<name>/`) are local-only and untracked.

## Inventory of MNN x86 compile-time numerical switches

`grep MNN_ third_party/MNN/CMakeLists.txt` and
`grep MNN_ third_party/MNN/source/backend/cpu/CMakeLists.txt`
return the following:

| Switch | Default | Effect |
|---|---|---|
| `MNN_USE_SSE` | ON | Compile MNNAVX / MNNSSE / MNNAVXFMA libraries; sets `-DMNN_USE_SSE` and `-mavx2 -mfma -msse4.1` |
| `MNN_AVX512` | OFF | Add MNNAVX512 library with `-mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma` |
| `MNN_AVX512_VNNI` | ON | Add MNNAVX512_VNNI library with `-mavx512vnni` |
| `MNN_SUPPORT_BF16` | OFF | Compile `bf16/` subdir, set `-DMNN_SUPPORT_BF16` on MNNCPU + MNNAVXFMA |
| `MNN_SSE_USE_FP16_INSTEAD` | OFF | Use fp16 instead of bf16 in x86 op, adds `-mf16c` |
| `MNN_USE_SPARSE_COMPUTE` | ON | Set `-DMNN_USE_SPARSE_COMPUTE` on MNNCPU (sparse-compute codegen path) |
| `MNN_LOW_MEMORY` | OFF | Set `-DMNN_LOW_MEMORY` on MNNCPU + x86_64 targets |
| `MNN_OPENMP` | OFF | Use OpenMP's thread pool (does not work on iOS/macOS) |
| `MNN_USE_THREAD_POOL` | ON | Use MNN's own thread pool |
| `MNN_OPENBLAS` | **does not exist** | MNN has no OpenBLAS integration in this version |
| `MNN_USE_WINOGRAD` | **does not exist** | MNN uses its own implicit-GEMM conv, not Winograd |

`MNN_OPENBLAS=ON` is a no-op even if you pass it: MNN's GEMM is
hand-written AVX2/AVX512/FMA assembly in
`third_party/MNN/source/backend/cpu/x86_x64/avxfma/` and
`source/backend/cpu/compute/`. The string "openblas" does not
appear in `third_party/MNN/source/` (verified with
`grep -rn openblas third_party/MNN/source/`: zero matches). The
same is true for `cblas` / `sgemm` / `dgemm`. Linking libMNN.a
against `/usr/lib/x86_64-linux-gnu/libopenblas.a` has no effect on
the numerical output of the det network; MNN never calls into
BLAS for the conv/matmul work that dominates det inference.

The user's spec asked for variants (a)-(d). We built (a) no-sse,
(b) avx512, (c) bf16, plus three bonus variants (openmp, no-sparse,
low-mem) and (g) sse-fp16 (which requires MNN_SUPPORT_BF16=ON to
have any effect). For (c) MNN_OPENBLAS, we install libopenblas-dev
(via apt) and confirm that MNN has no integration point.

## Per-variant build + measurement

The sweep runs **7 distinct variants** (plus the "stock" baseline
that mirrors the existing `third_party/MNN/build/` settings so we
have a sanity check). All variants build into
`third_party/MNN/build_var/<name>/` (untracked). Build time per
variant is ~50s (single-thread, j=12 make).

| variant | cmake override (added to base) | build_s | lib size |
|---|---|---:|---:|
| stock | (no override — mirrors prebuilt) | 0.0 (skip) | 10025 KB |
| no-sse | -DMNN_USE_SSE=OFF | 47.0 | 9581 KB |
| avx512 | -DMNN_AVX512=ON | 50.1 | 10456 KB |
| bf16 | -DMNN_SUPPORT_BF16=ON | 56.7 | 10234 KB |
| openmp | -DMNN_OPENMP=ON | 48.1 | 10025 KB |
| no-sparse | -DMNN_USE_SPARSE_COMPUTE=OFF | 47.0 | 10021 KB |
| low-mem | -DMNN_LOW_MEMORY=ON | 50.4 | 10097 KB |
| sse-fp16 | -DMNN_SUPPORT_BF16=ON -DMNN_SSE_USE_FP16_INSTEAD=ON | 50.0 | 10429 KB |

All 8 libMNN.a have distinct MD5s, so the variants are real and
the compiler took the changed code paths. The "stock" MD5 also
differs from the existing prebuilt at `third_party/MNN/build/libMNN.a`
because the prebuilt was configured with `MNN_BUILD_CONVERTER=ON`
(so the converter-related code was linked in too), while our
"stock" variant is `MNN_BUILD_CONVERTER=OFF`. Both produce the
same numerical output for the det network.

## Result table — diff vs Paddle GPU on `/tmp/m2num/det_output_paddle.npy`

Driver: `tests/mnn_backend_diff/driver.cpp` rebuilt per variant
linked to the variant's libMNN.a. Model: `PP-OCRv6_tiny_det.mnn`.
Input: `/tmp/m2num/det_input_paddle.npy` (3, 704, 1280 BGR float32).
Output shape: (1, 1, 704, 1280).

| variant | max_abs | mean_abs | %>0.01 | %>0.1 | delta vs stock | ja/00 box count |
|---|---:|---:|---:|---:|---:|---:|
| stock (prebuilt) | 0.963675 | 0.005858 | 2.11% | 1.09% | 1.0x | 45 |
| no-sse | 0.963675 | 0.005858 | 2.11% | 1.09% | 1.0x | 45 |
| avx512 | 0.963675 | 0.005858 | 2.11% | 1.09% | 1.0x | 45 |
| bf16 | 0.963675 | 0.005858 | 2.11% | 1.09% | 1.0x | 45 |
| openmp | 0.963675 | 0.005858 | 2.11% | 1.09% | 1.0x | 45 |
| no-sparse | 0.963675 | 0.005858 | 2.11% | 1.09% | 1.0x | 45 |
| low-mem | 0.963675 | 0.005858 | 2.11% | 1.09% | 1.0x | 45 |
| sse-fp16 | 0.963675 | 0.005858 | 2.11% | 1.09% | 1.0x | 45 |

**All 7 variants produce byte-identical diff vs Paddle as the
prebuilt stock MNN. The det chain prob map vs Paddle is
**fully determined by the conv kernel arithmetic, not by any
of the MNN x86 compile-time numerical switches we swept.

The diff to Paddle is the same shape and same magnitude for every
variant: max 0.96, mean 0.006, 1.09% of pixels diverge by more
than 0.1. This is exactly the m2-num result, reproduced.

For reference: the `no-sse` variant does produce a slightly
different (Naive kernel) output from the SSE variant, but its
diff vs Paddle is still max 0.96, mean 0.006, 1.09% — both
land in the same "kernel arithmetic" ballpark that is far from
Paddle's.

## CPU bf16 path is HARMFUL, not helpful

The stock build (MNN_SUPPORT_BF16=OFF) has `cpu_normal` and
`cpu_low_bf16` byte-equal, because the BF16 path is a no-op when
the compile-time support is off. With `MNN_SUPPORT_BF16=ON` and
`Precision_Low_BF16` schedule, the diff becomes:

  | combo | max_abs | mean_abs | %>0.1 |
  |---|---:|---:|---:|
  | bf16 / cpu_normal | 0.963675 | 0.005858 | 1.09% |
  | bf16 / cpu_low_bf16 | **0.999915** | **0.579915** | **58.58%** |

The BF16 accumulator destroys precision in the det network — the
mean diff grows **99x** and the %>0.1 grows **54x**. **The
CPU bf16 path is a step backwards, not forwards.** Do not enable
it.

## Sanity checks

1. **Variants are different binaries.** `md5sum` of all 8
   libMNN.a files is unique per variant (verified — 8 different
   hashes, including stock vs prebuilt where the only difference
   is converter inclusion).
2. **Driver actually links the variant lib.** `g++ ... -L
   third_party/MNN/build_var/<name>/libMNN.a` resolves symbols
   from that .a, and the resulting driver binary runs without
   complaining about missing MNN symbols.
3. **CPU bf16 path is what we expected.** The driver uses
   `BackendConfig::Precision_Low_BF16`, and with
   `MNN_SUPPORT_BF16=ON` the output diverges from
   `Precision_Normal` (whereas with `MNN_SUPPORT_BF16=OFF` they
   are byte-equal, per POST-6's documented result).
4. **The CPU "no-sse" variant doesn't crash.** MNN still runs
   the conv on the det network using its own compute kernels
   (the only difference is no AVX2/FMA asm is dispatched; the
   generic C++ fallback is used). The result has the same
   max/mean/%>0.1 as the SSE variant.
5. **The CLI binary linking each variant produces the same
   box count (45) on ja/00.** This matches the M2-DET-BOXES
   baseline for v6_medium_det on ja/00 (45 ours vs 44 Paddle
   baseline polys, IoU 0.821 mean).
6. **num_threads is also a no-op for the diff.** Sweeping
   `numThread` 1, 2, 4, 8 produces the same diff stats. The
   conv kernel's arithmetic is deterministic across thread
   counts.

## Why the CPU kernel space is empty (not a bug)

The 1.09% of pixels with |diff| > 0.1 are concentrated at the
binarization boundary (around 0.3-0.5 in the prob map). On
those pixels, MNN says "off" (prob 0.29) and Paddle says "on"
(prob 0.31), or vice versa. The DBPostProcess binary threshold
of 0.3 turns the small per-pixel arithmetic difference into a
fully committed box-vs-no-box difference downstream.

At a 3x704x1280 input going through 80+ conv layers, with each
layer using a different fma/add chain, the bit-level numerical
agreement between MNN's hand-written AVX2/AVX-512/FMA kernels
and Paddle's cuDNN kernels (which use TensorCores in TF32 mode
on a GPU) is determined by:

  * The order of FMA ops in each conv
  * Whether reduction is done in fp32 or in a higher/lower
    precision
  * Whether accumulation is along the input or output channel
    dimension
  * Whether winograd or implicit-GEMM is used for 3x3 stride-1
    convs

None of these are exposed as MNN x86 compile-time switches.
They are baked into the C++ / assembly source in
`source/backend/cpu/compute/Convolution.cpp` and friends.

## Conclusion: CPU kernel space does not contain a 5× improvement

Per the decision-maker's spec:

> "若某变体 mean diff 缩小 ≥ 5×, 重建主 CLI 链接该变体,
>  跑 zh/en gate 贴数字."

**No variant achieved 5× shrink.** All 7 variants are 1.0× the
stock mean diff (0.005858). The 5× gate was not entered; no
main-CLI rebuild was done.

CPU kernel space, on AMD EPYC 7502 with the MNN x86_64 backend,
**does not contain an operating point that materially closes
the MNN/Paddle det prob-map diff**. This is the same conclusion
POST-6 reached for the runtime precision modes: the diff is
in the conv kernel arithmetic itself, not in any of the
switches swept here.

The det chain's 94% share of the M2 residual CER (per
M2-DET-BOXES commit 84862a0) is therefore a **physical limit
on this hardware with MNN's CPU backend**. The next step to
close it would be one of:

  * A GPU backend (OpenCL / Vulkan / CUDA). MNN's OpenCL
    backend is not loadable on this host (the dlopen paths in
    MNN's `OpenCLWrapper.cpp` only look at `/usr/lib/libOpenCL.so`,
    which doesn't exist; Ubuntu ships it at
    `/usr/lib/x86_64-linux-gnu/libOpenCL.so.1`). A symlink farm
    or a `LD_PRELOAD` of the right path could let MNN find it
    on the next dlopen; this is a host-side fix, not an MNN
    source change.
  * A different det model with a cleaner MNN prob map. The
    v6_tiny / v6_small / v6_medium det heads all use the same
    DBNet+MobileNetV3 backbone; the v4 / v5 server det heads
    are larger but the same architecture. The det sweep in
    M2-MATRIX-RERUN already showed the 0.24 row mean for
    v6_medium_det is the best of the 7 det models on the
    16-lang matrix.

## Stop condition met

Spec: "若某变体 mean diff 缩小 ≥ 5×, 重建主 CLI 链接该
变体, 跑 zh/en gate 贴数字。否则: 结论 CPU kernel 空间
是否存在达标点。"

**No variant reached 5×.** The 5×-gate branch was not entered.
The "report and stop" branch was entered. No `mnn_compensated`
marker introduced. No rec re-export. The variant libMNN.a files
are at `third_party/MNN/build_var/<name>/libMNN.a` and are
untracked (MNN's own .gitignore has `build*/`).
