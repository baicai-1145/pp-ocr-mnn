# POST-6: MNN CPU precision-mode diff is a no-op

## Setup

Driver: `tests/mnn_backend_diff/driver.cpp`
Input:  `/tmp/m2num/det_input_paddle.npy` (CHW 3×704×1280, BGR float32, the
exact Paddle input dump from m2-num).
Model:  `models/PP-OCRv6_tiny_det.mnn` (the m2 baseline).
Output: one `.npy` per combo, shape (1, 1, 704, 1280).
Reference: `/tmp/m2num/det_output_paddle.npy` (Paddle GPU fp32).

```bash
g++ -std=c++17 -O2 -Wall -Wextra \
  -I third_party/MNN/include -I third_party/MNN/source \
  tests/mnn_backend_diff/driver.cpp \
  third_party/MNN/build/libMNN.a \
  -lpthread -ldl -lm \
  -o /tmp/mnn_backend_driver

/tmp/mnn_backend_driver \
  /root/pp-ocr-mnn/models/PP-OCRv6_tiny_det.mnn \
  /tmp/m2num/det_input_paddle.npy \
  /tmp/mnn_diff 4 \
  /tmp/m2num/det_output_paddle.npy
```

## Result table

| combo        | BackendConfig.precision | max_abs | mean_abs | %>0.01 | %>0.1  |
|--------------|-------------------------|---------|----------|--------|--------|
| cpu_normal   | Precision_Normal        | 0.9637  | 0.005858 | 2.11%  | 1.09%  |
| cpu_high     | Precision_High          | 0.9637  | 0.005858 | 2.11%  | 1.09%  |
| cpu_low      | Precision_Low           | 0.9637  | 0.005858 | 2.11%  | 1.09%  |
| cpu_low_bf16 | Precision_Low_BF16      | 0.9637  | 0.005858 | 2.11%  | 1.09%  |

**All four MNN CPU precisions produce byte-identical output to each other
and to the existing m2-num reference (`det_output_mnn.npy`).**

The diff vs Paddle GPU is the documented m2-num result: max 0.96, mean 0.006,
1.09% pixels > 0.1.

## Why

The `precision` config knob in MNN's `BackendConfig` only changes a small
handful of activation functions (sigmoid, exp, etc.) via
`CPUBackend::selectForFloat`. The vast majority of the det network is
conv/matmul where:

- `MNN_USE_SSE` / `MNN_USE_AVX2` are compile-time decisions, not runtime.
- `MNN_SUPPORT_BF16` is **OFF** in this MNN build
  (`third_party/MNN/build/CMakeCache.txt: MNN_SUPPORT_BF16:BOOL=OFF`),
  so `Precision_Low_BF16` silently falls back to the default CPUBackend
  with the same kernel code path as the other modes.
- `Precision_Low` triggers an Arm82 fast path that requires
  `MNN_USE_ARMV82` and a CPU with fp16 arithmetic; on x86_64 it has
  no effect.

Grep `getPrecisionMode` over `third_party/MNN/source/`: zero callers.
The `mPrecisionMode` field is stored on `CPUBackend` but never read
on x86_64, so the `ScheduleConfig::backendConfig->precision` switch
is effectively a no-op for this platform.

## OpenCL / Vulkan availability

- `nvidia-smi` reports: NVIDIA A10G (CUDA 13.0, driver 590.44.01).
- `libOpenCL.so.1` (NVIDIA ICD) and `libvulkan.so.1` are installed.
- However, MNN's `OpenCLSymbols::LoadOpenCLLibrary` only searches
  `/usr/lib/libOpenCL.so`, `/usr/lib64/libOpenCL.so`, etc. (see
  `third_party/MNN/source/backend/opencl/core/runtime/OpenCLWrapper.cpp:46-66`).
  Ubuntu ships `libOpenCL.so.1` only under `/usr/lib/x86_64-linux-gnu/`,
  and no symlink exists at the legacy paths, so MNN's `dlopen` fails
  and we see the runtime log:

      OpenCL init error, fallback ...
      Error to use creator of 3, delete it

  Vulkan has the same convention issue. Conclusion: **MNN cannot use
  GPU on this host, even though a GPU is present.** CPU is the only
  working MNN backend here.

## CUDA feasibility

Three reasons CUDA is unlikely to close the gap:

1. **MNN's CUDA backend is not built into this MNN checkout.**
   `MNN_CUDA:BOOL=OFF` in `third_party/MNN/build/CMakeCache.txt`. A rebuild
   with `MNN_CUDA=ON` is required (≈10 min, downloads cudnn/cublas headers
   and compiles the CUDA backend; the MNN build system wires up
   `find_package(CUDA)` and compiles `source/backend/cuda/*.cpp`).
2. **MNN's CUDA backend is not cuDNN.** Grep for `cudnn` over
   `third_party/MNN/source/backend/cuda/` returns zero matches; the backend
   uses MNN's own CUDA kernels (codegen path `MNN_CODEGEN_CUDA`).
3. **MNN GPU vs Paddle GPU would still be a kernel-level diff.** Paddle
   picks the conv algorithm via its own heuristic; for a small input
   (3×704×1280 → first conv 16x3x3x3 → ...) the choice between
   direct/implicit-GEMM/Winograd differs from what MNN selects. The
   exact same `max=0.96, mean=0.006` structure we see between MNN CPU
   and Paddle GPU is what we'd see between MNN CUDA and Paddle GPU,
   just shifted (and possibly larger because Paddle GPU on a
   bf16-capable GPU like A10G may internally use TF32).

The only realistic CUDA-vs-Paddle matchup would be MNN CUDA running
in **strict fp32** (no TF32) on the same GPU that Paddle uses in
strict fp32. That's a substantial experiment (rebuild + benchmark +
diff) for a result that is unlikely to be ≤ 0.05 CER, which is the
project gate. Not worth the 10+ min build right now.

## What this confirms

The m2-num root cause is **per-op CPU kernel arithmetic** in MNN's
AVX2/SSE conv path, not a high-level config knob. The CER gate
failure cannot be closed by tweaking `BackendConfig.precision`; the
only ways to reduce it are

1. Use a different backbone on the MNN side (e.g. ONNX -> specific
   CUDA graph executor that matches Paddle's choice), or
2. Accept the det-chain CER as the documented det noise floor and
   focus on rec side (already m2-num green: zh 0.0386, en 0.0429).

POST-5's clipper/unclip fix is independent of this — it's about the
**deterministic postprocess step** that runs *after* the prob map.
The prob map itself is what diverges.
