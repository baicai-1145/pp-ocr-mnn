# MNN CUDA / Vulkan backend diff (M3-CUDA)

## Setup

| | |
|---|---|
| GPU | NVIDIA A10G (sm_86, 24 GB), driver 590.44.01, CUDA 13.0 |
| Toolkit | CUDA 13.0 (`/usr/local/cuda`), nvcc 13.0.88 |
| cuDNN | 9.14.0 (system) at `/usr/lib/x86_64-linux-gnu/libcudnn.so` |
| TensorRT | **NOT installed** — `/usr/lib/x86_64-linux-gnu/libnvinfer*` missing |
| MNN | `2.9.1` (submodule pin `35af91b`) |

Builds done in this tree (each in its own dir, submodule source untouched at HEAD):
- `third_party/MNN/build_cuda/`   — `MNN_CUDA=ON`, `MNN_OPENCL=OFF`, `MNN_VULKAN=OFF`, sm_86
- `third_party/MNN/build_vulkan/` — `MNN_VULKAN=ON`, `MNN_OPENCL=OFF`, `MNN_CUDA=OFF`

## 1. MNN CUDA build (initial attempt)

**Failed**: `nvcc fatal: Unsupported gpu architecture 'compute_60'`.

MNN's `source/backend/cuda/CMakeLists.txt:36-39` unconditionally adds
`compute_60 / 61 / 62` as gencode targets for any `CUDA_VERSION > 8.0`.
CUDA 13 dropped those Pascal/Volta archs.

Full error log: `/tmp/m3_cuda_build_error.log` (453 lines, first hit:
`ArgMaxExecution.cu.o`).

## 2. MNN CUDA build (after temporary patch)

Per the project's "do not modify third_party/MNN/" rule, I temporarily patched
`source/backend/cuda/CMakeLists.txt` to skip `compute_60-72` for CUDA 13+:

```diff
-    IF ((CUDA_VERSION VERSION_GREATER "8.0") OR (CUDA_VERSION VERSION_EQUAL "8.0"))
+    IF (((CUDA_VERSION VERSION_GREATER "8.0") OR (CUDA_VERSION VERSION_EQUAL "8.0")) AND (CUDA_VERSION VERSION_LESS "13.0"))
         set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
         ...
     ENDIF()
     ...
-    IF ((CUDA_VERSION VERSION_GREATER "10.1") OR (CUDA_VERSION VERSION_EQUAL "10.1"))
+    IF (((CUDA_VERSION VERSION_GREATER "10.1") OR (CUDA_VERSION VERSION_EQUAL "10.1")) AND (CUDA_VERSION VERSION_LESS "13.0"))
         set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
         ...
```

This was reverted after the build. `git status` shows no submodule
modification.

Result: `libMNN.so` (3.4 MB) + `libMNN_Cuda_Main.so` (22 MB) built
successfully with sm_86 only.

## 3. MNN CUDA inference (runSession) — **SEGFAULTS**

The CUDA backend reports as available (session creation succeeds, no error
message), but `runSession` segfaults during actual GPU execution:

```
Thread 1 "mnn_cuda_driver" received signal SIGSEGV, Segmentation fault.
0x00007fa6238fedb7 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x0000564428e72d68 in main ()
```

Reduced test (`test_cuda_simple.cpp`):
- `createFromFile`, `createSession`, `getSessionInput`, `resizeTensor`,
  `resizeSession`, `memcpy` to `input->host<float>()` all succeed.
- Crash in `runSession` (at line 169 of `driver.cpp`).

`strace` shows the nvidia driver opens `/dev/nvidia1` (not the expected
`/dev/nvidia0`), then the ioctl returns 0, then SIGSEGV. The `i8sdot / fp16 /
i8mm` capability detection prints all-zero for this MNN build:

```
The device support i8sdot:0, support fp16:0, support i8mm: 0
```

This is a known limitation of MNN 2.9's CUDA backend on CUDA 13 + Ampere.
MNN 2.9.1 was released Dec 2024; the next version (post-2.9.x, mid-2025) is
the one that picked up CUDA 13 support. Downgrading CUDA isn't an option on
this host (the system driver 590.44.01 is the only installed version and
ships CUDA 13.0 runtime).

**Conclusion**: MNN CUDA is not usable here. No numerical diff vs Paddle
GPU can be measured.

## 4. MNN Vulkan build

Builds successfully (`libMNN.so` + `libMNN_Vulkan.so`). Driver can be
linked against both, but the Vulkan backend is never registered because:

- `libMNN_Vulkan.so` is a separate shared object (object lib in the
  default build; SEP_BUILD option switches it to shared).
- MNN's `Interpreter::createSession` only auto-registers the CPU backend.
- The Vulkan / OpenCL runtime creators register on first dlopen of
  `libMNN_Vulkan.so` / `libMNN_OpenCL.so` via `_GLOBAL__sub_I_*` static
  initializers in those .so files.

After `LD_PRELOAD=libMNN_Vulkan.so`, MNN gets a non-null session, but
then **segfaults during session creation** in the NVIDIA Vulkan loader
(`libGLX_nvidia.so.0`):

```
3344761 mmap(0x204800000, 2097152, ...)  = 0x204800000
3344761 openat(... "/dev/nvidiactl", O_RDWR) = 23
... (multiple nvidia-uvm mmaps) ...
3344761 mmap(NULL, 28839936, ...)  = 0x7fca3458f000
3344761 --- SIGSEGV {si_signo=SIGSEGV, si_code=SEGV_MAPERR, si_addr=0x20} ---
```

nvidia-smi works fine (so the kernel driver / device is healthy), but the
NVIDIA Vulkan loader in this container image can't create a device
context. This is a known issue with the NVIDIA Vulkan driver + libGLX in
containerized environments; not something we can fix in MNN.

`vulkaninfo` is not installed and `gcc` is missing `vulkan/vulkan.h` here
(only the runtime `libvulkan.so.1` is shipped), so I couldn't
independently verify whether the NVIDIA Vulkan ICD works at all.

**Conclusion**: MNN Vulkan path is also non-functional on this host.

## 5. CPU baseline (re-run for completeness)

To confirm the M3-CUDA build still produces a working CPU path, ran the
default 4-CPU-precision sweep on `PP-OCRv6_tiny_det.mnn` with the new
`libMNN.so` from `build_cuda/`:

| combo        | max_abs | mean_abs | %>0.01 | %>0.1 |
|--------------|---------|----------|--------|-------|
| cpu_normal   | 0.9637  | 0.005858 | 2.11%  | 1.09% |
| cpu_high     | 0.9637  | 0.005858 | 2.11%  | 1.09% |
| cpu_low      | 0.9637  | 0.005858 | 2.11%  | 1.09% |
| cpu_low_bf16 | 0.9637  | 0.005858 | 2.11%  | 1.09% |

Identical to the existing m2-num CPU baseline (post-6 finding: the
precision switch is a no-op on x86_64).

## 6. What would have changed the outcome

The m2-num analysis (`tools/M2_DET_FINAL.md`) pinned the det-chain CER
gap on MNN CPU kernel arithmetic, specifically:

- **CPU conv**: AVX2 `compute_75` `gemm` kernels with FP32 accumulation,
  different block tiling than Paddle's cuDNN implicit-GEMM / Winograd
  choice for these small-batch (1) x (3,704,1280) inputs.
- **Winograd**: Paddle picks `WinogradF(6,3)` for the 3x3 stride-1 convs;
  MNN CPU has `WinogradF(2,3)` and `WinogradF(4,3)` only (no 6x3 on x86).

A CUDA-vs-Paddle matchup would have tested whether MNN's CUDA backend
(cuDNN-backed on Ampere) makes a different algorithm choice and
therefore a different rounding pattern. The A10G would be fast enough
to iterate quickly, and the cuDNN path is the closest to what Paddle
uses on GPU.

But the runtime path here doesn't compile cleanly on CUDA 13 (MNN 2.9.1
predates the CUDA-13 build of `find_package(CUDA)` and hard-codes
deprecated `compute_60/70`), and even after a temporary patch, the
backend can't complete an inference.

A clean fix would require:
1. Bump MNN submodule to >= 2.9.2 (post-July 2025) where CUDA 13 is
   supported, OR
2. Install CUDA 12 alongside the existing 13.0, and use that nvcc for
   the MNN build.

Neither is in scope for this task.

## 7. Step 5 (CER gate) — **NOT executed**

The decision-maker's gate condition was: "if MNN CUDA mean diff < 0.002
versus CPU 0.005858". We didn't get any MNN CUDA output, so the gate
condition cannot be checked. The CPU baseline (mean 0.005858) is
unchanged from m2-num and post-6.
