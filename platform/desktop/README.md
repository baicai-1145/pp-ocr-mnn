# platform/desktop — Linux / macOS / Windows desktop packaging

## Files

This directory exists as a placeholder for future desktop
distribution work (debian package, brew formula, MSI, etc.).
The M5-PREP commit ships only the **install rules** and a
**minimal C example** to verify the install layout:

  * Top-level `CMakeLists.txt` — adds `install(TARGETS
    ppocr_cli ...)` and `install(DIRECTORY include/ppocr ...)`
    under `GNUInstallDirs` defaults.
  * `examples/c_api_demo.c` — pure C smoke test that calls
    `ppocr_create → ppocr_run_file → print → ppocr_destroy`.
  * `examples/CMakeLists.txt` — standalone build system
    for the example.

## Install (Linux)

```bash
cmake -B build-main -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build-main -j
DESTDIR=$PWD/install-prefix cmake --install build-main
```

The install layout is:

```
install-prefix/usr/local/
  bin/ppocr_cli
  lib/libppocr_core.a
  include/ppocr/ppocr.h
  include/ppocr/downloader.h
  include/ppocr/config.h
  include/ppocr/mnn_session.h
  include/ppocr/preprocess.h
  include/ppocr/image.h
  include/ppocr/postprocess/db_post.h
  include/ppocr/postprocess/geometry.h
  include/ppocr/postprocess/ctc_decode.h
```

`libMNN.a` is **deliberately not installed** — it is large
(10 MB), ties the install to a specific MNN build config
(which AVX/SIMD paths are compiled in, which backends are
available), and most downstream consumers will already have
their own. The C ABI does not require the consumer to use
the same MNN we built against.

## C example (verifies ABI has no C++ dependency)

```bash
cmake -B examples/build -S examples \
      -DPPOCR_EXAMPLES_MNN_ROOT=third_party/MNN
cmake --build examples/build

# Smoke test:
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
  ./examples/build/c_api_demo /path/to/image.jpg ./models
```

Verified end-to-end on this host:

  ```
  $ ./examples/build/c_api_demo /root/ocr_test_imgs/zh/04.jpg ./models
  ppocr-mnn v0.1.0
  engine created (handle=0x56533f56df10)
  backend: cpu, n_lines: 2, det_ms: 286.1, rec_ms: 29.7, ...
  ...
  engine destroyed
  ```

The example is intentionally tiny: it just exercises the
public C ABI to confirm the install layout is consumable
from a non-C++ language. It does not measure CER; that is
what `tools/run_reference.py` is for.

## macOS

The host build works on macOS out of the box; the only
caveat is that `find_package(CURL)` on Apple Silicon
sometimes needs a hint:

```bash
cmake -B build -S . \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/homebrew
```

The install rules use `GNUInstallDirs`, which on macOS
defaults to `/usr/local/` (matching the Apple conventions
for non-Homebrew installs). Homebrew users would override
with `-DCMAKE_INSTALL_PREFIX=$(brew --prefix)`.

## Windows

The `examples/CMakeLists.txt` does not yet have a Windows
target — cross-compile via x86_64-w64-mingw32 (verified
on this host as part of M5-PREP, see commit message):

```bash
# Build MNN for Windows:
CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ \
  cmake -B build-win-mnn -S third_party/MNN \
        -DCMAKE_SYSTEM_NAME=Windows \
        -DCMAKE_RC_COMPILER=x86_64-w64-mingw32-windres \
        -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
        -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ \
        -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_CONVERTER=OFF \
        -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_DEMO=OFF \
        -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_TEST=OFF \
        -DMNN_BUILD_CODEGEN=OFF -DMNN_BUILD_LLM=OFF \
        -DMNN_BUILD_PROTOBUFFER=OFF \
        -DMNN_OPENCL=OFF -DMNN_OPENGL=OFF -DMNN_VULKAN=OFF \
        -DMNN_CUDA=OFF -DMNN_TENSORRT=OFF -DMNN_COREML=OFF \
        -DMNN_NNAPI=OFF -DMNN_ARM82=OFF -DMNN_METAL=OFF \
        -DMNN_ONEDNN=OFF \
        -DCMAKE_BUILD_TYPE=Release
cmake --build build-win-mnn --target MNN -j

# Stage the lib where find_library looks for it:
mkdir -p third_party/MNN/build_windows
ln -sf $(pwd)/build-win-mnn/libMNN.a third_party/MNN/build_windows/libMNN.a

# Build libppocr_core for Windows:
CC=x86_64-w64-mingw32-gcc CXX=x86_64-w64-mingw32-g++ \
  cmake -B build-windows -S . \
        -DCMAKE_SYSTEM_NAME=Windows \
        -DCMAKE_RC_COMPILER=x86_64-w64-mingw32-windres \
        -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
        -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ \
        -DPPOCR_BUILD_TESTS=OFF -DPPOCR_BUILD_TOOLS=OFF \
        -DPPOCR_BUILD_CLS=OFF \
        -DCMAKE_FIND_ROOT_PATH=/usr/x86_64-w64-mingw32 \
        -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
        -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
        -DCMAKE_BUILD_TYPE=Release
cmake --build build-windows --target ppocr_core -j

# Build the C example:
x86_64-w64-mingw32-gcc -std=c99 -Wall \
  -I include \
  examples/c_api_demo.c \
  build-windows/libppocr_core.a \
  third_party/MNN/build_windows/libMNN.a \
  -L/usr/x86_64-w64-mingw32/lib \
  -lstdc++ -lpthread -lm -lz -lws2_32 -lbcrypt \
  -o c_api_demo.exe
```

The resulting `c_api_demo.exe` is a 5.4 MB PE32+ binary
that the cross-linker reports as `Mach-O 64-bit ...`
(wait, no — it reports `ELF 64-bit ...` for the .o files
and `PE32+` for the linked .exe).

  ```
  $ head -c 2 c_api_demo.exe | xxd
  00000000: 4d5a                                   MZ
  ```

Verified: the .exe starts with the `MZ` magic; it is a
valid Windows PE file. No wine runtime on this host to
actually execute it, but the cross-compile completes
cleanly and the symbol resolution succeeds (no undefined
references).

## macOS (Apple Silicon / arm64, native host build)

Status: **validated** — CPU backend (MNN ARM82 + KleidiAI, NEON fp32), on
MacBook Air M4 / macOS 26.6 / Xcode 26.2 / CMake 4.3, against the published
eval dataset (`hf.co/datasets/baicai1145/pp-ocr-mnn-eval`).

There is no prebuilt `libMNN.a` in `third_party/MNN/build` on a fresh macOS
clone, so the top-level CMake takes its documented fallback path:
`add_subdirectory(third_party/MNN)` with a CPU-only MNN config
(ARM82/KleidiAI ON, Metal/OpenCL/Vulkan OFF — Metal/CoreML validation is M5
scope). Build:

```sh
cmake -S . -B build-mac -DCMAKE_BUILD_TYPE=Release
cmake --build build-mac -j8
ctest --test-dir build-mac   # with PPORC_MNN_MODELS/PPOCR_IMG_ROOT set
```

Two things differ from the prebuilt-MNN Linux path (both handled in the
top-level CMakeLists):

1. The fallback branch aliases `MNN_LIBRARY=MNN` (the CMake target) so the
   downloader test targets resolve without a `find_library` hit.
2. `-Wl,--whole-archive` is GNU-ld only; Apple ld64 uses
   `-Wl,-force_load,<archive>` (applied to the two downloader test targets,
   which then also need the public include dirs + libjpeg linked explicitly
   because a raw link item does not propagate `ppocr_core`'s interface).
   `find_package(CURL)` on Homebrew can find the lib without creating the
   `CURL::CURL` target, so CMake synthesizes it when missing.

NFS-hosted worktrees: macOS creates AppleDouble `._*` metadata files next to
every source file, which MNN's source GLOBs pick up and clang rejects
(`no such file: .../._Backend.cpp`). Clean them before configuring and after
any FetchContent extraction:

```sh
find third_party/MNN build-mac/_mnn_build/_deps -name '._*' -delete
```

then re-run `cmake -S . -B build-mac` (GLOB results are cached in the
generated Makefiles) and build.

### e2e acceptance run (published dataset, no /root paths)

```sh
export PPOCR_MNN_MODELS=./models
export PPOCR_IMG_ROOT=_downloads/eval/root/ocr_test_imgs
export PPOCR_REF_ROOT=_downloads/eval/root/ppocr_reference
python3 tools/run_reference.py --cli ./build-mac/ppocr_cli --backend cpu \
  --threads 4 --jobs 4 --results-dir results/mac-cpu \
  --only-combo PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec \
  --only-combo PP-OCRv4_mobile_det__PP-OCRv4_mobile_rec \
  --only-combo PP-OCRv5_mobile_det__en_PP-OCRv5_mobile_rec \
  --only-combo PP-OCRv5_mobile_det__th_PP-OCRv5_mobile_rec \
  --only-combo PP-OCRv5_mobile_det__korean_PP-OCRv5_mobile_rec
python3 tools/score.py --results-dir results/mac-cpu --report report.md
```

Verified results (800 images, MLC gate ≤ 0.05, all PASS):

| cell | MLC (join) |
|---|---|
| PP-OCRv4_mobile_det__PP-OCRv4_mobile_rec | 0.0388 (0.0322) |
| PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec | 0.0187 (0.0182) |
| PP-OCRv5_mobile_det__en_PP-OCRv5_mobile_rec | 0.0000 (0.0001) |
| PP-OCRv5_mobile_det__th_PP-OCRv5_mobile_rec | 0.0030 (0.0031) |
| PP-OCRv5_mobile_det__korean_PP-OCRv5_mobile_rec | 0.0119 (0.0254) |

Single-image smoke (zh/04.jpg, PP-OCRv6_tiny, 4 threads): det 110 ms /
rec 13 ms / total 122 ms — `examples/c_api_demo.c` links and runs with
`clang ... build-mac/libppocr_core.a build-mac/_mnn_build/libMNN.a -lz
-ljpeg -lcurl -lc++`.

## Metal backend (Apple GPU) — validated: 5/5 cells PASS

`third_party/MNN` is rebuilt locally with Metal for this backend:

```sh
cmake -S third_party/MNN -B third_party/MNN/build -DCMAKE_BUILD_TYPE=Release \
  -DMNN_METAL=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_DEMO=OFF -DMNN_BUILD_BENCHMARK=OFF \
  -DMNN_BUILD_TEST=OFF -DMNN_OPENCL=OFF -DMNN_VULKAN=OFF -DMNN_CUDA=OFF
cmake --build third_party/MNN/build -j8   # -> libMNN.a, picked up as "prebuilt"
cmake -S . -B build-mac && cmake --build build-mac -j8
```

With a Metal-enabled libMNN.a present, the top-level CMake switches to its
prebuilt path; linking needs `-framework Metal -framework Foundation`
(added in the top-level CMakeLists) and `-Wl,-force_load` test targets need
CXX linker language + the same frameworks explicitly (raw link items do not
propagate interface deps).

`--backend metal` and `--backend auto` both work end-to-end (AUTO resolves
to Metal on this build; output is bit-identical to `--backend metal`).

### Numerics (MNN 3.6.1, MacBook Air M4, published eval dataset)

Two MNN Metal issues were diagnosed and are mitigated in the engine
(`src/mnn_session.cpp`), without touching the submodule:

1. **fp16 divergence** — MNN's Metal fp16 path (its default) diverges badly
   from CPU: with fp16 the 5-cell subset scores MLC 0.05–0.62 (gate ≤ 0.05),
   0/5 PASS even with winograd off. fp16 on Metal is therefore not usable
   for this workload.
2. **Winograd first-rows corruption (upstream MNN bug)** — MNN's Metal
   Winograd transform emits elevated prob values in the first output rows
   of the det prob map for large det inputs (ru/02, 1280×853 → ~864-row
   input: baseline 1 box → 21 boxes, all hugging y ≤ 22 with 4–9 px
   heights; the true box matches CPU exactly). With
   `Interpreter::setSessionHint(WINOGRAD_MEMORY_LEVEL, 0)` (Metal only; the
   CPU winograd path is part of the validated matrix numerics) the artifact
   disappears and det becomes box-exact vs CPU.
   `PPOCR_MNN_WINOGRAD=1` re-enables it for upstream comparison.

Final config = **fp32 (Precision_High) + winograd off**. The engine forces
`Precision_High` on all GPU backends; Metal joins CUDA/OpenCL/Vulkan there.

| cell (800 imgs) | Metal fp32+w0 MLC | CPU MLC | status |
|---|---|---|---|
| PP-OCRv4_mobile_det__PP-OCRv4_mobile_rec | 0.0388 | 0.0388 | PASS |
| PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec | 0.0227 | 0.0187 | PASS |
| PP-OCRv5_mobile_det__en_PP-OCRv5_mobile_rec | 0.0000 | 0.0000 | PASS |
| PP-OCRv5_mobile_det__th_PP-OCRv5_mobile_rec | 0.0030 | 0.0030 | PASS |
| PP-OCRv5_mobile_det__korean_PP-OCRv5_mobile_rec | 0.0119 | 0.0119 | PASS |

`--backend metal` and `--backend auto` both work end-to-end.

### Backend selection: AUTO resolves to CPU

`pickBackend()` now maps `PPOCR_BACKEND_AUTO` to CPU unconditionally: CPU
is the only backend validated on every host, and MNN's own
`MNN_FORWARD_AUTO` picked Metal on this build — slower than CPU for this
workload (see perf table) and previously numerically off-contract. GPU
remains explicit opt-in (`--backend metal` etc.).

### Performance (MacBook Air M4)

| metric | CPU (4 thr) | Metal fp32+w0 | Metal fp16 (default) |
|---|---|---|---|
| zh/04.jpg e2e (steady, batch-dir w1) | ~193 ms | ~810 ms | ~425 ms |
| batch-dir zh w1 throughput | 4.87 FPS | 1.23 FPS | 2.34 FPS |
| 800-image single-image-mode matrix wall | 172 s | 3363 s | 1983 s |

Metal fp32 is slower than CPU for this workload on M4 (Apple GPU fp32
throughput is a fraction of fp16; winograd off costs conv throughput; shader
pipelines also recompile per process in single-image mode). AUTO→CPU is
therefore the right default on macOS; Metal is available explicitly and
numerically safe. Filing the winograd first-rows bug upstream would make
fp16... still off-contract numerically — so CPU stays the macOS default.
