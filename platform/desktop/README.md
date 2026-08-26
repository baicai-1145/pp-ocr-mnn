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
