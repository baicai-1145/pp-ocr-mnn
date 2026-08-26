#!/usr/bin/env bash
# Build pp-ocr-mnn for Android arm64-v8a.
#
# Two phases:
#   1. Build MNN as a static library (libMNN.a) for arm64-v8a
#      using NDK's bundled android.toolchain.cmake. Output to
#      third_party/MNN/build_android_arm64/.
#   2. Build the JNI .so (libppocr-jni.so) that links MNN +
#      ppocr_core + the JNI wrapper. Output to platform/android/lib/.
#
# Usage:
#   ANDROID_NDK=/opt/android-ndk ./platform/android/build.sh
#   # or with explicit NDK path:
#   ANDROID_NDK=/path/to/ndk ./platform/android/build.sh
#
# Outputs:
#   platform/android/lib/libppocr-jni.so
#   third_party/MNN/build_android_arm64/libMNN.a
#   third_party/MNN/build_android_arm64/CMakeCache.txt
#
# The build is non-destructive to the host build (build-main/)
# and to the per-variant CPU kernel build dirs (build_var/).
# All Android build artifacts are untracked by virtue of MNN's
# own .gitignore having "build*/" and our top-level .gitignore
# having "build-*/" — and "lib/" is the only artifact we copy
# out, into a directory not covered by any ignore rule, so we
# have to gitignore it explicitly in CI.

set -euo pipefail

# ---- locate repo root and NDK --------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${ANDROID_NDK:-}" ]]; then
  for cand in /opt/android-ndk "$HOME/Library/Android/sdk/ndk/27.0.12077973" \
              "$HOME/Android/Sdk/ndk/27.0.12077973"; do
    if [[ -d "$cand" ]]; then ANDROID_NDK="$cand"; break; fi
  done
fi
if [[ -z "${ANDROID_NDK:-}" || ! -d "$ANDROID_NDK" ]]; then
  echo "ANDROID_NDK not set and no default found. Set ANDROID_NDK=/path/to/ndk" >&2
  echo "Recommended: NDK r27 (clang 18). Earlier NDKs may not have C++20 headers." >&2
  exit 1
fi
echo "ANDROID_NDK = $ANDROID_NDK"

# ---- defaults (override via env) -----------------------------------------
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-21}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
MNN_BUILD_DIR="$ROOT/third_party/MNN/build_android_${ANDROID_ABI//-/_}"
JNI_BUILD_DIR="$ROOT/platform/android/build-android-${ANDROID_ABI//-/_}"
JNI_OUT_DIR="$ROOT/platform/android/lib"
TOOLCHAIN="$SCRIPT_DIR/android-arm64-v8a.toolchain.cmake"

# The existing host CMakeCache is the MNN submodule's. The Android
# cross-build needs a separate build dir so the host CMAKE_CXX_COMPILER
# (gcc) and the Android toolchain (clang) don't conflict.
echo
echo "== Phase 1: build MNN static lib for $ANDROID_ABI (NDK r27)"
echo "   build dir: $MNN_BUILD_DIR"
if [[ ! -f "$MNN_BUILD_DIR/libMNN.a" ]]; then
  mkdir -p "$MNN_BUILD_DIR"
  cmake \
    -S "$ROOT/third_party/MNN" \
    -B "$MNN_BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_POLICY_DEFAULT_CMP0057=NEW \
    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW \
    -DANDROID_NDK="$ANDROID_NDK" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_SHARED_LIBS=OFF \
    -DMNN_BUILD_CONVERTER=OFF \
    -DMNN_BUILD_TOOLS=OFF \
    -DMNN_BUILD_DEMO=OFF \
    -DMNN_BUILD_BENCHMARK=OFF \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_CODEGEN=OFF \
    -DMNN_BUILD_LLM=OFF \
    -DMNN_BUILD_DIFFUSION=OFF \
    -DMNN_BUILD_TRAIN=OFF \
    -DMNN_BUILD_QUANTOOLS=OFF \
    -DMNN_BUILD_MINI=OFF \
    -DMNN_BUILD_PROTOBUFFER=OFF \
    -DMNN_OPENCL=OFF \
    -DMNN_OPENGL=OFF \
    -DMNN_VULKAN=OFF \
    -DMNN_CUDA=OFF \
    -DMNN_TENSORRT=OFF \
    -DMNN_COREML=OFF \
    -DMNN_NNAPI=OFF \
    -DMNN_ARM82=OFF \
    -DMNN_METAL=OFF \
    -DMNN_ONEDNN=OFF \
    -DMNN_OPENMP=OFF \
    -DMNN_USE_THREAD_POOL=ON \
    -DMNN_USE_SPARSE_COMPUTE=ON
  cmake --build "$MNN_BUILD_DIR" --target MNN -- -j "$JOBS"
else
  echo "   (libMNN.a already built, skipping)"
fi
ls -la "$MNN_BUILD_DIR/libMNN.a"

# ---- phase 2: JNI .so -----------------------------------------------------
echo
echo "== Phase 2: build JNI .so linking libMNN.a + ppocr_core"
mkdir -p "$JNI_BUILD_DIR"
cmake \
  -S "$ROOT" \
  -B "$JNI_BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_NDK="$ANDROID_NDK" \
  -DANDROID_ABI="$ANDROID_ABI" \
  -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPPOCR_BUILD_TESTS=OFF \
  -DPPOCR_BUILD_TOOLS=OFF \
  -DPPOCR_BUILD_CLS=OFF \
  -DPPOCR_BUILD_JNI=ON \
  -DPPOCR_MNN_ROOT="$MNN_BUILD_DIR/.."

cmake --build "$JNI_BUILD_DIR" --target ppocr-jni -- -j "$JOBS"

# ---- install: lib/ + JNI symbols -----------------------------------------
mkdir -p "$JNI_OUT_DIR"
cp -f "$JNI_BUILD_DIR/libppocr-jni.so" "$JNI_OUT_DIR/libppocr-jni.so"
echo
echo "== Done"
echo "  $JNI_OUT_DIR/libppocr-jni.so"
file "$JNI_OUT_DIR/libppocr-jni.so" 2>/dev/null || ls -la "$JNI_OUT_DIR/libppocr-jni.so"
