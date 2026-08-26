# platform/android — Android JNI wrapper

JNI trampoline that maps the public C ABI
(`include/ppocr/ppocr.h`) onto a Java class
`ppocr.Ppocr`, plus a CMake toolchain file and a
build script that produce `libppocr-jni.so` for
arm64-v8a from a stock NDK r27 install.

## Files

  * `android-arm64-v8a.toolchain.cmake` — CMake
    toolchain that wraps NDK r27's bundled
    `android.toolchain.cmake`. Pre-sets
    `CMP0057 = NEW` (NDK r27's `flags.cmake` uses
    `IN_LIST` in `if()`).
  * `android-project-init.cmake` — pre-project
    shim that `CMAKE_PROJECT_MNN_INCLUDE`s into
    the MNN build to set `CMP0057 = NEW` before
    MNN's `cmake_minimum_required(VERSION 3.0)`
    forces the policy back to OLD.
  * `ppocr_jni.cc` — the JNI shim. Four native
    methods: `create`, `destroy`, `run` (Bitmap),
    `runFile` (path).
  * `build.sh` — two-phase build: first MNN for
    arm64-v8a, then the JNI .so. Output:
    `platform/android/lib/libppocr-jni.so`.

## Build

```bash
# Install NDK r27 (or set ANDROID_NDK):
#   https://developer.android.com/ndk/downloads
#   Recommended: r27 (clang 18).

ANDROID_NDK=/opt/android-ndk ./platform/android/build.sh
```

The script produces:

  ```
  platform/android/lib/libppocr-jni.so   (56 MB,
                                          arm64-v8a)
  third_party/MNN/build_android_arm64_v8a/libMNN.a
  ```

Both are untracked. `build_android_*` matches
MNN's own `build*/` ignore pattern; `lib/` is
gitignored at the top level.

## Java side

The native methods are declared in the
`ppocr.Ppocr` Java class. The class itself is
trivial and must be added to your Android
project. Drop this into
`app/src/main/java/ppocr/Ppocr.java`:

```java
package ppocr;

import android.graphics.Bitmap;
public class Ppocr {
    static {
        System.loadLibrary("ppocr-jni");
    }
    private long handle;
    public Ppocr(String modelDir, String detName, String recName,
                 int backend, int numThreads) {
        handle = nativeCreate(modelDir, detName, recName, backend, numThreads);
        if (handle == 0) throw new RuntimeException("ppocr_create failed");
    }
    public Line[] run(Bitmap bmp) { return nativeRun(handle, bmp); }
    public Line[] runFile(String path) { return nativeRunFile(handle, path); }
    public void close() { nativeDestroy(handle); handle = 0; }
    @Override protected void finalize() { close(); }

    private static native long  nativeCreate(String modelDir, String detName,
                                             String recName, int backend, int numThreads);
    private static native void  nativeDestroy(long handle);
    private static native Line[] nativeRun(long handle, Bitmap bmp);
    private static native Line[] nativeRunFile(long handle, String path);

    public static class Line {
        public final int[] poly;   // 4 corner points (TL, TR, BR, BL)
        public final String text;
        public final float score;
        public Line(int[] p, String t, float s) { poly = p; text = t; score = s; }
    }
}
```

The `backend` arg is one of the
`ppocr_backend` enum values from
`include/ppocr/ppocr.h`: `PPOCR_BACKEND_CPU = 1`,
`PPOCR_BACKEND_OPENCL = 3`, `PPOCR_BACKEND_VULKAN = 4`,
`PPOCR_BACKEND_NNAPI = 7`. Pass 0 (AUTO) to let
`pickBackend()` decide.

## Models: auto-downloader offline mode

The C ABI's auto-downloader is **off by default on
Android** because:

  1. Phones usually do not have reliable network
     access when OCR is being run (camera preview
     pipelines).
  2. The `PPORC_MNN_MIRROR` URL is set at build
     time; shipping a mirror URL in the binary is
     a supply-chain consideration the app author
     must opt into.
  3. Most apps want a one-time install of the
     models and never re-download.

The expected app flow is:

  1. Bundle `*.mnn` files in
     `app/src/main/assets/models/`.
  2. On first launch, copy them to
     `context.getFilesDir() + "/models/"`
     (or `getExternalFilesDir` for user-visible
     storage).
  3. Pass that directory as `modelDir` to the
     `Ppocr` constructor.

The JNI shim forces `cfg.download = 0; cfg.offline = 1;`
in `ppocr_create`, so a missing model file will
fail loudly (`PPOCR_ERR_MODEL`) rather than
silently hitting the network. To re-enable
auto-download you would need to change the
trampoline (or expose a `setOffline(boolean)`
method on the Java class) — out of scope for
the M5-PREP commit.

## ABI

Currently only `arm64-v8a` is wired up. To
support more ABIs, add `x86_64` and `armeabi-v7a`
toolchain files and run `build.sh` once per ABI.
The JNI shim has no ABI-specific code; only the
toolchain file, the build script, and CMake's
linker need to know.

## Verified

Tested on this host with NDK r27 (clang 18):

  ```
  $ ANDROID_NDK=/opt/android-ndk ./platform/android/build.sh
  == Phase 1: build MNN static lib for arm64-v8a (NDK r27)
  ...
  == Phase 2: build JNI .so linking libMNN.a + ppocr_core
  ...
  $ file platform/android/lib/libppocr-jni.so
  ELF 64-bit LSB shared object, ARM aarch64, ...
  $ llvm-nm --dynamic platform/android/lib/libppocr-jni.so | \
      grep Java_ppocr
  T Java_ppocr_Ppocr_create
  T Java_ppocr_Ppocr_destroy
  T Java_ppocr_Ppocr_run
  T Java_ppocr_Ppocr_runFile
  ```

Runtime verification (loading the .so from an
Android app and running inference on a Bitmap)
is out of scope for M5-PREP — that needs an
Android emulator, which this build host does
not have. The host-side build succeeds, the
.so contains the expected exported symbols,
and the C ABI is unchanged from the desktop
build (we run the same `test_preprocess` and
`test_post` against the desktop libppocr_core
on this host).
