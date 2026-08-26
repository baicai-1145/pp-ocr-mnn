# platform/ios — iOS / macOS CocoaPods / Swift Package wrapper

Thin Objective-C wrapper around the public C ABI
(`include/ppocr/ppocr.h`) for iOS and macOS apps. M5-PREP
ships only the **wiring**, not a binary XCFramework —
producing one requires a Mac with Xcode and is out of
scope for this host (Linux only, no `swift` or `xcodebuild`).

## Files

  * `PpocrEngine.mm` — Objective-C wrapper. The
    `PpocrEngine` class is exposed to Swift as
    `import Ppocr`. Each `run` returns
    `NSArray<NSDictionary*>` of `{poly, text, score}` per
    line.
  * `Package.swift` — Swift Package manifest. Builds the
    C ABI targets and the ObjC wrapper into a single
    `Ppocr` library.

## What this commit does NOT do

  * **No XCFramework is built.** Producing one requires
    `xcodebuild -create-xcframework` on macOS, which we
    cannot do on this Linux host. The user's own Mac build
    pipeline will produce the XCFramework from the host
    repo's `src/` and `include/` and link it into the
    `PpocrC` target.
  * **No Metal / CoreML backend is enabled.** Those are
    MNN compile-time switches (`MNN_METAL=ON`,
    `MNN_COREML=ON`) that affect `libMNN.a` at the C++ level,
    not at the C ABI level. The C ABI is backend-agnostic;
    `ppocr_config.backend = PPOCR_BACKEND_METAL` simply
    passes through to `pickBackend()` (see
    `src/ppocr.cpp`) which dispatches to MNN's Metal
    backend. The Metal backend is "wired" in the sense
    that the enum exists and the C ABI exposes it; the
    actual Metal compute kernels are MNN's responsibility
    and live behind the MNN build flag.

## How `pickBackend()` is enabled for Metal / CoreML

The pickBackend() function in `src/ppocr.cpp` already
implements:

  ```cpp
  case PPOCR_BACKEND_METAL:  return MNN_FORWARD_METAL;
  case PPOCR_BACKEND_COREML: return MNN_FORWARD_NN;  // CoreML
  ```

The runtime check is whether the linked MNN library has
those backends compiled in. MNN's CMake:

  * `MNN_METAL=ON` adds `source/backend/metal/*` to the
    build. The resulting `libMNN.a` exports Metal-related
    symbols.
  * `MNN_COREML=ON` adds `source/backend/coreml/*`. Same
    idea.

To produce a Metal-enabled `libMNN.a` for iOS:

  ```bash
  # On macOS, with Xcode installed:
  cmake -B build-ios-arm64 -S third_party/MNN \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_OSX_SYSROOT=iphoneos \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
        -DMNN_BUILD_SHARED_LIBS=OFF \
        -DMNN_METAL=ON \
        -DMNN_COREML=ON \
        -DMNN_OPENCL=OFF -DMNN_VULKAN=OFF \
        -DMNN_CUDA=OFF -DMNN_ARM82=OFF \
        -DCMAKE_BUILD_TYPE=Release
  cmake --build build-ios-arm64 --target MNN -j
  ```

The resulting `libMNN.a` is then linked into the
`PpocrC` target of the Swift Package. The user can either:

  1. Replace the `linkedLibrary("MNN")` in `Package.swift`
     with a `binaryTarget` pointing at a prebuilt
     `MNN.xcframework`, or
  2. Use CocoaPods (the legacy path; see the deprecated
     `MNN.podspec` in `third_party/MNN/`) to pull in
     `MNN` as a Pod and link it into the ObjC wrapper.

The current `Package.swift` uses option 1 with a
placeholder; option 2 is left as an exercise for the
consumer.

## Swift usage (illustrative, not buildable here)

```swift
import Ppocr

let engine = try PpocrEngine(
    modelDir: Bundle.main.resourcePath! + "/models",
    detName: "PP-OCRv6_tiny_det",
    recName: "PP-OCRv6_tiny_rec",
    backend: 5,            // PPOCR_BACKEND_METAL
    numThreads: 4)
defer { engine.close() }

let lines = try engine.runFile("/path/to/image.jpg")
for line in lines {
    print(line["text"] ?? "", line["score"] ?? 0)
}
```

The `backend: 5` argument maps to
`PPOCR_BACKEND_METAL` from the public C ABI
(`include/ppocr/ppocr.h`). 0 is `AUTO` and lets
`pickBackend()` choose at runtime.

## Models: same offline-only policy as Android

The ObjC wrapper sets `cfg.download = 0; cfg.offline = 1;`
in `PpocrEngine.mm`, mirroring the Android JNI shim's
policy. iOS apps are expected to bundle the `.mnn` files
in the app bundle (e.g. `Resources/models/`) and pass the
resolved `Bundle.main.resourcePath + "/models"` as
`modelDir`. The auto-downloader is intentionally disabled
on mobile because (a) the network may not be available,
(b) shipping a `PPORC_MNN_MIRROR` URL in the binary is a
supply-chain consideration the app author must opt into.

## Verified

This commit **does not verify the iOS build on this host**.
The Linux container has no Swift toolchain and no Xcode.
What is verified:

  1. The `PpocrEngine.mm` source compiles in isolation
     against a host libMNN.a (we did this as part of
     compiling the ObjC wrapper for syntactic check on
     Linux with `clang -x objective-c++`).
  2. The C ABI itself (the .h file the wrapper includes)
     is unchanged from the desktop / Android builds; the
     iOS wrapper just adds a class around it.
  3. The MNN backend enum values used by the wrapper
     (`PPOCR_BACKEND_METAL = 5`, `PPOCR_BACKEND_COREML = 6`)
     exist in `include/ppocr/ppocr.h` and are forwarded
     correctly to MNN's `MNN_FORWARD_METAL` / `MNN_FORWARD_NN`
     in `src/ppocr.cpp::pickBackend()`.

The end-to-end iOS build verification (xcframework
production, simulator launch, real-device run) is the
responsibility of the consumer's Mac CI, not M5-PREP.
