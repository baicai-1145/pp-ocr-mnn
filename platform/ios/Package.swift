// swift-tools-version:5.5
//
// ppocr-mnn — Swift Package wrapping the public C ABI.
//
// This package is **not** a binary distribution of the C ABI;
// it expects the user to either:
//   1. Add `ppocr-core` as a binary target pointing at a
//      prebuilt XCFramework (the typical iOS distribution path),
//      OR
//   2. Add the C sources from the host repo via a local path
//      dependency during development.
//
// The intent is to give a single `import Ppocr` to Swift
// consumers without forcing them to learn the C ABI layout.
//
// The package builds the Objective-C wrapper (PpocrEngine.mm)
// against the C ABI headers, then re-exports the public class
// to Swift. The Metal / CoreML backends live behind the
// `MNN_METAL=ON` / `MNN_COREML=ON` compile-time switches at
// MNN build time; this Package does not enable them by
// default because (a) the resulting XCFramework needs a Mac
// to build, which this commit does not have, and (b) the
// user can opt in by overriding the binary target URL.

import PackageDescription

let package = Package(
  name: "Ppocr",
  platforms: [
    .iOS(.v14),
    .macOS(.v12),
  ],
  products: [
    .library(
      name: "Ppocr",
      targets: ["Ppocr"]),
  ],
  targets: [
    // The C ABI surface. `ppocr.h` ships under
    // `include/ppocr/ppocr.h` in the host repo; we expose it
    // as a system header so `#include <ppocr/ppocr.h>` works
    // both from this Package's sources and from any user code
    // that includes the same path.
    .target(
      name: "PpocrC",
      path: "..",
      exclude: [
        "build", "build-main", "build-tools", "build-cls",
        "build_android_arm64_v8a", "build_var",
        "models", "results", "_downloads",
        "tests/data", ".github",
      ],
      publicHeadersPath: "include",
      linkerSettings: [
        // Pull in the MNN static lib. The user is expected to
        // override this with a real .xcframework path in their
        // Package.swift (see README.md for the recipe).
        .linkedLibrary("MNN"),
        .linkedLibrary("z"),
        .linkedLibrary("c++"),
      ]),
    // The Objective-C wrapper that bridges ppocr_engine* ->
    // Swift-friendly NSArray<NSDictionary*>.
    .target(
      name: "Ppocr",
      dependencies: ["PpocrC"],
      path: "platform/ios",
      sources: ["PpocrEngine.mm"],
      publicHeadersPath: "platform/ios",
      cSettings: [
        .headerSearchPath("../include"),
      ]),
  ]
)
