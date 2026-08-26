# pp-ocr-mnn — Android (arm64-v8a) CMake toolchain
#
# Use with:  cmake -B build-android-arm64 \
#             -DCMAKE_TOOLCHAIN_FILE=platform/android/android-arm64-v8a.toolchain.cmake \
#             -DANDROID_NDK=/opt/android-ndk \
#             -DANDROID_PLATFORM=android-21 \
#             -DANDROID_ABI=arm64-v8a \
#             -DCMAKE_BUILD_TYPE=Release
#
# Tested with NDK r27 (clang 18). The default ANDROID_PLATFORM=21
# matches the MNN CMake minimum (MNN's `set(MNN_MINIMUM_API_LEVEL 21)`
# in CMakeLists.txt).
#
# This toolchain is a thin wrapper over NDK's bundled
# android.toolchain.cmake. We default the variables that matter
# for our build (NDK location, API level, ABI) so the command line
# above can be a one-liner, and we add an explicit -fPIC so the
# static lib links into the .so cleanly. No MNN source modification
# happens here — `third_party/MNN` is read-only.
#
# Note: we deliberately do not include MNN as a subdirectory.
# MNN's `add_subdirectory` pulls in protobuf, codegen, and the
# converter; cross-compiling those is fragile and slow. Instead,
# the main project's CMake (the file that includes this toolchain)
# is expected to pass a prebuilt libMNN.a via PPOCR_MNN_ROOT.
# See platform/android/README.md for the bootstrap recipe that
# builds libMNN.a for arm64-v8a with the same toolchain.

if(DEFINED ENV{ANDROID_NDK_HOME})
  set(ANDROID_NDK "$ENV{ANDROID_NDK_HOME}")
endif()
if(NOT ANDROID_NDK)
  set(ANDROID_NDK "/opt/android-ndk")
endif()
get_filename_component(ANDROID_NDK "${ANDROID_NDK}" ABSOLUTE)

# Pick a sensible default if the caller didn't specify.
if(NOT DEFINED ANDROID_PLATFORM)
  set(ANDROID_PLATFORM "android-21")
endif()
if(NOT DEFINED ANDROID_ABI)
  set(ANDROID_ABI "arm64-v8a")
endif()

# Load NDK's bundled toolchain file (it expects these variables to
# already be set: ANDROID_NDK, ANDROID_PLATFORM, ANDROID_ABI).
# NDK r27's flags.cmake uses the IN_LIST operator in if(), which
# requires CMP0057 = NEW. The MNN submodule declares
# cmake_minimum_required(VERSION 3.0), which would set the policy
# to OLD. Set CMP0057 = NEW at the top level (CMP variables are
# stack-scoped, not just project-scoped) and pass the same shim to
# MNN's project() via CMAKE_PROJECT_MNN_INCLUDE.
if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()
set(CMAKE_PROJECT_MNN_INCLUDE
    "${CMAKE_CURRENT_LIST_DIR}/android-project-init.cmake"
    CACHE FILEPATH "pre-project init shim for MNN")
include("${ANDROID_NDK}/build/cmake/android.toolchain.cmake")

# PIC is required for linking into the JNI .so.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# C++17 to match the host build. NDK r27's clang defaults to
# gnu++17 anyway, but make it explicit.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Thrift warnings to MNN's third-party headers. -w mirrors the
# host build's -w on the variant-only MNN builds.
add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations)
