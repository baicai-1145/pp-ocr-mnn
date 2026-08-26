# pp-ocr-mnn — Android NDK pre-project CMake policy shim
#
# MNN's CMakeLists.txt (third_party/MNN/CMakeLists.txt) starts
# with `cmake_minimum_required(VERSION 3.0)`. Under CMake 3.27+
# the default for CMP0057 (IN_LIST in if()) is NEW, but the
# MNN CMakeLists does not explicitly set it, and the NDK r27
# toolchain (toolchains/llvm/prebuilt/.../android.toolchain.cmake)
# uses IN_LIST in its flags.cmake. The two together produce:
#
#   CMake Error: if given arguments:
#       "hwaddress" "IN_LIST" "ANDROID_SANITIZE"
#     Unknown arguments specified
#
# We work around by prepending this file via CMAKE_PROJECT_INCLUDE,
# which is sourced BEFORE the project() call in MNN's CMakeLists.
# Setting CMP0057 = NEW here lets the NDK's flags.cmake evaluate
# IN_LIST without error.
#
# This file is referenced from platform/android/android-arm64-v8a.toolchain.cmake
# via `set(CMAKE_PROJECT_INCLUDE_BEFORE "${CMAKE_CURRENT_LIST_FILE}")`.
if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
