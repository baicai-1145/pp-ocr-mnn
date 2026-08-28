# pp-ocr-mnn — top-level CMake
#
# Build the public C ABI as a static library `ppocr_core`, the CLI
# `ppocr_cli`, and unit tests under `tests/`. All platform / backend
# selection lives in src/ppocr.cpp::pickBackend(); this file does not
# branch on platform.
#
# MNN linkage strategy
# --------------------
# We deliberately use find_library against third_party/MNN/build/libMNN.a
# instead of add_subdirectory(third_party/MNN). Reasons:
#   1. MNN's CMake pulls in protobuf, a code generator, and ~5 minutes of
#      configuration on top of a 16 MB prebuilt static library. Re-running
#      that from scratch during a ws/m1 cycle is wasted CPU.
#   2. The prebuilt libMNN.a in third_party/MNN/build was configured with
#      MNN_OPENCL=ON, MNN_VULKAN=ON, MNN_BUILD_CONVERTER=ON. We accept
#      the bloat: those backends live behind MNNForwardType at session
#      create time, and the default MNN_FORWARD_CPU + numThread schedule
#      does not initialize the GPU backends. M1 therefore exercises
#      CPU inference only, while leaving M3 free to flip on OpenCL/Vulkan
#      by passing --backend opencl to the CLI.
#   3. third_party/MNN is a submodule that the contract forbids us from
#      modifying. Re-adding it under add_subdirectory would mean
#      including a separate MNN CMake build that is bound to the same
#      sources; if a sibling worktree (ws/post, ws/tools) rebuilds MNN
#      for itself, our worktree would race on third_party/MNN/build.
#
# If the libMNN.a is missing, this CMake falls back to add_subdirectory
# and forces a CPU-only build with the converter disabled. This is the
# "report the trade-off" path the M1 task description allows.

cmake_minimum_required(VERSION 3.16)
# CMP0057 = NEW lets if() accept the IN_LIST operator, which is what
# the NDK r27 toolchain's flags.cmake uses. Setting it before
# project() means the toolchain file (loaded by CMAKE_TOOLCHAIN_FILE
# during the first project() call) sees the correct policy.
if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()
project(ppocr_mnn
        VERSION 0.1.0
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Toggle targets that are not part of the public ABI. Default ON
# so existing ws/m1 / ws/m2 dev workflows keep working; OFF when
# cross-compiling (M5-PREP) so we don't waste time building
# desktop-only tests on a phone toolchain.
option(PPOCR_BUILD_TESTS "Build pp-ocr-mnn unit tests" ON)
option(PPOCR_BUILD_TOOLS "Build pp-ocr-mnn CLI tools" ON)
option(PPOCR_BUILD_CLS  "Build pp-ocr-mnn cls test driver" ON)

find_package(Threads REQUIRED)

# Warnings: -Wall is the baseline. -Wextra is enabled but treated as
# non-fatal so ws/post can land without our approval; -Werror is OFF
# so we don't hard-fail on third-party headers.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ---- third-party paths ---------------------------------------------------
set(PPOCR_MNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/MNN"
    CACHE PATH "Path to the MNN submodule")
set(PPOCR_STB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb"
    CACHE PATH "Path to the stb_image vendored copy")

# Try the prebuilt static library first. If a future CI environment
# does not ship it, we fall back to a CPU-only add_subdirectory build.
# The PATHS list supports both the host build (build/) and any
# cross-build directory the user has staged (build_android_*/,
# build_ios_*/, build_windows_*/). The first one that contains a
# libMNN.a wins.
set(_mnn_lib_candidates
    "${PPOCR_MNN_ROOT}/build/libMNN.a"
    "${PPOCR_MNN_ROOT}/build/source/libMNN.a"
    "${PPOCR_MNN_ROOT}/build_android_arm64_v8a/libMNN.a"
    "${PPOCR_MNN_ROOT}/build_android_x86_64/libMNN.a"
    "${PPOCR_MNN_ROOT}/build_ios/libMNN.a"
    "${PPOCR_MNN_ROOT}/build_windows/libMNN.a")
find_library(MNN_LIBRARY
             NAMES MNN
             PATHS "${PPOCR_MNN_ROOT}/build"
                   "${PPOCR_MNN_ROOT}/build_android_arm64_v8a"
                   "${PPOCR_MNN_ROOT}/build_android_x86_64"
                   "${PPOCR_MNN_ROOT}/build_ios"
                   "${PPOCR_MNN_ROOT}/build_windows"
             NO_DEFAULT_PATH)

set(_ppocr_using_prebuilt_mnn FALSE)
if(MNN_LIBRARY)
  set(_ppocr_using_prebuilt_mnn TRUE)
  message(STATUS "ppocr-mnn: linking prebuilt MNN at ${MNN_LIBRARY}")
else()
  message(STATUS "ppocr-mnn: no prebuilt libMNN.a — falling back to "
                 "add_subdirectory(third_party/MNN) (CPU only, converter OFF).")
endif()

# ---- libcurl (optional, for the auto-downloader) -----------------------
# The downloader in src/downloader.cpp uses libcurl when available
# (PPOCR_HAS_CURL is defined, the curl target is linked in). Without
# curl, the downloader compiles to a stub that returns PPOCR_ERR_BACKEND
# for any download request; offline / model_dir-local usage still works.
# This matches the contract in include/ppocr/downloader.h and the
# TOOLS-5 / M3 brief: "允许 #include <curl/curl.h> 但 CMake 要
# find_package(CURL) 可选退化：无 curl 时 download=off 只用本地".
find_package(CURL QUIET)
if(NOT CURL_FOUND)
  # find_package on Debian/Ubuntu often fails because the CMake
  # config is missing; fall back to a direct find_library on the
  # common libcurl.so filename. The system curl binary at /usr/bin/curl
  # is not the same package as the dev headers, so we still need a
  # curl/curl.h.
  find_path(PPOCR_CURL_INCLUDE_DIRS
            NAMES curl/curl.h
            HINTS /usr/include /usr/local/include
                  /opt/conda/include /opt/miniconda3/include /root/miniconda3/include
            DOC "Directory containing curl/curl.h")
  find_library(PPOCR_CURL_LIBRARIES
               NAMES curl libcurl
               HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
                     /opt/conda/lib /opt/miniconda3/lib /root/miniconda3/lib
               DOC "Path to libcurl")
  if(PPOCR_CURL_INCLUDE_DIRS AND PPOCR_CURL_LIBRARIES)
    set(PPOCR_HAS_CURL TRUE)
    add_library(CURL::CURL UNKNOWN IMPORTED)
    set_target_properties(CURL::CURL PROPERTIES
      IMPORTED_LOCATION "${PPOCR_CURL_LIBRARIES}"
      INTERFACE_INCLUDE_DIRECTORIES "${PPOCR_CURL_INCLUDE_DIRS}")
    message(STATUS "ppocr-mnn: found libcurl (manual) at ${PPOCR_CURL_LIBRARIES}")
    message(STATUS "ppocr-mnn: curl headers at ${PPOCR_CURL_INCLUDE_DIRS}")
  else()
    set(PPOCR_HAS_CURL FALSE)
    message(STATUS "ppocr-mnn: libcurl not found — downloader will refuse "
                   "network calls (offline / model_dir-local only).")
  endif()
else()
  set(PPOCR_HAS_CURL TRUE)
endif()
if(PPOCR_HAS_CURL)
  message(STATUS "ppocr-mnn: model auto-downloader will use libcurl")
endif()

# ---- public library ------------------------------------------------------
# Aggregates all platform-agnostic sources. Tests and apps link against
# this; platform wrappers (added in M5) will too.
add_library(ppocr_core STATIC
  src/image.cpp
  src/config.cpp
  src/preprocess.cpp
  src/mnn_session.cpp
  src/ppocr.cpp
  src/downloader.cpp
  src/postprocess/db_post.cpp
  src/postprocess/geometry.cpp
  src/postprocess/ctc_decode.cpp
  third_party/clipper/clipper.cpp
)
target_include_directories(ppocr_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/clipper
)
# stb and MNN are third-party; treat their headers as SYSTEM so our
# strict -Wall/-Wextra pass stays focused on ppocr code.
target_include_directories(ppocr_core SYSTEM PUBLIC
  ${PPOCR_STB_ROOT}
  ${PPOCR_MNN_ROOT}/include
  ${PPOCR_MNN_ROOT}/express
)
target_link_libraries(ppocr_core PUBLIC
  Threads::Threads
  dl
)
# JPEG decode parity with cv2.imread (baseline contract). libjpeg-turbo on
# every mainstream distro; see tools/M2_DECODE_ALIGN.md.
option(PPOCR_HAVE_LIBJPEG "Link system libjpeg for bit-exact JPEG decode" ON)
if(PPOCR_HAVE_LIBJPEG)
  find_path(PPOCR_JPEG_INCLUDE_DIRS jpeglib.h)
  find_library(PPOCR_JPEG_LIBRARIES NAMES jpeg)
  if(PPOCR_JPEG_INCLUDE_DIRS AND PPOCR_JPEG_LIBRARIES)
    target_include_directories(ppocr_core PUBLIC ${PPOCR_JPEG_INCLUDE_DIRS})
    target_compile_definitions(ppocr_core PUBLIC PPDECODE_HAVE_LIBJPEG=1)
    target_link_libraries(ppocr_core PUBLIC ${PPOCR_JPEG_LIBRARIES})
  else()
    message(STATUS "libjpeg not found — falling back to stb_image for JPEG")
  endif()
endif()
if(PPOCR_HAS_CURL)
  # Define the macro for every translation unit that includes
  # src/downloader.cpp (or its header). PUBLIC propagates to
  # consumers like ppocr_cli and test_downloader so they can also
  # use the curl path if they want to.
  target_compile_definitions(ppocr_core PUBLIC PPOCR_HAS_CURL=1)
  target_link_libraries(ppocr_core PUBLIC CURL::CURL)
endif()
option(PPOCR_PREP_BILEXACT "Use bit-exact OpenCV INTER_LINEAR replica in det prep (default: float bilinear)" OFF)
if(PPOCR_PREP_BILEXACT)
  target_compile_definitions(ppocr_core PUBLIC PPOCR_PREP_BILEXACT=1)
endif()
if(_ppocr_using_prebuilt_mnn)
  # libMNN.a needs zlib (for weight compression). MNN itself declares
  # -lz transitively, so we propagate that requirement up.
  # Debian/Ubuntu ship libz.so.1 without a libz.so symlink, so
  # find_package(ZLIB) fails unless we hint at the explicit filename.
  set(_ppocr_zlib_candidates
      /usr/lib/x86_64-linux-gnu/libz.so.1
      /usr/lib/libz.so.1
      /lib/x86_64-linux-gnu/libz.so.1
      /lib/libz.so.1)
  find_library(PPOCR_ZLIB_LIBRARY
               NAMES z libz
               PATHS /usr/lib /usr/lib/x86_64-linux-gnu /lib /lib/x86_64-linux-gnu /opt/conda/lib /opt/miniconda3/lib
               NO_DEFAULT_PATH)
  if(NOT PPOCR_ZLIB_LIBRARY)
    foreach(p IN LISTS _ppocr_zlib_candidates)
      if(EXISTS "${p}")
        set(PPOCR_ZLIB_LIBRARY "${p}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT PPOCR_ZLIB_LIBRARY)
    message(FATAL_ERROR "ppocr-mnn: libz not found. Install zlib1g-dev or "
                        "set PPOCR_ZLIB_LIBRARY to its absolute path.")
  endif()
  # libMNN ships hand-written .S files (e.g. _AVX_MNNPackedSparseMatMulEpx4EFMA_ASM.S)
  # without a .note.GNU-stack section, which makes the linker warn about
  # an implicit executable stack. Suppress at link time so our build
  # output is clean — the sections are still safe, just stale.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_link_options("-Wl,-z,noexecstack")
  endif()
  target_link_libraries(ppocr_core PUBLIC
    "${MNN_LIBRARY}"
    "${PPOCR_ZLIB_LIBRARY}"
    pthread)
else()
  # CPU-only, no converter, no express auto-link. This path is only
  # used in development environments without the prebuilt libMNN.a.
  set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)
  set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(MNN_BUILD_TRAIN OFF CACHE BOOL "" FORCE)
  set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(MNN_OPENCL OFF CACHE BOOL "" FORCE)
  set(MNN_VULKAN OFF CACHE BOOL "" FORCE)
  set(MNN_CUDA OFF CACHE BOOL "" FORCE)
  set(MNN_METAL OFF CACHE BOOL "" FORCE)
  set(MNN_NNAPI OFF CACHE BOOL "" FORCE)
  set(MNN_OPENGL OFF CACHE BOOL "" FORCE)
  set(MNN_OPENMP OFF CACHE BOOL "" FORCE)
  set(MNN_USE_THREAD_POOL ON CACHE BOOL "" FORCE)
  set(MNN_USE_SSE ON CACHE BOOL "" FORCE)
  add_subdirectory(${PPOCR_MNN_ROOT} ${CMAKE_BINARY_DIR}/_mnn_build EXCLUDE_FROM_ALL)
  target_link_libraries(ppocr_core PUBLIC MNN)
endif()

# ---- MNN CUDA companion lib (optional) ----------------------------------
# MNN's Linux CUDA backend is a SHARED companion (libMNN_Cuda_Main.so) whose
# implementation lives outside libMNN.a; the backend factory (Register.cpp
# object) is inside libMNN.a. Linking the .so directly AND exporting dynamic
# symbols lets its weak references into the statically-linked MNN core
# resolve. Without this, --backend cuda silently falls back to CPU
# ("Can't Find type=2 backend").
set(_mnn_cuda_so "${PPOCR_MNN_ROOT}/build/source/backend/cuda/libMNN_Cuda_Main.so")

# ---- CLI ----------------------------------------------------------------
add_executable(ppocr_cli apps/ppocr_cli.cpp)
target_link_libraries(ppocr_cli PRIVATE ppocr_core)
if(EXISTS "${_mnn_cuda_so}")
  message(STATUS "ppocr-mnn: linking MNN CUDA companion ${_mnn_cuda_so}")
  # --no-as-needed MUST precede the .so on the link line; CMake places
  # target_link_options AFTER libraries, so we wrap it into the link items
  # themselves (policy CMP0099 / classic behaviour).
  target_compile_definitions(ppocr_cli PRIVATE PPOCR_MNN_CUDA_COMPANION)
  target_link_libraries(ppocr_cli PRIVATE
    "-Wl,--no-as-needed" "${_mnn_cuda_so}" "-Wl,--as-needed" "-Wl,--export-dynamic")
endif()

# ---- tests --------------------------------------------------------------
enable_testing()
if(PPOCR_BUILD_TESTS)
add_executable(test_preprocess tests/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE ppocr_core)
add_test(NAME test_preprocess COMMAND test_preprocess)
# test_preprocess writes test_config.json into the cwd at run time; pin
# the working dir to the build dir so it doesn't pollute the source tree.
set_tests_properties(test_preprocess PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# Auto-downloader test driver (TOOLS-5 / M3). It links against
# ppocr_core via --whole-archive so the `ensure_model`, `load_registry`,
# and `find_entry` symbols are pulled in (they are only called from
# src/ppocr.cpp, which doesn't link this test binary). See
# tests/test_downloader.cpp for the scenarios.
if(PPOCR_BUILD_TESTS)
add_executable(test_downloader tests/test_downloader.cpp)
# Static libs don't pull unreferenced objects into a downstream
# executable, so we wrap ppocr_core with --whole-archive. MNN, curl,
# and zlib are linked normally (the same set ppocr_core uses).
target_link_libraries(test_downloader PRIVATE
  -Wl,--whole-archive ppocr_core -Wl,--no-whole-archive
  "${MNN_LIBRARY}"
  "${PPOCR_ZLIB_LIBRARY}"
  Threads::Threads
  dl
)
if(PPOCR_HAS_CURL)
  target_link_libraries(test_downloader PRIVATE CURL::CURL)
endif()
endif()

# M3 cls integration test. Loads the cls model and verifies that
# ppocr_run() with cls enabled reports cls_ms > 0 when there are
# det boxes, and cls_ms == 0 when cls_name is null.
if(PPOCR_BUILD_CLS)
add_executable(test_cls tests/test_cls.cpp)
target_link_libraries(test_cls PRIVATE ppocr_core)
add_test(NAME test_cls COMMAND test_cls)
set_tests_properties(test_cls PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# ---- JNI (Android) ---------------------------------------------------------
# Off by default. Enabled with -DPPOCR_BUILD_JNI=ON, typically by the
# Android toolchain file (platform/android/android-arm64-v8a.toolchain.cmake).
# The JNI .so is the only M5-PREP build that needs the whole
# libppocr_core + libMNN.a bundled into a single shared object.
# Other Android users are expected to build via the host CMake and
# just include the JNI source via `add_subdirectory(platform/android)`.
option(PPOCR_BUILD_JNI "Build the Android JNI wrapper (libppocr-jni.so)" OFF)
if(PPOCR_BUILD_JNI)
  if(NOT ANDROID OR NOT DEFINED ANDROID_ABI)
    message(FATAL_ERROR "PPOCR_BUILD_JNI=ON requires the Android NDK toolchain "
                        "(set CMAKE_TOOLCHAIN_FILE to one of the platform/android/*.cmake).")
  endif()
  add_library(ppocr-jni SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/platform/android/ppocr_jni.cc"
  )
  # ppocr_jni.cc uses Android-specific log and bitmap APIs via
  # <jni.h> + <android/log.h>. The NDK ships these in
  # $NDK/sources/android/support/include and the sysroot that the
  # toolchain adds. We link against `log` for __android_log_print.
  target_link_libraries(ppocr-jni PRIVATE
    ppocr_core
    log
  )
  target_compile_definitions(ppocr-jni PRIVATE PPOCR_BUILD_JNI=1)
  # The JNI shim doesn't need the existing host test data files;
  # the Java side is responsible for asset management.
endif()

# ---- install rules (M5-PREP) -----------------------------------------------
# `cmake --install .` is the standard way to package pp-ocr-mnn
# for downstream users. We install the public headers, the static
# library, and the CLI binary. The MNN library is intentionally
# NOT installed; downstream users who want a self-contained
# distribution should re-link against their own libMNN.a.
#
# Header layout under the install prefix:
#   <prefix>/include/ppocr/ppocr.h
#   <prefix>/include/ppocr/downloader.h
#   <prefix>/include/ppocr/config.h
#   ... (all the public headers in include/ppocr/ and
#        include/ppocr/postprocess/)
include(GNUInstallDirs)
install(TARGETS ppocr_cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS ppocr_core
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ppocr
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
