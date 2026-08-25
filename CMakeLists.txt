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
project(ppocr_mnn
        VERSION 0.1.0
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
set(_mnn_lib_candidates
    "${PPOCR_MNN_ROOT}/build/libMNN.a"
    "${PPOCR_MNN_ROOT}/build/source/libMNN.a")
find_library(MNN_LIBRARY
             NAMES MNN
             PATHS "${PPOCR_MNN_ROOT}/build"
             NO_DEFAULT_PATH)

set(_ppocr_using_prebuilt_mnn FALSE)
if(MNN_LIBRARY)
  set(_ppocr_using_prebuilt_mnn TRUE)
  message(STATUS "ppocr-mnn: linking prebuilt MNN at ${MNN_LIBRARY}")
else()
  message(STATUS "ppocr-mnn: no prebuilt libMNN.a — falling back to "
                 "add_subdirectory(third_party/MNN) (CPU only, converter OFF).")
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

# ---- CLI ----------------------------------------------------------------
add_executable(ppocr_cli apps/ppocr_cli.cpp)
target_link_libraries(ppocr_cli PRIVATE ppocr_core)

# ---- tests --------------------------------------------------------------
enable_testing()
add_executable(test_preprocess tests/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE ppocr_core)
add_test(NAME test_preprocess COMMAND test_preprocess)
# test_preprocess writes test_config.json into the cwd at run time; pin
# the working dir to the build dir so it doesn't pollute the source tree.
set_tests_properties(test_preprocess PROPERTIES
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
