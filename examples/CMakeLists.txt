# pp-ocr-mnn — examples/
#
# Builds the small standalone C example that exercises the
# public C ABI. Designed to be invoked from the install prefix
# (`cc c_api_demo.c -lppocr_core -lMNN ...`) or from the source
# tree using the host's prebuilt libMNN.a.
#
# Note: this is intentionally NOT added to the top-level CMake
# (so it does not get built as part of `make` in build-main/).
# Build with:
#   cmake -B examples/build -S examples \
#         -DPPOCR_MNN_ROOT=third_party/MNN
#   cmake --build examples/build

cmake_minimum_required(VERSION 3.16)
project(ppocr_examples C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

find_package(Threads REQUIRED)

# libcurl: optional (matches the main project's behavior — see
# include/ppocr/downloader.h). When the host was built with
# PPOCR_HAS_CURL=1 the downloader is linked against libcurl, so
# the consumer must link against it too. We try find_package
# first, then fall back to find_library.
find_package(CURL QUIET)
if(NOT CURL_FOUND)
  find_library(PPOCR_EXAMPLES_CURL_LIB
               NAMES curl libcurl
               PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
                     /opt/conda/lib /opt/miniconda3/lib /root/miniconda3/lib)
  if(PPOCR_EXAMPLES_CURL_LIB)
    set(PPOCR_HAS_CURL TRUE)
  endif()
else()
  set(PPOCR_HAS_CURL TRUE)
endif()

# Locate the prebuilt libppocr_core.a. We expect the user to
# either:
#   (a) install via `cmake --install build-main` and have
#       find_package handle it, or
#   (b) point PPOCR_EXAMPLES_MNN_ROOT at a third_party/MNN that
#       has a build/ or build_android_*/libMNN.a.
# For M5-PREP we use the simpler (b) path.
set(PPOCR_EXAMPLES_MNN_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/MNN"
    CACHE PATH "Path to the MNN submodule (with build/ or build_android_*/)")

# Public headers: the source-tree include/.
set(PPOCR_EXAMPLES_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    CACHE PATH "Path to the public include/ directory")

# Find libppocr_core.a in the source tree's build-main/, or
# fall back to a system install.
find_library(PPOCR_EXAMPLES_CORE_LIB
             NAMES ppocr_core
             HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../build-main"
                   "${CMAKE_CURRENT_SOURCE_DIR}/../build-linux"
             NO_DEFAULT_PATH)
if(NOT PPOCR_EXAMPLES_CORE_LIB)
  # system install: /usr/local/lib/libppocr_core.a
  find_library(PPOCR_EXAMPLES_CORE_LIB NAMES ppocr_core)
endif()
if(NOT PPOCR_EXAMPLES_CORE_LIB)
  message(FATAL_ERROR
    "libppocr_core.a not found. Either:\n"
    "  - Build the host project first: cmake -B build-main .. && make -C build-main\n"
    "  - Or install it: make -C build-main install DESTDIR=/some/prefix\n")
endif()

# libMNN.a from the same root.
find_library(PPOCR_EXAMPLES_MNN_LIB
             NAMES MNN
             PATHS "${PPOCR_EXAMPLES_MNN_ROOT}/build"
                   "${PPOCR_EXAMPLES_MNN_ROOT}/build_android_arm64_v8a"
             NO_DEFAULT_PATH)
if(NOT PPOCR_EXAMPLES_MNN_LIB)
  find_library(PPOCR_EXAMPLES_MNN_LIB NAMES MNN)
endif()
if(NOT PPOCR_EXAMPLES_MNN_LIB)
  message(FATAL_ERROR
    "libMNN.a not found under ${PPOCR_EXAMPLES_MNN_ROOT}. "
    "Build MNN for your platform first.")
endif()

# libz: same Debian workaround as the main project.
find_library(PPOCR_EXAMPLES_ZLIB_LIB
             NAMES z libz
             PATHS /usr/lib /usr/lib/x86_64-linux-gnu /lib /lib/x86_64-linux-gnu
             NO_DEFAULT_PATH)
if(NOT PPOCR_EXAMPLES_ZLIB_LIB)
  foreach(p /usr/lib/x86_64-linux-gnu/libz.so.1
             /usr/lib/libz.so.1
             /lib/x86_64-linux-gnu/libz.so.1
             /lib/libz.so.1)
    if(EXISTS "${p}")
      set(PPOCR_EXAMPLES_ZLIB_LIB "${p}")
      break()
    endif()
  endforeach()
endif()

# c_api_demo: pure C. This is the only way to verify the C ABI
# is callable from a non-C++ consumer.
add_executable(c_api_demo c_api_demo.c)
target_include_directories(c_api_demo PRIVATE "${PPOCR_EXAMPLES_INCLUDE_DIR}")
target_link_libraries(c_api_demo PRIVATE
  "${PPOCR_EXAMPLES_CORE_LIB}"
  "${PPOCR_EXAMPLES_MNN_LIB}"
  Threads::Threads
  dl
  m
  stdc++)
if(PPOCR_EXAMPLES_ZLIB_LIB)
  target_link_libraries(c_api_demo PRIVATE "${PPOCR_EXAMPLES_ZLIB_LIB}")
endif()
if(PPOCR_HAS_CURL)
  if(TARGET CURL::CURL)
    target_link_libraries(c_api_demo PRIVATE CURL::CURL)
  elseif(PPOCR_EXAMPLES_CURL_LIB)
    target_link_libraries(c_api_demo PRIVATE "${PPOCR_EXAMPLES_CURL_LIB}")
  endif()
endif()
set_target_properties(c_api_demo PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON)
