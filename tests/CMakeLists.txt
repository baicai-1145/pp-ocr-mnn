# pp-ocr-mnn — postprocess module tests (post owner).
# Standalone — no MNN, no OpenCV. Stb not needed.
cmake_minimum_required(VERSION 3.13)
project(ppocr_post CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sources live at the project root, this CMakeLists lives in tests/.
set(POST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

# test_post: pure C++17, links only our postprocess sources + clipper (header).
add_executable(test_post
               test_post.cpp
               ${POST_ROOT}/src/postprocess/geometry.cpp
               ${POST_ROOT}/src/postprocess/db_post.cpp
               ${POST_ROOT}/src/postprocess/ctc_decode.cpp
               ${POST_ROOT}/third_party/clipper/clipper.cpp)
target_include_directories(test_post PRIVATE
                           ${POST_ROOT}/include
                           ${POST_ROOT}/third_party/clipper
                           # Fallback: sibling m1 include for config.h
                           ${POST_ROOT}/../m1/include)
target_compile_options(test_post PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations)
