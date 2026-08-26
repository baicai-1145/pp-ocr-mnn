// pp-ocr-mnn — minimal C API example.
//
// Build:
//   cc examples/c_api_demo.c \
//      build-main/libppocr_core.a \
//      third_party/MNN/build/libMNN.a \
//      -lpthread -ldl -lm -lz \
//      -I include \
//      -o c_api_demo
//
// Or with the system install:
//   cc examples/c_api_demo.c -lppocr_core -lMNN \
//      -lpthread -ldl -lm -lz \
//      -I /usr/local/include \
//      -o c_api_demo
//
// Or via the bundled examples/CMakeLists.txt:
//   cmake -B build -S examples \
//         -DPPOCR_EXAMPLES_MNN_ROOT=third_party/MNN
//   cmake --build build
//
// Run:
//   ./c_api_demo /path/to/image.jpg /path/to/models
//
// This is a smoke test of the C ABI only — it does not
// measure CER, does not compare against any baseline.
// The goal is to prove that:
//   1. The C ABI is callable from plain C (no C++ required).
//   2. The install rules produce a usable header + lib
//      layout that downstream consumers can link against.
//   3. create / run_file / destroy all wire up correctly
//      when linked against the static libMNN.a.
//
// It compiles under both gcc and clang with -std=c99 -Wall.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppocr/ppocr.h"

static void print_result(const ppocr_result* r) {
  printf("backend: %s, n_lines: %d, det_ms: %.1f, rec_ms: %.1f, "
         "cls_ms: %.1f, total_ms: %.1f\n",
         r->backend_used, r->n_lines,
         r->det_ms, r->rec_ms, r->cls_ms, r->total_ms);
  for (int i = 0; i < r->n_lines; ++i) {
    const ppocr_line* ln = &r->lines[i];
    printf("  [%3d] (%4d,%4d)-(%4d,%4d)-(%4d,%4d)-(%4d,%4d) "
           "score=%.3f text=\"%s\"\n",
           i,
           ln->poly[0], ln->poly[1], ln->poly[2], ln->poly[3],
           ln->poly[4], ln->poly[5], ln->poly[6], ln->poly[7],
           ln->score, ln->text ? ln->text : "");
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <image> <model_dir>\n"
                    "  <image>     path to a jpg/png/bmp file\n"
                    "  <model_dir> directory containing .mnn + configs/registry.json\n",
            argv[0]);
    return 2;
  }
  const char* image_path = argv[1];
  const char* model_dir  = argv[2];

  int maj = 0, min = 0, pat = 0;
  if (ppocr_version(&maj, &min, &pat) != PPOCR_OK) {
    fprintf(stderr, "ppocr_version failed\n");
    return 1;
  }
  printf("ppocr-mnn v%d.%d.%d\n", maj, min, pat);

  ppocr_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.model_dir   = model_dir;
  cfg.det_name    = NULL;       // default: PP-OCRv6_tiny_det
  cfg.rec_name    = NULL;       // default: PP-OCRv6_tiny_rec
  cfg.cls_name    = NULL;       // off
  cfg.backend     = PPOCR_BACKEND_CPU;
  cfg.num_threads = 4;
  cfg.rec_batch   = 8;
  cfg.max_side    = 0;
  cfg.offline     = 1;          // don't try to download
  cfg.download    = 0;

  ppocr_engine* e = NULL;
  char err_buf[256] = {0};
  ppocr_status st = ppocr_create(&cfg, &e, err_buf, sizeof(err_buf));
  if (st != PPOCR_OK) {
    fprintf(stderr, "ppocr_create failed: %s (%s)\n",
            ppocr_status_string(st), err_buf);
    return 1;
  }
  printf("engine created (handle=%p)\n", (void*)e);

  ppocr_result* res = NULL;
  st = ppocr_run_file(e, image_path, &res);
  if (st != PPOCR_OK) {
    fprintf(stderr, "ppocr_run_file(%s) failed: %s\n",
            image_path, ppocr_status_string(st));
    ppocr_destroy(e);
    return 1;
  }
  print_result(res);

  ppocr_destroy(e);
  printf("engine destroyed\n");
  return 0;
}
