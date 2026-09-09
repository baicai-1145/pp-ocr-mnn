// E2E: auto-download from the default mirror into an empty model dir.
// Fetches the tiniest det+rec pair, verifies sha256 handling inside
// ensure_model, then runs one image through the full pipeline.
#include "ppocr/ppocr.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
  const char* img = (argc > 1) ? argv[1] : "/root/ocr_test_imgs/de/00.jpg";
  const char* mdir = (argc > 2) ? argv[2] : "/tmp/ppocr_dl_e2e";

  char cmd[512];
  snprintf(cmd, sizeof cmd, "rm -rf %s && mkdir -p %s/configs", mdir, mdir);
  if (system(cmd) != 0) return 2;
  // registry only — every .mnn must come from the mirror
  snprintf(cmd, sizeof cmd,
           "cp configs/registry.json %s/configs/registry.json", mdir);
  if (system(cmd) != 0) return 2;

  ppocr_config cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.model_dir  = mdir;
  cfg.det_name   = "PP-OCRv6_tiny_det";
  cfg.rec_name   = "PP-OCRv6_tiny_rec";
  cfg.backend    = PPOCR_BACKEND_CPU;
  cfg.num_threads= 4;
  cfg.offline    = 0;
  cfg.download   = 1;

  char err[256] = {0};
  ppocr_engine* e = NULL;
  ppocr_status st = ppocr_create(&cfg, &e, err, sizeof err);
  if (st != PPOCR_OK) {
    fprintf(stderr, "ppocr_create failed: %s (%s)\n",
            ppocr_status_string(st), err);
    return 1;
  }
  ppocr_result* r = NULL;
  st = ppocr_run_file(e, img, &r);
  if (st != PPOCR_OK) {
    fprintf(stderr, "ppocr_run_file failed: %s\n", ppocr_status_string(st));
    return 1;
  }
  printf("auto-download E2E: %d boxes, first text: %s\n",
         r->n_lines, r->n_lines ? (r->lines[0].text ? r->lines[0].text : "") : "-");
  return 0;
}
