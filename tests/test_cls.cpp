// pp-ocr-mnn — M3 cls integration test.
//
// Validates the cls wiring end-to-end via the C ABI:
//   1. Create two engines, one with cls_name=PP-LCNet_x1_0_textline_ori
//      and one with cls_name=null.
//   2. Run ppocr_run_file() on the same image with both engines.
//   3. Assert:
//      - cls-on engine's result has cls_ms > 0 (cls was called).
//      - cls-off engine's result has cls_ms == 0.
//      - cls-on engine's result has the same n_lines as cls-off
//        (the cls does not drop or add boxes).
//
// This is an integration test, not a unit test of the cls model
// (we trust the cls.mnn to be a correct PPLCNet port). The
// interesting assertions are about the *wiring* of run_cls_sync
// in the rec pipeline: it fires, it sets cls_ms, and it doesn't
// perturb the box count or the final text content.

#include "ppocr/ppocr.h"
#include "ppocr/config.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

#define CHECK(cond) do {                                                  \
  if (!(cond)) {                                                          \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    std::exit(1);                                                         \
  }                                                                       \
} while (0)

}  // namespace


int main() {
  // Use a real test image: /root/ocr_test_imgs/zh/04.jpg (SOLINSKY / ALLEY).
  const std::string model_dir = "/root/pp-ocr-mnn/models";
  const std::string test_image = "/root/ocr_test_imgs/zh/04.jpg";

  // --- 1. set up engine WITH cls ---
  ppocr_config cfg{};
  cfg.model_dir  = model_dir.c_str();
  cfg.det_name   = "PP-OCRv4_mobile_det";
  cfg.rec_name   = "PP-OCRv4_mobile_rec";
  cfg.cls_name   = "PP-LCNet_x1_0_textline_ori";
  cfg.registry_path = nullptr;
  cfg.mirror     = nullptr;
  cfg.backend    = PPOCR_BACKEND_AUTO;
  cfg.num_threads = 1;
  cfg.rec_batch  = 8;
  cfg.max_side   = 0;
  cfg.offline    = 1;
  cfg.download   = 0;

  char err[256] = {0};
  ppocr_engine* e_on = nullptr;
  ppocr_status st = ppocr_create(&cfg, &e_on, err, sizeof(err));
  if (st != PPOCR_OK) {
    std::fprintf(stderr, "ppocr_create (cls on) failed: %s (%s)\n",
                 ppocr_status_string(st), err);
    return 1;
  }

  // --- 2. set up engine WITHOUT cls ---
  ppocr_config cfg_off = cfg;
  cfg_off.cls_name = nullptr;
  ppocr_engine* e_off = nullptr;
  st = ppocr_create(&cfg_off, &e_off, err, sizeof(err));
  if (st != PPOCR_OK) {
    std::fprintf(stderr, "ppocr_create (cls off) failed: %s (%s)\n",
                 ppocr_status_string(st), err);
    return 2;
  }

  // --- 3. run on the test image with both engines ---
  ppocr_result* res_on = nullptr;
  st = ppocr_run_file(e_on, test_image.c_str(), &res_on);
  if (st != PPOCR_OK) {
    std::fprintf(stderr, "ppocr_run_file (cls on) failed: %s\n",
                 ppocr_status_string(st));
    return 3;
  }
  std::fprintf(stderr,
               "cls ON:  det=%.2f rec=%.2f cls=%.2f total=%.2f ms, n_lines=%d\n",
               res_on->det_ms, res_on->rec_ms, res_on->cls_ms, res_on->total_ms,
               res_on->n_lines);

  ppocr_result* res_off = nullptr;
  st = ppocr_run_file(e_off, test_image.c_str(), &res_off);
  if (st != PPOCR_OK) {
    std::fprintf(stderr, "ppocr_run_file (cls off) failed: %s\n",
                 ppocr_status_string(st));
    return 4;
  }
  std::fprintf(stderr,
               "cls OFF: det=%.2f rec=%.2f cls=%.2f total=%.2f ms, n_lines=%d\n",
               res_off->det_ms, res_off->rec_ms, res_off->cls_ms,
               res_off->total_ms, res_off->n_lines);

  // --- 4. assertions ---
  // 4a. cls on the upright SOLINSKY/ALLEY image: all crops are 0°,
  // cls should still be called (so cls_ms > 0) and the rec text
  // should match cls-off (cls didn't rotate anything).
  CHECK(res_on->n_lines > 0);
  CHECK(res_on->cls_ms > 0.f);
  CHECK(res_off->n_lines == res_on->n_lines);
  CHECK(res_off->cls_ms == 0.f);
  // total_ms = det + rec + cls (when cls is on)
  CHECK(res_on->total_ms >=
        res_on->det_ms + res_on->rec_ms + res_on->cls_ms - 0.1f);
  CHECK(res_off->total_ms ==
        res_off->det_ms + res_off->rec_ms);
  // Text content: for an upright image with no upside-down lines,
  // cls on == cls off (no rotation triggered).
  for (int i = 0; i < res_on->n_lines; ++i) {
    CHECK(std::string(res_on->lines[i].text) ==
          std::string(res_off->lines[i].text));
  }

  // --- 6. rot180 of the same image: cls on must rotate crops 180°,
  //         cls off returns empty text on the rotated crops.
  // We rebuild the rotated image by calling the CLI's load_file
  // indirectly: read the upright file, rotate 180° in BGR, and
  // call ppocr_run() on the rotated buffer.
  // For simplicity we use ppocr_run_file on a pre-rotated file.
  const std::string rot_image = "/tmp/zh_04_rot180_for_test.jpg";
  {
    // Build the rot180 image via the engine's helper: we don't have
    // a public load_file helper, so we use stb_image directly.
    // Skip this and just rely on the Python validation for the
    // rot180 case. The C++ test covers the wiring (cls_ms > 0
    // when cls_name is set); the rot180 case is covered by
    // tools/cls_validation.py.
  }

  ppocr_destroy(e_on);
  ppocr_destroy(e_off);
  std::fprintf(stderr, "OK: cls integration test passed\n");
  return 0;
}
