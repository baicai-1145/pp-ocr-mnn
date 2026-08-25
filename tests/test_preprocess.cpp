// pp-ocr-mnn — preprocessing unit tests (M1)
//
// Plain-assert tests; no framework. Run after build with:
//   ./build-m1/tests/test_preprocess
// Exit 0 on success, non-zero on first failure.
//
// Coverage:
//   1. det type 0 (LimitMin) on a 1280x720 image → expected H/W from
//      ratio = 736/min(1280,720) = 1.0222..., rounded to 32.
//   2. det type 2 (ResizeLong) on the same image → ratio = 960/max(...),
//      rounded to 128 (v4/v5 stride).
//   3. rec keep-ratio: 200x80 source with h=48 w=320 →
//        ratio = 48/80 = 0.6
//        w' = round(200 * 0.6) = 120 → valid_w = 120
//      plus a sample of padded pixels (column 200..319 should be 0
//      pre-normalization, which means -1.0 in CHW float).
//   4. rec normalization: a black image in BGR should produce -1.0
//      everywhere; a white image should produce +1.0 everywhere.
//   5. config JSON loader: parses a hand-written v6 det config and a
//      rec config with a small dict, asserting every field round-trips.
#include "ppocr/config.h"
#include "ppocr/preprocess.h"

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

ppocr::Image make_solid(int w, int h, uint8_t r, uint8_t g, uint8_t b) {
  ppocr::Image img;
  img.w = w; img.h = h; img.c = 3;
  img.data.assign(static_cast<size_t>(w) * h * 3, 0);
  for (int i = 0; i < w * h; ++i) {
    img.data[3 * i + 0] = b;
    img.data[3 * i + 1] = g;
    img.data[3 * i + 2] = r;
  }
  return img;
}

void test_det_type0() {
  // 1280x720, LimitMin(736), stride 32 → ratio=736/720≈1.0222,
  // resize_w=round(1280*1.0222)=1308, resize_h=round(720*1.0222)=736,
  // snapped to 32: 1312 x 736.
  ppocr::Image img = make_solid(1280, 720, 128, 128, 128);
  ppocr::DetResizeConfig rc;
  rc.mode = ppocr::DetResizeConfig::Mode::LimitMin;
  rc.limit_side_len = 736;
  rc.stride = 32;
  rc.max_side_limit = 4000;
  ppocr::DetInput in = ppocr::prep_det(img, rc);
  std::fprintf(stderr, "[det type0] %dx%d -> %dx%d  ratio_w=%.4f ratio_h=%.4f\n",
               img.w, img.h, in.in_w, in.in_h, in.ratio_w, in.ratio_h);
  CHECK(in.in_w == 1312);
  CHECK(in.in_h == 736);
  CHECK(in.chw.size() == static_cast<size_t>(3) * in.in_w * in.in_h);
  // ratio_w ≈ 1312/1280 = 1.025; ratio_h = 736/720 ≈ 1.0222.
  CHECK(std::fabs(in.ratio_w - 1312.f / 1280.f) < 1e-4f);
  CHECK(std::fabs(in.ratio_h - 736.f / 720.f)  < 1e-4f);
}

void test_det_type2() {
  // 1280x720, ResizeLong(960), stride 128 → ratio=960/1280=0.75,
  // resize_w=960, resize_h=540, snapped up to 128-multiples:
  //   960 /128=7.5 → 1024
  //   540 /128=4.21875 → 640
  ppocr::Image img = make_solid(1280, 720, 64, 64, 64);
  ppocr::DetResizeConfig rc;
  rc.mode = ppocr::DetResizeConfig::Mode::ResizeLong;
  rc.resize_long = 960;
  rc.stride = 128;
  rc.max_side_limit = 4000;
  ppocr::DetInput in = ppocr::prep_det(img, rc);
  std::fprintf(stderr, "[det type2] %dx%d -> %dx%d  ratio_w=%.4f ratio_h=%.4f\n",
               img.w, img.h, in.in_w, in.in_h, in.ratio_w, in.ratio_h);
  CHECK(in.in_w == 1024);
  CHECK(in.in_h == 640);
  CHECK(in.chw.size() == static_cast<size_t>(3) * in.in_w * in.in_h);
}

void test_det_channel_order() {
  // PaddleOCR DetPreProcess keeps BGR channel order — mean[0] is
  // subtracted from B (0.485), mean[2] from R (0.406). Verifies the
  // M1 fix: do not flip mean/std across the channel axis.
  // Pure blue pixel (B=255, G=0, R=0):
  //   B: (1.0 - 0.485) / 0.229 = 2.249
  //   G: (0.0 - 0.456) / 0.224 = -2.0357
  //   R: (0.0 - 0.406) / 0.225 = -1.804
  ppocr::Image blue = make_solid(8, 8, /*R*/0, /*G*/0, /*B*/255);
  ppocr::DetResizeConfig rc;
  rc.mode = ppocr::DetResizeConfig::Mode::LimitMin;
  rc.limit_side_len = 736; rc.stride = 32;
  ppocr::DetInput in = ppocr::prep_det(blue, rc);
  const int plane = in.in_w * in.in_h;
  const float b = in.chw[0 * plane + 0];
  const float g = in.chw[1 * plane + 0];
  const float r = in.chw[2 * plane + 0];
  std::fprintf(stderr, "[det channel] BGR(255,0,0) -> b=%.4f g=%.4f r=%.4f\n",
               b, g, r);
  CHECK(std::fabs(b - (1.0f - 0.485f) / 0.229f) < 1e-3f);
  CHECK(std::fabs(g - (0.0f - 0.456f) / 0.224f) < 1e-3f);
  CHECK(std::fabs(r - (0.0f - 0.406f) / 0.225f) < 1e-3f);
}

void test_rec_keep_ratio_and_pad() {
  // 200x80 → ratio 48/80 = 0.6 → w' = round(120) = 120; padded to 320.
  ppocr::Image img = make_solid(200, 80, 255, 255, 255);
  int valid_w = -1;
  std::vector<float> chw = ppocr::prep_rec_line(img, 48, 320, valid_w);
  std::fprintf(stderr, "[rec 200x80] valid_w=%d chw=%zu floats\n",
               valid_w, chw.size());
  CHECK(valid_w == 120);
  CHECK(chw.size() == static_cast<size_t>(3) * 48 * 320);

  // Padded area (column >= 120) should normalize to -1 (zero in source).
  // The (channel, y, x) linear index for CHW is c*H*W + y*W + x.
  const int H = 48, W = 320;
  for (int c = 0; c < 3; ++c) {
    // Sample one cell deep in the padded region: y=10, x=300.
    float v = chw[c * H * W + 10 * W + 300];
    CHECK(std::fabs(v - (-1.f)) < 1e-3f);
  }
  // Unpadded interior (y=10, x=50) for a white image: (1 - 0.5)/0.5 = 1.0
  for (int c = 0; c < 3; ++c) {
    float v = chw[c * H * W + 10 * W + 50];
    CHECK(std::fabs(v - 1.f) < 1e-3f);
  }
}

void test_rec_normalization() {
  // Black image → -1; white image → +1; mid-gray (128) → ≈ 0.0039
  // (since (0.5019 - 0.5)/0.5 = 0.0039).
  ppocr::Image black = make_solid(50, 50, 0, 0, 0);
  int vw = -1;
  std::vector<float> chw = ppocr::prep_rec_line(black, 48, 320, vw);
  CHECK(vw > 0 && vw <= 320);
  for (int c = 0; c < 3; ++c) {
    float v = chw[c * 48 * 320 + 0];
    CHECK(std::fabs(v - (-1.f)) < 1e-3f);
  }
  ppocr::Image white = make_solid(50, 50, 255, 255, 255);
  chw = ppocr::prep_rec_line(white, 48, 320, vw);
  for (int c = 0; c < 3; ++c) {
    float v = chw[c * 48 * 320 + 0];
    CHECK(std::fabs(v - 1.f) < 1e-3f);
  }
}

void test_config_json_parse() {
  const std::string path = "test_config.json";
  {
    std::ofstream f(path, std::ios::binary);
    f << R"({
      "name": "PP-OCRv6_tiny_det",
      "type": "det",
      "file": "PP-OCRv6_tiny_det.mnn",
      "sha256": "deadbeef",
      "bytes": 1234,
      "url": "PP-OCRv6_tiny_det.mnn",
      "det": {
        "thresh": 0.2,
        "box_thresh": 0.4,
        "unclip_ratio": 1.4,
        "max_candidates": 3000,
        "resize": {
          "mode": "limit_min",
          "limit_side_len": 736,
          "stride": 32,
          "max_side_limit": 4000
        }
      }
    })";
  }
  ppocr::ModelConfig cfg = ppocr::load_model_config(path);
  std::remove(path.c_str());
  CHECK(cfg.name == "PP-OCRv6_tiny_det");
  CHECK(cfg.type == "det");
  CHECK(cfg.file == "PP-OCRv6_tiny_det.mnn");
  CHECK(cfg.sha256 == "deadbeef");
  CHECK(cfg.bytes == 1234u);
  CHECK(cfg.url == "PP-OCRv6_tiny_det.mnn");
  CHECK(std::fabs(cfg.det.thresh       - 0.2f) < 1e-5f);
  CHECK(std::fabs(cfg.det.box_thresh   - 0.4f) < 1e-5f);
  CHECK(std::fabs(cfg.det.unclip_ratio - 1.4f) < 1e-5f);
  CHECK(cfg.det.max_candidates == 3000);
  CHECK(cfg.det.resize.mode == ppocr::DetResizeConfig::Mode::LimitMin);
  CHECK(cfg.det.resize.limit_side_len == 736);
  CHECK(cfg.det.resize.stride == 32);
  CHECK(cfg.det.resize.max_side_limit == 4000);
}

void test_config_json_parse_rec() {
  const std::string path = "test_rec.json";
  {
    std::ofstream f(path, std::ios::binary);
    f << R"({
      "name": "PP-OCRv6_tiny_rec",
      "type": "rec",
      "file": "PP-OCRv6_tiny_rec.mnn",
      "rec": {
        "shape": [3, 48, 320],
        "use_space": true,
        "dict": ["a", "b", "c", "你好"]
      }
    })";
  }
  ppocr::ModelConfig cfg = ppocr::load_model_config(path);
  std::remove(path.c_str());
  CHECK(cfg.type == "rec");
  CHECK(cfg.rec.c == 3 && cfg.rec.h == 48 && cfg.rec.w == 320);
  CHECK(cfg.rec.use_space == true);
  CHECK(cfg.rec.dict.size() == 4);
  CHECK(cfg.rec.dict[0] == "a");
  CHECK(cfg.rec.dict[3] == "你好");
}

} // namespace

// PaddleOCR NumPy round() compatibility: half-to-even (banker's
// rounding). The 32 px discrepancy in M2-PIPE (1280x720 -> 1280x704
// in baseline, 1280x736 in our M1 path) was caused by the previous
// ceil-based alignment. These cases pin the new contract.
void test_stride_align() {
  using ppocr::round_up_to_stride;
  // Reference: numpy.round() in Python uses banker's rounding
  // (FE_TONEAREST, half-to-even) on IEEE-754 doubles. std::nearbyint
  // does the same on the host C++ side. We pin every half-value
  // boundary that arises during a normal 32-align det resize so
  // future drift is caught at test time.
  // Reference: numpy.round() in Python uses banker's rounding
  // (FE_TONEAREST, half-to-even) on IEEE-754 doubles. std::nearbyint
  // does the same on the host C++ side. We pin every half-value
  // boundary that arises during a normal 32-align det resize so
  // future drift is caught at test time.
  CHECK(round_up_to_stride(720, 32) == 704);  // 22.5 -> 22 -> 704
  CHECK(round_up_to_stride(736, 32) == 736);  // 23   -> 23 -> 736
  CHECK(round_up_to_stride(944, 32) == 960);  // 29.5 -> 30 -> 960
  CHECK(round_up_to_stride(976, 32) == 960);  // 30.5 -> 30 -> 960
  CHECK(round_up_to_stride(992, 32) == 992);  // 31   -> 31 -> 992
  CHECK(round_up_to_stride(0,   32) == 32);   // 0    -> 0  -> clamp to 32
  CHECK(round_up_to_stride(16,  32) == 32);   // 0.5  -> 0  -> clamp to 32
  CHECK(round_up_to_stride(48,  32) == 64);   // 1.5  -> 2  -> 64  (half-to-even)
  CHECK(round_up_to_stride(64,  32) == 64);   // 2    -> 2  -> 64
  CHECK(round_up_to_stride(704, 32) == 704);  // 22   -> 22 -> 704
  CHECK(round_up_to_stride(736, 32) == 736);  // 23   -> 23 -> 736
  // 128-align sanity (used by seal det):
  CHECK(round_up_to_stride(720, 128) == 768);  // 5.625 -> 6 -> 768
  CHECK(round_up_to_stride(768, 128) == 768);  // 6 -> 6 -> 768
  CHECK(round_up_to_stride(960, 128) == 1024); // 7.5 -> 8 -> 1024 (half-to-even)
  CHECK(round_up_to_stride(896, 128) == 896);  // 7 -> 7 -> 896
}

int main() {
  test_det_type0();
  test_det_type2();
  test_det_channel_order();
  test_rec_keep_ratio_and_pad();
  test_rec_normalization();
  test_config_json_parse();
  test_config_json_parse_rec();
  test_stride_align();
  std::fprintf(stderr, "all preprocess tests passed\n");
  return 0;
}
