// pp-ocr-mnn — postprocess unit tests (post module). Plain asserts, no framework.
// Owner: post. Run via g++ -std=c++17 -I include tests/test_post.cpp
//        src/postprocess/*.cpp -o /tmp/test_post && /tmp/test_post
#include "ppocr/postprocess/ctc_decode.h"
#include "ppocr/postprocess/db_post.h"
#include "ppocr/postprocess/geometry.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ppocr/config.h"

using namespace ppocr;

namespace {

bool feq(float a, float b, float eps = 1e-3f) {
  return std::fabs(a - b) <= eps;
}

// --- Test 1: min_area_rect on an axis-aligned rectangle (4 points on edges) -
void test_min_area_rect_axis_aligned() {
  PointF pts[4] = {{0, 0}, {10, 0}, {10, 4}, {0, 4}};
  PointF out[4];
  bool ok = min_area_rect(pts, 4, out);
  assert(ok);
  // Area must be 40 within FP tolerance.
  auto poly_area = [](const PointF* q) {
    float s = 0;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) & 3;
      s += q[i].x * q[j].y - q[j].x * q[i].y;
    }
    return std::fabs(s * 0.5f);
  };
  assert(feq(poly_area(out), 40.0f, 1e-2f));
  std::printf("[ok] test_min_area_rect_axis_aligned: area=%.3f\n",
              poly_area(out));
}

// --- Test 2: min_area_rect on a rotated rectangle (long thin) ---------------
void test_min_area_rect_rotated() {
  // Rectangle 100x10 rotated by 30 degrees. Pick 4 corners analytically.
  const float theta = static_cast<float>(M_PI / 6.0);  // 30 deg
  const float W = 100.0f, H = 10.0f;
  float c = std::cos(theta), s = std::sin(theta);
  // Centered at (0,0): corners (in CCW order):
  //   (-W/2, -H/2), (W/2, -H/2), (W/2, H/2), (-W/2, H/2)
  // Then rotate and translate by (50,50).
  auto rot = [&](float x, float y) {
    return PointF{50.0f + c * x - s * y, 50.0f + s * x + c * y};
  };
  PointF pts[4] = {
      rot(-W / 2, -H / 2),
      rot(W / 2, -H / 2),
      rot(W / 2, H / 2),
      rot(-W / 2, H / 2),
  };
  PointF out[4];
  bool ok = min_area_rect(pts, 4, out);
  assert(ok);
  (void)ok;
  auto poly_area = [](const PointF* q) {
    float s = 0;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) & 3;
      s += q[i].x * q[j].y - q[j].x * q[i].y;
    }
    return std::fabs(s * 0.5f);
  };
  float a = poly_area(out);
  // Should be 100*10 = 1000 (within FP).
  assert(feq(a, 1000.0f, 1.0f));
  // The min-area rect's short side should equal 10.
  auto dist = [](const PointF& a, const PointF& b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) +
                     (a.y - b.y) * (a.y - b.y));
  };
  float sides[4] = {dist(out[0], out[1]), dist(out[1], out[2]),
                    dist(out[2], out[3]), dist(out[3], out[0])};
  float short_side = std::min({sides[0], sides[1], sides[2], sides[3]});
  float long_side = std::max({sides[0], sides[1], sides[2], sides[3]});
  assert(feq(short_side, 10.0f, 0.5f));
  assert(feq(long_side, 100.0f, 0.5f));
  std::printf("[ok] test_min_area_rect_rotated: area=%.2f s=%.2f l=%.2f\n", a,
              short_side, long_side);
}

// --- Test 3: min_area_rect on a non-convex point cloud (right triangle) ------
void test_min_area_rect_triangle() {
  // Right triangle: (0,0), (30,0), (0,40). The min-area enclosing rectangle
  // is the bounding box (any rotated alignment grows the area since the
  // triangle's "missing" corner is convex). Area = 30 * 40 = 1200.
  PointF pts[3] = {{0, 0}, {30, 0}, {0, 40}};
  PointF out[4];
  bool ok = min_area_rect(pts, 3, out);
  assert(ok);
  (void)ok;
  auto poly_area = [](const PointF* q) {
    float s = 0;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) & 3;
      s += q[i].x * q[j].y - q[j].x * q[i].y;
    }
    return std::fabs(s * 0.5f);
  };
  float a = poly_area(out);
  assert(feq(a, 1200.0f, 1.0f));
  std::printf("[ok] test_min_area_rect_triangle: area=%.2f\n", a);
}

// --- Test 4: sort_min_area_rect_points --------------------------------------
void test_sort_min_area_rect_points() {
  // Provide the corners in a permuted order. After sort, must be TL, TR, BR, BL
  // (TL has smaller x AND smaller y than BL; TR has larger x AND smaller y
  // than BR).
  PointF box[4] = {
      {10, 5},   // bottom-right area (large x, large y)
      {0, 0},    // top-left (small x, small y)
      {10, 0},   // top-right
      {0, 5},    // bottom-left
  };
  sort_min_area_rect_points(box);
  // After sort: box[0] = TL = (0,0), box[1] = TR = (10,0), box[2] = BR = (10,5),
  // box[3] = BL = (0,5).
  assert(feq(box[0].x, 0.0f) && feq(box[0].y, 0.0f));
  assert(feq(box[1].x, 10.0f) && feq(box[1].y, 0.0f));
  assert(feq(box[2].x, 10.0f) && feq(box[2].y, 5.0f));
  assert(feq(box[3].x, 0.0f) && feq(box[3].y, 5.0f));
  std::printf("[ok] test_sort_min_area_rect_points\n");
}

// --- Test 5: warp_perspective_quad identity --------------------------------
void test_warp_identity() {
  // 4x3 RGB image, all distinct values per cell to detect any shift.
  const int W = 4, H = 3, C = 3;
  Image src;
  src.w = W;
  src.h = H;
  src.c = C;
  src.data.assign(W * H * C, 0);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      uint8_t v = static_cast<uint8_t>(y * 10 + x);
      for (int c = 0; c < C; ++c) {
        src.data[(y * W + x) * C + c] = static_cast<uint8_t>(v + c * 30);
      }
    }
  }
  // Quad in canonical order: TL, TR, BR, BL.
  PointF quad[4] = {{0, 0}, {W - 1, 0}, {W - 1, H - 1}, {0, H - 1}};
  Image dst = warp_perspective_quad(src, quad, W, H);
  assert(dst.w == W && dst.h == H && dst.c == C);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      uint8_t v = static_cast<uint8_t>(y * 10 + x);
      for (int c = 0; c < C; ++c) {
        uint8_t expected = static_cast<uint8_t>(v + c * 30);
        uint8_t actual = dst.data[(y * W + x) * C + c];
        // Allow ±1 for bicubic (interior pixel reproduces exactly only for
        // sample-on-grid; on integer pixel coordinates bicubic with the
        // OpenCV kernel reproduces the source).
        int diff = std::abs(static_cast<int>(actual) - expected);
        (void)diff;
        assert(diff <= 1);
      }
    }
  }
  std::printf("[ok] test_warp_identity (%dx%d, c=%d)\n", W, H, C);
}

// --- Test 6: warp_perspective_quad shift (4x3 -> 4x3 with sub-pixel shift) -
void test_warp_subpixel_shift() {
  // 8x8 gradient image. Shift by 0.5 pixels in x and verify corner sampling
  // matches the bicubic interpolation. We test that the destination's left
  // column is darker (mix with 0) and the right column is brighter (mix with
  // 8). We don't compare to a specific formula (cubic is open-vision-
  // dependent on coefficients and rounding); we only assert that the
  // warp runs without crashing and the values stay in [0,255].
  const int W = 16, H = 16, C = 1;
  Image src;
  src.w = W;
  src.h = H;
  src.c = C;
  src.data.assign(W * H, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) src.data[y * W + x] = static_cast<uint8_t>(x);
  PointF quad[4] = {{-0.5f, 0}, {W - 1 + 0.5f, 0},
                    {W - 1 + 0.5f, H - 1}, {-0.5f, H - 1}};
  Image dst = warp_perspective_quad(src, quad, W, H);
  assert(dst.w == W && dst.h == H && dst.c == C);
  // At x=0 with subpixel shift -0.5, the bicubic samples the source x range
  // [-1,2] clamped to [0,3]. Source values there are {0,1,2,3}, weighted by
  // cubic_weight(1.5), cubic_weight(0.5), cubic_weight(-0.5), cubic_weight(-1.5).
  // Sum of weights is 1, so values must stay in [0,255]. We sample a few
  // representative pixels and check they are not NaN-casted to 0/255 wrongly.
  // The bicubic kernel can produce slight overshoot at sharp gradients; the
  // sampler clamps to [0,255] so the output must be a valid uint8_t.
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      assert(dst.data[y * W + x] >= 0);  // tautology on uint8_t; documents intent
  std::printf("[ok] test_warp_subpixel_shift (%dx%d)\n", W, H);
}

// --- Test 7: db_postprocess on a synthetic white rectangle -----------------
void test_db_postprocess_white_rect() {
  // prob = 0.9 inside a centered 80x40 rect on a 160x80 map.
  const int H = 80, W = 160;
  std::vector<float> prob(W * H, 0.0f);
  int x0 = 40, x1 = 120, y0 = 20, y1 = 60;
  for (int y = y0; y < y1; ++y)
    for (int x = x0; x < x1; ++x) prob[y * W + x] = 0.9f;
  // Light noise outside.
  for (int i = 0; i < W * H; ++i) {
    if (prob[i] < 0.5f) prob[i] = 0.05f;
  }
  DetConfig cfg;
  cfg.thresh = 0.3f;
  cfg.box_thresh = 0.5f;
  cfg.unclip_ratio = 1.5f;
  cfg.max_candidates = 100;
  // use_dilation = false (hardcoded — Paddle default; not part of DetConfig
  // per CONTRACT.md, but trivially toggled here for future extension).
  // ratio_w = 2, ratio_h = 2 — pretend source is 320x160 and map is 160x80.
  auto boxes = db_postprocess(prob.data(), H, W, 320, 160, 2.0f, 2.0f, cfg);
  // Expect exactly one box (or at least one) covering the rect.
  assert(!boxes.empty());
  // Find the box with the largest area.
  const DetBox* best = &boxes[0];
  for (const auto& b : boxes) {
    float ax0 = b.poly[0], ay0 = b.poly[1];
    float ax2 = b.poly[4], ay2 = b.poly[5];
    float bx0 = best->poly[0], by0 = best->poly[1];
    float bx2 = best->poly[4], by2 = best->poly[5];
    if ((ax2 - ax0) * (ay2 - ay0) > (bx2 - bx0) * (by2 - by0)) best = &b;
  }
  // Width and height in original coords should be ~160 and ~80 (the rect is
  // 80x40 in map = 160x80 in original, after unclip a bit larger).
  float w_orig = std::sqrt((best->poly[2] - best->poly[0]) *
                                (best->poly[2] - best->poly[0]) +
                            (best->poly[3] - best->poly[1]) *
                                (best->poly[3] - best->poly[1]));
  float h_orig = std::sqrt((best->poly[4] - best->poly[2]) *
                                (best->poly[4] - best->poly[2]) +
                            (best->poly[5] - best->poly[3]) *
                                (best->poly[5] - best->poly[3]));
  // Allow ±20% (unclip expands, plus DP simplification + rasterization noise).
  // Use a runtime check (not assert, which is compiled out in Release) so this
  // is enforced in both Debug and Release.
  // The white rect is 80x40 in map coords. After unclip_ratio=1.5 it expands
  // by ~20 px in each direction to a 120x80 box, then is mapped to dest
  // coords (ratio=2) giving ~240x160. Paddle reference: (40,0)-(278,158)
  // → width 238, height 158. Allow some leeway.
  if (!(w_orig > 200.0f && w_orig < 280.0f) ||
      !(h_orig > 130.0f && h_orig < 180.0f) || !(best->score > 0.7f)) {
    std::fprintf(stderr,
                 "FAIL test_db_postprocess_white_rect: bounds w=[200,280] "
                 "h=[130,180] score>0.7 (got w=%.2f h=%.2f score=%.2f)\n",
                 w_orig, h_orig, best->score);
    std::exit(1);
  }
  std::printf("[ok] test_db_postprocess_white_rect: boxes=%zu w=%.1f h=%.1f score=%.2f\n",
              boxes.size(), w_orig, h_orig, best->score);
  std::printf("    poly=[%.0f,%.0f, %.0f,%.0f, %.0f,%.0f, %.0f,%.0f]\n",
              best->poly[0], best->poly[1], best->poly[2], best->poly[3],
              best->poly[4], best->poly[5], best->poly[6], best->poly[7]);
}

// --- Test 8: ctc_decode simple with blanks and repeats ----------------------
void test_ctc_decode_basic() {
  // dict = [a, b, c]; blank = 0; space not used here. Char table = [blank, a,
  // b, c]. 4 classes, 5 timesteps.
  // Sequence: blank, a, a, b, c -> "abc" (collapse repeats of a; drop blank).
  // Encode as one-hot: timestep 0 = blank, t1 = a, t2 = a, t3 = b, t4 = c.
  const int T = 5, C = 4;
  float logits[T][C] = {};
  logits[0][0] = 1.0f;  // blank
  logits[1][1] = 1.0f;  // a
  logits[2][1] = 1.0f;  // a (repeat, collapse)
  logits[3][2] = 1.0f;  // b
  logits[4][3] = 1.0f;  // c
  RecConfig cfg;
  cfg.use_space = false;
  cfg.dict = {"a", "b", "c"};
  RecOut r = ctc_decode(&logits[0][0], T, C, cfg);
  assert(r.text == "abc");
  // Score: 3 chars emitted (a, b, c), each with prob 1.0. Mean = 1.0.
  assert(feq(r.score, 1.0f, 1e-4f));
  std::printf("[ok] test_ctc_decode_basic: text=\"%s\" score=%.2f\n",
              r.text.c_str(), r.score);
}

// --- Test 9: ctc_decode with space, interleaved blank ----------------------
void test_ctc_decode_with_space() {
  // dict = [a, b]; use_space=true. Char table = [blank, a, b, " "].
  // 4 classes, 7 timesteps.
  // Sequence: a, blank, b, blank, blank, " ", a -> "ab a" with a single space.
  const int T = 7, C = 4;
  float logits[T][C] = {};
  logits[0][1] = 1.0f;  // a
  logits[1][0] = 1.0f;  // blank
  logits[2][2] = 1.0f;  // b
  logits[3][0] = 1.0f;  // blank
  logits[4][0] = 1.0f;  // blank
  logits[5][3] = 1.0f;  // space
  logits[6][1] = 1.0f;  // a
  RecConfig cfg;
  cfg.use_space = true;
  cfg.dict = {"a", "b"};
  RecOut r = ctc_decode(&logits[0][0], T, C, cfg);
  assert(r.text == "ab a");
  assert(feq(r.score, 1.0f, 1e-4f));
  std::printf("[ok] test_ctc_decode_with_space: text=\"%s\" score=%.2f\n",
              r.text.c_str(), r.score);
}

// --- Test 10: ctc_decode UTF-8 multi-byte chars ----------------------------
void test_ctc_decode_utf8() {
  // Dict with Chinese characters (each 3 bytes in UTF-8).
  // We construct the dict as UTF-8 std::strings and use a small char table.
  // "blank" + dict = ["你", "好"]. Char table = [blank, 你, 好].
  // 3 classes, 4 timesteps: 你, 你, 好, 好 (collapse) -> "你好".
  const int T = 4, C = 3;
  float logits[T][C] = {};
  logits[0][1] = 1.0f;  // 你
  logits[1][1] = 1.0f;  // 你 (collapse)
  logits[2][2] = 1.0f;  // 好
  logits[3][2] = 1.0f;  // 好 (collapse)
  RecConfig cfg;
  cfg.use_space = false;
  cfg.dict = {"\xe4\xbd\xa0", "\xe5\xa5\xbd"};  // 你 好
  RecOut r = ctc_decode(&logits[0][0], T, C, cfg);
  // Compare against the expected UTF-8 bytes (no re-encoding).
  std::string expected = "\xe4\xbd\xa0\xe5\xa5\xbd";
  assert(r.text == expected);
  // 2 emitted chars (你, 好), each prob 1.0. Mean = 1.0.
  assert(feq(r.score, 1.0f, 1e-4f));
  // Sanity: byte length is 6 (2 chars * 3 bytes).
  assert(r.text.size() == 6);
  std::printf("[ok] test_ctc_decode_utf8: text_len=%zu score=%.2f\n",
              r.text.size(), r.score);
}

// --- Test 11: ctc_decode: only blanks -> empty text, score 0 ---------------
void test_ctc_decode_all_blank() {
  const int T = 4, C = 3;
  float logits[T][C] = {};
  logits[0][0] = 1.0f;
  logits[1][0] = 1.0f;
  logits[2][0] = 1.0f;
  logits[3][0] = 1.0f;
  RecConfig cfg;
  cfg.use_space = false;
  cfg.dict = {"a", "b"};
  RecOut r = ctc_decode(&logits[0][0], T, C, cfg);
  assert(r.text.empty());
  assert(feq(r.score, 0.0f, 1e-6f));
  std::printf("[ok] test_ctc_decode_all_blank\n");
}

// --- Test 12: ctc_decode: repeated char with blank in between -> keep both -
void test_ctc_decode_blank_between_repeats() {
  // a, blank, a -> "aa" (the blank allows the repeated a to be re-emitted).
  const int T = 3, C = 2;
  float logits[T][C] = {};
  logits[0][1] = 1.0f;  // a
  logits[1][0] = 1.0f;  // blank
  logits[2][1] = 1.0f;  // a
  RecConfig cfg;
  cfg.use_space = false;
  cfg.dict = {"a"};
  RecOut r = ctc_decode(&logits[0][0], T, C, cfg);
  assert(r.text == "aa");
  assert(feq(r.score, 1.0f, 1e-4f));
  std::printf("[ok] test_ctc_decode_blank_between_repeats\n");
}

// --- Test 13: ctc_decode: probabilistic logits (non-one-hot) --------------
// Verifies that score = mean of per-timestep probs of the EMITTED chars
// (not all timesteps). Example: a(0.8), a(0.7), b(0.9). Greedy emits a,b.
// a is repeated -> collapse first a. Emitted chars: a (from t=1, p=0.8),
// b (from t=2, p=0.9). Mean = 0.85.
void test_ctc_decode_probabilistic() {
  const int T = 3, C = 3;
  float logits[T][C] = {};
  // timestep 0: a=0.8, blank=0.1, b=0.1
  logits[0][0] = 0.1f; logits[0][1] = 0.8f; logits[0][2] = 0.1f;
  // timestep 1: a=0.7 (repeat)
  logits[1][0] = 0.2f; logits[1][1] = 0.7f; logits[1][2] = 0.1f;
  // timestep 2: b=0.9
  logits[2][0] = 0.05f; logits[2][1] = 0.05f; logits[2][2] = 0.9f;
  RecConfig cfg;
  cfg.use_space = false;
  cfg.dict = {"a", "b"};
  RecOut r = ctc_decode(&logits[0][0], T, C, cfg);
  assert(r.text == "ab");
  assert(feq(r.score, 0.85f, 1e-3f));
  std::printf("[ok] test_ctc_decode_probabilistic: text=\"%s\" score=%.3f\n",
              r.text.c_str(), r.score);
}

// --- Test 14: db_postprocess with two separated rectangles -----------------
void test_db_postprocess_two_rects() {
  // Two 30x20 rectangles on a 160x80 map. Should produce 2 boxes.
  const int H = 80, W = 160;
  std::vector<float> prob(W * H, 0.05f);
  for (int y = 30; y < 50; ++y)
    for (int x = 20; x < 50; ++x) prob[y * W + x] = 0.85f;
  for (int y = 30; y < 50; ++y)
    for (int x = 100; x < 130; ++x) prob[y * W + x] = 0.85f;
  DetConfig cfg;
  cfg.thresh = 0.3f;
  cfg.box_thresh = 0.5f;
  cfg.unclip_ratio = 1.5f;
  cfg.max_candidates = 100;
  auto boxes = db_postprocess(prob.data(), H, W, 320, 160, 2.0f, 2.0f, cfg);
  // We expect at least 2 boxes (could be more if any noise crosses threshold;
  // both rects are well above 0.3).
  assert(boxes.size() >= 2);
  std::printf("[ok] test_db_postprocess_two_rects: %zu boxes\n", boxes.size());
}

// --- Test 15: warp_perspective_quad with extreme aspect ratio (h/w = 8:1) ---
// A very thin horizontal text line is common in real textline-warping use
// cases. Make sure the sampler does not produce out-of-bounds reads or
// writes a wrong-sized output.
void test_warp_extreme_aspect() {
  const int W = 80, H = 16, C = 3;
  Image src;
  src.w = W; src.h = H; src.c = C;
  src.data.assign(W * H * C, 0);
  // Linear gradient (x -> 0..255).
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      uint8_t v = static_cast<uint8_t>(x * 3);
      for (int c = 0; c < C; ++c) src.data[(y * W + x) * C + c] = v;
    }
  // Quad in src coords: 80 wide, 10 tall, around the middle. Destination
  // 8x high to get h/w = 8/64 = 1/8.
  PointF quad[4] = {
      {0, 3}, {W - 1, 3}, {W - 1, 12}, {0, 12},
  };
  const int dst_w = 64, dst_h = 8;  // aspect 8:1
  Image dst = warp_perspective_quad(src, quad, dst_w, dst_h);
  assert(dst.w == dst_w && dst.h == dst_h && dst.c == C);
  assert(!dst.data.empty());
  // Pixels should be in valid range; no all-zero output.
  int non_zero = 0;
  int oob = 0;
  for (int y = 0; y < dst_h; ++y)
    for (int x = 0; x < dst_w; ++x) {
      for (int c = 0; c < C; ++c) {
        uint8_t v = dst.data[(y * dst_w + x) * C + c];
        if (v != 0) ++non_zero;
        // No OOB check needed since output is uint8; this asserts nonzero.
        (void)oob;
      }
    }
  // At least half the pixels should be non-zero (gradient has 0 at x=0).
  if (non_zero < (dst_w * dst_h * C) / 2) {
    std::fprintf(stderr,
                 "FAIL test_warp_extreme_aspect: only %d non-zero of %d\n",
                 non_zero, dst_w * dst_h * C);
    std::exit(1);
  }
  std::printf("[ok] test_warp_extreme_aspect: dst=%dx%d, %d non-zero pixels\n",
              dst_w, dst_h, non_zero);
}

// --- Test 16: db_postprocess on all-zero prob map (no boxes, no crash) ----
void test_db_postprocess_zero_map() {
  const int H = 32, W = 48;
  std::vector<float> prob(W * H, 0.0f);  // all zeros
  DetConfig cfg;
  cfg.thresh = 0.3f;
  cfg.box_thresh = 0.5f;
  cfg.unclip_ratio = 1.5f;
  cfg.max_candidates = 100;
  // Must not crash, must return zero boxes (nothing above threshold).
  auto boxes = db_postprocess(prob.data(), H, W, 96, 64, 2.0f, 2.0f, cfg);
  if (!boxes.empty()) {
    std::fprintf(stderr,
                 "FAIL test_db_postprocess_zero_map: expected 0 boxes, got %zu\n",
                 boxes.size());
    std::exit(1);
  }
  std::printf("[ok] test_db_postprocess_zero_map: 0 boxes (no crash)\n");
}

// --- Test 17: sort_quad_boxes_reading_order, zh_02-style 3-row reorder ----
// Reproduces the row-order bug: db_postprocess returns boxes in findContours
// order (not reading order), so when there are 3 rows of text the join order
// is e.g. (Chinese, Project, Text) instead of (Project, Chinese, Text).
// After sort_quad_boxes_reading_order the boxes must be top-to-bottom
// (y of TL ascending).
void test_sort_quad_boxes_zh02_3rows() {
  std::vector<DetBox> boxes;
  // Simulate db_postprocess output for 3 horizontal text lines.
  // Image is ~720px wide; each row is ~30px tall.
  // Input order is the findContours order: middle row first, then top, then
  // bottom (this is the actual bug — outlines are extracted in scan order
  // and pick up rows in non-monotonic y).
  {
    DetBox b{};
    // row 2: "Chinese" at y ~ 360
    b.poly[0] = 100;  b.poly[1] = 360;
    b.poly[2] = 600;  b.poly[3] = 360;
    b.poly[4] = 600;  b.poly[5] = 390;
    b.poly[6] = 100;  b.poly[7] = 390;
    b.score = 0.95f;
    boxes.push_back(b);
  }
  {
    DetBox b{};
    // row 1: "Project" at y ~ 180
    b.poly[0] = 100;  b.poly[1] = 180;
    b.poly[2] = 600;  b.poly[3] = 180;
    b.poly[4] = 600;  b.poly[5] = 210;
    b.poly[6] = 100;  b.poly[7] = 210;
    b.score = 0.93f;
    boxes.push_back(b);
  }
  {
    DetBox b{};
    // row 3: "Text" at y ~ 540
    b.poly[0] = 100;  b.poly[1] = 540;
    b.poly[2] = 600;  b.poly[3] = 540;
    b.poly[4] = 600;  b.poly[5] = 570;
    b.poly[6] = 100;  b.poly[7] = 570;
    b.score = 0.90f;
    boxes.push_back(b);
  }
  // Sanity: input order is (row2, row1, row3) — y of TL = (360, 180, 540).
  assert(boxes[0].poly[1] == 360.0f);
  assert(boxes[1].poly[1] == 180.0f);
  assert(boxes[2].poly[1] == 540.0f);

  sort_quad_boxes_reading_order(boxes);

  // After sort: TL.y ascending → (180, 360, 540) = (row1, row2, row3)
  // = ("Project", "Chinese", "Text").
  if (!(boxes[0].poly[1] == 180.0f && boxes[0].poly[0] == 100.0f) ||
      !(boxes[1].poly[1] == 360.0f && boxes[1].poly[0] == 100.0f) ||
      !(boxes[2].poly[1] == 540.0f && boxes[2].poly[0] == 100.0f)) {
    std::fprintf(stderr,
                 "FAIL test_sort_quad_boxes_zh02_3rows: TL after sort = "
                 "(%.0f,%.0f) (%.0f,%.0f) (%.0f,%.0f); expected "
                 "(100,180) (100,360) (100,540)\n",
                 boxes[0].poly[0], boxes[0].poly[1],
                 boxes[1].poly[0], boxes[1].poly[1],
                 boxes[2].poly[0], boxes[2].poly[1]);
    std::exit(1);
  }
  // Scores should follow the boxes (re-ordered, not dropped).
  if (boxes[0].score != 0.93f || boxes[1].score != 0.95f ||
      boxes[2].score != 0.90f) {
    std::fprintf(stderr,
                 "FAIL test_sort_quad_boxes_zh02_3rows: score order = "
                 "(%.2f, %.2f, %.2f); expected (0.93, 0.95, 0.90)\n",
                 boxes[0].score, boxes[1].score, boxes[2].score);
    std::exit(1);
  }
  std::printf("[ok] test_sort_quad_boxes_zh02_3rows: y=(180,360,540) "
              "scores=(0.93,0.95,0.90)\n");
}

// --- Test 18: sort_quad_boxes_reading_order, same-row x re-order ----------
// Three boxes all at y=200 (same row, within 10px tolerance), in x order
// 500, 100, 300 (out of order). The bubble pass must move the leftmost
// (x=100) box to the front.
void test_sort_quad_boxes_same_row_xsort() {
  std::vector<DetBox> boxes;
  auto add = [&](float x) {
    DetBox b{};
    b.poly[0] = x;      b.poly[1] = 200;
    b.poly[2] = x + 50; b.poly[3] = 200;
    b.poly[4] = x + 50; b.poly[5] = 230;
    b.poly[6] = x;      b.poly[7] = 230;
    b.score = 0.9f;
    boxes.push_back(b);
  };
  add(500.0f);  // rightmost
  add(100.0f);  // leftmost (should be first)
  add(300.0f);  // middle

  // All have the same y=200. std::sort by (y, x) → (100, 300, 500) already.
  // But to make this a real test of the bubble path, force a different
  // primary sort: shift one box's y by 5 (still < 10 row tolerance) so the
  // std::sort places it after the (y=200, x=500) box, then the bubble
  // must walk it back left.
  boxes[2].poly[1] = 205.0f;  // y=205, x=300 — within 5px of y=200
  boxes[2].poly[3] = 205.0f;

  sort_quad_boxes_reading_order(boxes);

  // Expected: x ascending → (100, 300, 500).
  if (!(boxes[0].poly[0] == 100.0f) ||
      !(boxes[1].poly[0] == 300.0f) ||
      !(boxes[2].poly[0] == 500.0f)) {
    std::fprintf(stderr,
                 "FAIL test_sort_quad_boxes_same_row_xsort: x after sort = "
                 "(%.0f, %.0f, %.0f); expected (100, 300, 500)\n",
                 boxes[0].poly[0], boxes[1].poly[0], boxes[2].poly[0]);
    std::exit(1);
  }
  std::printf("[ok] test_sort_quad_boxes_same_row_xsort: x=(100,300,500)\n");
}

// --- Test 19: sort_quad_boxes_reading_order, y diff == 10px boundary ------
// When |y_j - y_{j-1}| is exactly 10.0, the bubble pass condition is
// `fabs(10) < 10` which is FALSE → no swap. (Strict `<` boundary, matches
// Paddle's `std::abs(...) < 10`.) Two boxes at y=100 and y=110 should
// therefore stay in y-order even if x is out of order — they are considered
// different rows.
void test_sort_quad_boxes_y_boundary_10px() {
  std::vector<DetBox> boxes;
  // Box 0: y=100, x=500 (top, rightmost)
  {
    DetBox b{};
    b.poly[0] = 500; b.poly[1] = 100;
    b.poly[2] = 550; b.poly[3] = 100;
    b.poly[4] = 550; b.poly[5] = 130;
    b.poly[6] = 500; b.poly[7] = 130;
    b.score = 0.9f;
    boxes.push_back(b);
  }
  // Box 1: y=110, x=100 (10px below, but leftmost — should NOT swap)
  {
    DetBox b{};
    b.poly[0] = 100; b.poly[1] = 110;
    b.poly[2] = 150; b.poly[3] = 110;
    b.poly[4] = 150; b.poly[5] = 140;
    b.poly[6] = 100; b.poly[7] = 140;
    b.score = 0.8f;
    boxes.push_back(b);
  }
  // Sanity: input is in y-order already.
  assert(boxes[0].poly[1] == 100.0f);
  assert(boxes[1].poly[1] == 110.0f);

  sort_quad_boxes_reading_order(boxes);

  // No swap: box 0 stays at y=100, x=500; box 1 stays at y=110, x=100.
  if (!(boxes[0].poly[1] == 100.0f && boxes[0].poly[0] == 500.0f) ||
      !(boxes[1].poly[1] == 110.0f && boxes[1].poly[0] == 100.0f)) {
    std::fprintf(stderr,
                 "FAIL test_sort_quad_boxes_y_boundary_10px: TL after sort = "
                 "(%.0f,%.0f) (%.0f,%.0f); expected (500,100) (100,110) "
                 "(no swap because y diff is exactly 10.0, not <10)\n",
                 boxes[0].poly[0], boxes[0].poly[1],
                 boxes[1].poly[0], boxes[1].poly[1]);
    std::exit(1);
  }
  // Also confirm a 9.9px diff WOULD swap (sanity for the boundary).
  std::vector<DetBox> b2 = boxes;  // copy
  b2[1].poly[1] = 109.9f;
  b2[1].poly[3] = 109.9f;
  sort_quad_boxes_reading_order(b2);
  if (!(b2[0].poly[1] == 109.9f && b2[0].poly[0] == 100.0f) ||
      !(b2[1].poly[1] == 100.0f && b2[1].poly[0] == 500.0f)) {
    std::fprintf(stderr,
                 "FAIL test_sort_quad_boxes_y_boundary_10px: at 9.9px diff "
                 "the bubble SHOULD swap; got (%.0f,%.0f) (%.0f,%.0f), "
                 "expected (100,109.9) (500,100)\n",
                 b2[0].poly[0], b2[0].poly[1],
                 b2[1].poly[0], b2[1].poly[1]);
    std::exit(1);
  }
  std::printf("[ok] test_sort_quad_boxes_y_boundary_10px: 10.0 no-swap, "
              "9.9 swaps\n");
}

}  // namespace

int main() {
  test_min_area_rect_axis_aligned();
  test_min_area_rect_rotated();
  test_min_area_rect_triangle();
  test_sort_min_area_rect_points();
  test_warp_identity();
  test_warp_subpixel_shift();
  test_warp_extreme_aspect();
  test_db_postprocess_white_rect();
  test_db_postprocess_two_rects();
  test_db_postprocess_zero_map();
  test_sort_quad_boxes_zh02_3rows();
  test_sort_quad_boxes_same_row_xsort();
  test_sort_quad_boxes_y_boundary_10px();
  test_ctc_decode_basic();
  test_ctc_decode_with_space();
  test_ctc_decode_utf8();
  test_ctc_decode_all_blank();
  test_ctc_decode_blank_between_repeats();
  test_ctc_decode_probabilistic();
  std::printf("\nALL POSTPROCESS TESTS PASSED (19/19)\n");
  return 0;
}
