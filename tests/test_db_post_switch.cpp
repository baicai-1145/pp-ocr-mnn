// pp-ocr-mnn — db_postprocess extractor-switch test.
//
// db_postprocess has two contour extractors, selected by PPOCR_SUZUKI:
//   default (unset/0) -> legacy CCL + hole-flood + Moore trace  (main's output)
//   PPOCR_SUZUKI=1    -> Suzuki-Abe single-pass border following
//
// This test pins the CONTRACT of the switch, not a golden output:
//
//   1. both extractors produce well-formed boxes (in-bounds, non-degenerate,
//      score in (0,1]) on the same input;
//   2. on CLEAN synthetic masks the two extractors agree EXACTLY.
//
// (2) is the interesting invariant, and it is not a tautology. The two paths
// use entirely different algorithms (union-find + wall-followed hole rings vs
// Suzuki-Abe border following) and a different candidate ordering, yet on
// noiseless masks they land on identical boxes. That agreement is evidence for
// the root-cause analysis behind defaulting Suzuki to OFF: the real-corpus
// divergence (PP-OCRv4_mobile_det/en 0.0121 PASS -> 0.0898 FAIL) comes from
// blob-level differences between the MNN and Paddle prob maps, NOT from the
// tracer logic -- which is also why 19/21 cells are bit-identical and why
// Suzuki recovers baseline boxes bit-for-bit.
//
// Because the divergence is prob-map driven, it cannot be reproduced from a
// synthetic mask and is therefore deliberately NOT asserted here. Byte-parity
// of the DEFAULT path against main is covered by tools/diff_detpolys.py on the
// real corpus, which is the authoritative gate.
#include "ppocr/postprocess/db_post.h"
#include "ppocr/config.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
  else     { std::printf("ok  : %s\n", what); }
}

ppocr::DetConfig make_cfg() {
  ppocr::DetConfig c;
  c.thresh = 0.3f; c.box_thresh = 0.6f; c.unclip_ratio = 1.5f;
  c.max_candidates = 1000; c.min_size = 3;
  return c;
}

bool boxes_well_formed(const std::vector<ppocr::DetBox>& bs, int W, int H) {
  for (const auto& b : bs) {
    float xs[4] = {b.poly[0], b.poly[2], b.poly[4], b.poly[6]};
    float ys[4] = {b.poly[1], b.poly[3], b.poly[5], b.poly[7]};
    for (int k = 0; k < 4; ++k) {
      if (!(xs[k] >= -1.0f && xs[k] <= static_cast<float>(W))) return false;
      if (!(ys[k] >= -1.0f && ys[k] <= static_cast<float>(H))) return false;
    }
    if (!(b.score > 0.0f && b.score <= 1.0f)) return false;
    // A box must have some extent in at least one direction.
    float w = std::fabs(xs[1] - xs[0]);
    float h = std::fabs(ys[3] - ys[0]);
    if (w <= 0.0f && h <= 0.0f) return false;
  }
  return true;
}

bool identical(const std::vector<ppocr::DetBox>& a,
               const std::vector<ppocr::DetBox>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].score != b[i].score) return false;
    for (int k = 0; k < 8; ++k) if (a[i].poly[k] != b[i].poly[k]) return false;
  }
  return true;
}

// Shape zoo of clean (noiseless) masks, covering the cases where the legacy
// tracer's approximations (size-desc ordering, synthesized hole rings) could
// have diverged from Suzuki's exact cv2 semantics.
std::vector<std::vector<float>> clean_masks(int W, int H) {
  std::vector<std::vector<float>> out;
  auto blank = [&] { return std::vector<float>((size_t)W * H, 0.02f); };
  auto rect = [&](std::vector<float>& p, int x0, int y0, int x1, int y1,
                  float v) {
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) p[(size_t)y * W + x] = v;
  };

  { auto p = blank(); rect(p, 20, 20, 120, 70, 0.95f); out.push_back(p); }
  { auto p = blank();                                  // size-desc != raster order
    rect(p, 10, 5, 40, 25, 0.95f);                     // small, top
    rect(p, 50, 40, 110, 90, 0.95f); out.push_back(p); }
  { auto p = blank();                                  // ring -> hole border
    rect(p, 20, 20, 120, 90, 0.95f);
    rect(p, 50, 40, 90, 70, 0.02f); out.push_back(p); }
  { auto p = blank();                                  // nested ring
    rect(p, 10, 10, 130, 110, 0.95f);
    rect(p, 30, 30, 110, 90, 0.02f);
    rect(p, 45, 45, 95, 75, 0.95f); out.push_back(p); }
  { auto p = blank();                                  // two rings
    rect(p, 10, 10, 60, 55, 0.95f); rect(p, 25, 25, 45, 40, 0.02f);
    rect(p, 70, 60, 130, 110, 0.95f); rect(p, 85, 75, 115, 95, 0.02f);
    out.push_back(p); }
  { auto p = blank();                                  // comb: many holes
    rect(p, 5, 5, 135, 105, 0.95f);
    for (int x = 20; x < 130; x += 15) rect(p, x, 30, x + 5, 80, 0.02f);
    out.push_back(p); }
  { auto p = blank();                                  // + diagonal chain
    for (int i = 0; i < 40; ++i) {
      rect(p, 10 + i, 10 + i, 12 + i, 12 + i, 0.95f);
    }
    out.push_back(p); }
  { auto p = blank(); out.push_back(p); }               // empty
  { std::vector<float> p((size_t)W * H, 0.95f); out.push_back(p); }  // solid
  return out;
}

}  // namespace

int main() {
  const int W = 160, H = 120;
  ppocr::DetConfig cfg = make_cfg();

  auto run_with = [&](const char* v, const std::vector<float>& prob) {
    if (v) setenv("PPOCR_SUZUKI", v, 1); else unsetenv("PPOCR_SUZUKI");
    return ppocr::db_postprocess(prob.data(), H, W, W, H, 1.0f, 1.0f, cfg);
  };

  auto masks = clean_masks(W, H);
  size_t total = 0;
  bool all_wellformed = true, all_agree = true;
  for (auto& p : masks) {
    auto leg = run_with(nullptr, p);
    auto sz  = run_with("1", p);
    total += leg.size();
    if (!boxes_well_formed(leg, W, H) || !boxes_well_formed(sz, W, H)) {
      all_wellformed = false;
    }
    if (!identical(leg, sz)) {
      all_agree = false;
      std::printf("  divergence on a clean mask: legacy n=%zu suzuki n=%zu\n",
                  leg.size(), sz.size());
    }
  }
  unsetenv("PPOCR_SUZUKI");

  std::printf("clean masks=%zu, boxes seen=%zu\n", masks.size(), total);
  check(total > 0, "the shape zoo actually produces boxes");
  check(all_wellformed, "both extractors emit well-formed boxes on every mask");
  check(all_agree,
        "on CLEAN masks both extractors agree exactly (divergence needs prob-map "
        "noise, not tracer logic)");

  // The env var must be read as truthy/falsy, not merely "present".
  auto with_zero = run_with("0", masks[1]);
  auto with_unset = run_with(nullptr, masks[1]);
  check(identical(with_zero, with_unset), "PPOCR_SUZUKI=0 selects the legacy path");

  std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall switch checks pass\n", failures);
  return failures ? 1 : 0;
}
