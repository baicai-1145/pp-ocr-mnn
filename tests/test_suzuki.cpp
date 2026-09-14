// pp-ocr-mnn — Suzuki-Abe border follower test.
//
// Two modes:
//   test_suzuki                 built-in cases; asserts shape invariants
//   test_suzuki FILE H W        read a raw H*W byte mask, dump contours to
//                               stdout as "x,y x,y ..." lines (for diffing
//                               against cv2.findContours by a driver script)
//
// The built-in cases need no OpenCV; the file mode is what
// tools/verify_suzuki.py drives to prove byte-parity with cv2.
#include "ppocr/postprocess/suzuki.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using ppocr::IntPoint;

std::vector<std::vector<IntPoint>> run(std::vector<uint8_t> mask, int w, int h) {
  std::vector<std::vector<IntPoint>> cs;
  ppocr::suzuki_borders(mask.data(), w, h, &cs);
  return cs;
}

std::string fmt(const std::vector<IntPoint>& c) {
  std::string s;
  for (size_t i = 0; i < c.size(); ++i) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%d,%d", c[i].x, c[i].y);
    if (i) s += ' ';
    s += buf;
  }
  return s;
}

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
  else     { std::printf("ok  : %s\n", what); }
}

void expect_contours(const std::vector<std::vector<IntPoint>>& cs,
                     const std::vector<std::string>& want, const char* what) {
  std::vector<std::string> got;
  for (auto& c : cs) got.push_back(fmt(c));
  bool ok = got.size() == want.size();
  if (ok) for (size_t i = 0; i < got.size(); ++i) if (got[i] != want[i]) ok = false;
  if (!ok) {
    std::printf("FAIL: %s\n  got %zu contours:\n", what, got.size());
    for (auto& g : got) std::printf("    %s\n", g.c_str());
    std::printf("  want %zu contours:\n", want.size());
    for (auto& w : want) std::printf("    %s\n", w.c_str());
  }
  check(ok, what);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 4) {
    const char* path = argv[1];
    int H = std::atoi(argv[2]), W = std::atoi(argv[3]);
    std::vector<uint8_t> mask(static_cast<size_t>(W) * H);
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    if (std::fread(mask.data(), 1, mask.size(), f) != mask.size()) {
      std::fprintf(stderr, "short read %s\n", path); std::fclose(f); return 2;
    }
    std::fclose(f);
    for (auto& c : run(mask, W, H)) std::printf("%s\n", fmt(c).c_str());
    return 0;
  }

  // ---- built-in cases (values verified against cv2 5.0) ----
  {
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 0);
    for (int y = 4; y < 8; ++y) for (int x = 4; x < 10; ++x) m[y * W + x] = 1;
    expect_contours(run(m, W, H),
      {"4,4 4,5 4,6 4,7 5,7 6,7 7,7 8,7 9,7 9,6 9,5 9,4 8,4 7,4 6,4 5,4"},
      "solid rect: 1 outer border, clockwise from TL, every pixel (CHAIN_APPROX_NONE)");
  }
  {
    // donut: hole border first (first boundary run is at y=5), then the outer.
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 0);
    for (int y = 4; y < 8; ++y) for (int x = 4; x < 10; ++x) m[y * W + x] = 1;
    for (int y = 5; y < 7; ++y) for (int x = 5; x < 9; ++x) m[y * W + x] = 0;
    expect_contours(run(m, W, H),
      {"4,5 5,4 6,4 7,4 8,4 9,5 9,6 8,7 7,7 6,7 5,7 4,6",
       "4,4 4,5 4,6 4,7 5,7 6,7 7,7 8,7 9,7 9,6 9,5 9,4 8,4 7,4 6,4 5,4"},
      "donut: 2 contours — hole border (12pts) then outer border (16pts), cv2's emitted order");
  }
  {
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 0);
    for (int y = 4; y < 8; ++y) for (int x = 4; x < 6; ++x) m[y * W + x] = 1;
    expect_contours(run(m, W, H),
      {"4,4 4,5 4,6 4,7 5,7 5,6 5,5 5,4"},
      "narrow rect (2px wide)");
  }
  {
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 0);
    m[1 * W + 1] = 1;
    expect_contours(run(m, W, H), {"1,1"}, "single pixel domain");
  }
  {
    // 8-connectivity: two boxes touching corner-to-corner are ONE contour.
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 0);
    for (int y = 4; y < 8; ++y) for (int x = 4; x < 10; ++x) m[y * W + x] = 1;
    for (int y = 8; y < 12; ++y) for (int x = 10; x < 14; ++x) m[y * W + x] = 1;
    expect_contours(run(m, W, H),
      {"4,4 4,5 4,6 4,7 5,7 6,7 7,7 8,7 9,7 10,8 10,9 10,10 10,11 11,11 "
       "12,11 13,11 13,10 13,9 13,8 12,8 11,8 10,8 9,7 9,6 9,5 9,4 "
       "8,4 7,4 6,4 5,4"},
      "diagonal touching boxes merge into ONE 8-connected contour (exact cv2 point list)");
  }
  {
    // Concentric rings: 3 borders, innermost first.
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 0);
    for (int y = 2; y < 14; ++y) for (int x = 2; x < 14; ++x) m[y * W + x] = 1;
    for (int y = 4; y < 12; ++y) for (int x = 4; x < 12; ++x) m[y * W + x] = 0;
    for (int y = 6; y < 10; ++y) for (int x = 6; x < 10; ++x) m[y * W + x] = 1;
    auto cs = run(m, W, H);
    check(cs.size() == 3, "concentric rings -> 3 borders");
    check(cs[0].size() == 12 && cs[1].size() == 32 && cs[2].size() == 44,
          "concentric rings: sizes 12 / 32 / 44 in cv2's emitted order");
  }
  {
    // Foreground touching the mask edge DOES produce a contour: cv2 pads with a
    // 1px zero border and then shifts coordinates back by (-1,-1), so the
    // border run shows up at x/y == -1. This blob occupies y 0..4, x 0..11, so
    // its outer border walks the padded edge at -1 and closes along the bottom.
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 0);
    for (int y = 0; y < 5; ++y) for (int x = 0; x < 12; ++x) m[y * W + x] = 1;
    auto cs = run(m, W, H);
    check(cs.size() == 1, "edge-touching blob -> 1 contour (cv2 pads, then offsets by -1)");
    check(cs.size() == 1 && cs[0].size() == 30, "edge-touching blob border has 30 points");
    check(cs.size() == 1 && cs[0][0].x == 0 && cs[0][0].y == 0,
          "edge-touching blob starts at (0,0)");
  }
  {
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 0);
    auto cs = run(m, W, H);
    check(cs.empty(), "all-zero mask -> no contours");
  }
  {
    // All-ones: the padded border makes this one blob whose border hugs the
    // padding, i.e. 4 corner arcs — 60 points total in cv2.
    const int W = 16, H = 16;
    std::vector<uint8_t> m(W * H, 1);
    auto cs = run(m, W, H);
    check(cs.size() == 1, "all-ones mask -> 1 contour (padding turns it into a bordered blob)");
    check(cs.size() == 1 && cs[0].size() == 60, "all-ones mask: 60-point border, matching cv2");
    check(cs.size() == 1 && cs[0][0].x == 0 && cs[0][0].y == 0,
          "all-ones mask starts at (0,0)");
  }
  std::printf(failures ? "\n%d FAILURE(S)\n" : "\nall built-in cases pass\n", failures);
  return failures ? 1 : 0;
}
