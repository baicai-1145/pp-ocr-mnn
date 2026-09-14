// pp-ocr-mnn — Suzuki-Abe single-pass border following. See suzuki.h for the
// full contract and the mapping to OpenCV's contours_new.cpp.
#include "ppocr/postprocess/suzuki.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>

namespace ppocr {
namespace {

// cv::chainCodeDeltas, i.e. clockwise from East:
//   { (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1) }
// getDelta() indexes with `s % 8` because the follower deliberately lets `s`
// run up to 15 before clamping (OpenCV's MAX_SIZE trick).
constexpr int kDx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int kDy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
inline int dx_of(int s) { return kDx[s & 7]; }
inline int dy_of(int s) { return kDy[s & 7]; }

constexpr int kMaxSize = 16;
inline int clamp_direction(int s) { return s < kMaxSize - 1 ? s : kMaxSize - 1; }

// Marks (contours_new.cpp).
constexpr uint8_t kBlack = 0x01;  // MASK8_BLACK — still-unmarked foreground
constexpr uint8_t kNew = 0x02;    // MASK8_NEW
constexpr uint8_t kRight = 0x80;  // MASK8_RIGHT

// OpenCV takes the mask as `schar` (SIGNED), so a pixel value may only ever be
// 0/1 on entry. Callers must hand us a 0/1 mask; we normalise defensively so a
// stray marker-style byte can't be mistaken for foreground.

// Trait<schar>: OpenCV declares the mask as `schar*` — SIGNED. That sign is
// load-bearing, not cosmetic:
//   checkValue(e)  is `*e != 0`                  -> marker bytes still count
//   isVal(e)       is `*e == MASK8_BLACK (0x01)` -> only unmarked foreground
//   contourScan's  `prev < 1` is a SIGNED test   -> a marker byte (0x82 == -126)
//                                                    reads as < 1 and therefore
//                                                    takes the resume_scan branch
// With an unsigned view, image[y][x-1] == 0x82 would be `>= 1` and the scan
// would keep re-detecting the pixel it just traced as a hole border. So we do
// the sign-sensitive comparisons through int8_t.
inline bool check_value(const uint8_t* p) { return *p != 0; }
inline bool is_val(const uint8_t* p) { return *p == kBlack; }
inline int8_t sig(uint8_t v) { return static_cast<int8_t>(v); }

// contourScan's classify step, verbatim (prev is the SIGNED byte to the left):
//     if (!(prev == 0 && p == 1)) {
//         if (p != 0 || prev < 1) return false;   // resume_scan
//         is_hole = true;
//     }
inline bool is_border_start(uint8_t prev, uint8_t p, bool* is_hole) {
  if (prev == 0 && p == 1) { *is_hole = false; return true; }
  if (p != 0 || sig(prev) < 1) return false;
  *is_hole = true;
  return true;
}

struct Rect {
  int x = 0, y = 0, w = 0, h = 0;
};

// --- icvFetchContourEx<schar> ------------------------------------------------
// Trace the border that starts at (start_x, start_y) — already
// `x - (is_hole ? 1 : 0)` — appending its points to `pts`.
//
// `is_direct` is OpenCV's `approx_method1 == CHAIN_APPROX_NONE`, i.e. the
// `s != prev_s || isDirect` clause in the point-emitting test. Callers here
// always pass true: db_post feeds the raw contour into minAreaRect, which is
// what `cv2.findContours(..., cv2.CHAIN_APPROX_NONE)` (and the old Moore
// tracer) produce. The `false` mode reproduces CHAIN_APPROX_SIMPLE.
//
// Faithful transliteration of the original loop. Note that `i3` is advanced to
// `i4` AFTER the loop-closure test (`i4 == i0 && i3 == i1`), which is what lets
// the walk terminate on the pixel it started from.
void fetch_contour(uint8_t* image, int w, int h, int start_x, int start_y,
                   bool is_hole, std::vector<IntPoint>* pts, Rect* rect,
                   bool is_direct = true) {
  const int step = w;
  uint8_t* i0 = image + static_cast<size_t>(start_y) * w + start_x;
  uint8_t* i1 = nullptr;
  uint8_t* i3 = nullptr;
  uint8_t* i4 = nullptr;
  const uint8_t nbd = kNew;

  Rect r{start_x, start_y, start_x, start_y};

  int s_end = is_hole ? 0 : 4;
  int s = s_end;
  do {
    s = (s - 1) & 7;
    i1 = i0 + dy_of(s) * step + dx_of(s);
  } while (!check_value(i1) && s != s_end);

  if (s == s_end) {
    // Single-pixel domain.
    *i0 = static_cast<uint8_t>(nbd | kRight);
    pts->push_back({start_x, start_y});
  } else {
    i3 = i0;
    int prev_s = s ^ 4;
    for (;;) {
      s_end = s;
      s = clamp_direction(s);
      while (s < kMaxSize - 1) {
        ++s;
        i4 = i3 + dy_of(s) * step + dx_of(s);
        if (check_value(i4)) break;
      }
      s &= 7;

      if (static_cast<unsigned>(s - 1) < static_cast<unsigned>(s_end)) {
        *i3 = static_cast<uint8_t>(nbd | kRight);
      } else if (is_val(i3)) {
        *i3 = nbd;
      }

      const int px = static_cast<int>((i3 - image) % step);
      const int py = static_cast<int>((i3 - image) / step);
      if (s != prev_s || is_direct) {
        pts->push_back({px, py});
      }
      if (s != prev_s) {
        if (px < r.x) r.x = px;
        else if (px > r.w) r.w = px;
        if (py < r.y) r.y = py;
        else if (py > r.h) r.h = py;
      }

      prev_s = s;

      if (i4 == i0 && i3 == i1) break;

      i3 = i4;
      s = (s + 4) & 7;
    }
  }
  if (rect) {
    rect->x = r.x;
    rect->y = r.y;
    rect->w = r.w - r.x + 1;
    rect->h = r.h - r.y + 1;
  }
}

}  // namespace

void suzuki_borders(const uint8_t* image, int w, int h,
                    std::vector<std::vector<IntPoint>>* contours) {
  contours->clear();
  if (!image || w < 1 || h < 1) return;

  // cv2 pads with a 1px zero border before scanning (copyMakeBorder,
  // BORDER_ISOLATED) and reports coordinates shifted back by (-1, -1).
  const int pw = w + 2;
  const int ph = h + 2;
  const int step = pw;
  std::vector<uint8_t> buf(static_cast<size_t>(pw) * ph, 0);
  for (int y = 0; y < h; ++y) {
    const uint8_t* src = image + static_cast<size_t>(y) * w;
    uint8_t* dst = buf.data() + static_cast<size_t>(y + 1) * step + 1;
    for (int x = 0; x < w; ++x) dst[x] = src[x] ? 1 : 0;
  }

  const int width = pw - 1;   // scan window: x in [1, width), y in [1, height)
  const int height = ph - 1;

  int x = 1;
  int y = 1;
  int8_t prev = sig(buf[static_cast<size_t>(y) * step + (x - 1)]);

  for (; y < height; ++y) {
    uint8_t* row = buf.data() + static_cast<size_t>(y) * step;
    int p = 0;
    for (; x < width; ++x) {
      // ---- findNextX: skip the run of bytes equal to `prev` ----
      p = row[x];
      if (p == prev) {
        int nx = x + 1;
        while (nx < width && row[nx] == prev) ++nx;
        x = nx;
        if (x >= width) break;
        p = row[x];
      }

      // ---- contourScan ----
      bool is_hole = false;
      if (!is_border_start(static_cast<uint8_t>(prev), static_cast<uint8_t>(p),
                           &is_hole)) {
        prev = static_cast<int8_t>(p);
        continue;
      }

      // ---- makeContour ----
      const int start_x = is_hole ? x - 1 : x;
      std::vector<IntPoint> pts;
      pts.reserve(128);
      Rect r;
      fetch_contour(buf.data(), step, ph, start_x, y, is_hole, &pts, &r);

      // findNext resumes at pt = (x + 1, y) and re-derives prev from
      // image[y][x + 1 - 1] — which now holds the marker the trace wrote.
      if (!pts.empty()) {
        for (auto& q : pts) { q.x -= 1; q.y -= 1; }
        contours->push_back(std::move(pts));
      }
      prev = sig(row[x]);

      // Hard backstop against a runaway scan: each contour visits at least one
      // pixel and marks it, so the number of contours is bounded by the area.
      if (contours->size() > static_cast<size_t>(pw) * ph) {
        return;
      }
    }
    // End of row: last_pos = (0, y + 1); x = 1; prev = 0.
    x = 1;
    prev = 0;
  }

  // cv2's output order is NOT discovery order. Every contour becomes a child of
  // the root node via TreeNode::addChild (contours_common.hpp), which PREPENDS
  // to the child list, and contourTreeToResults then walks that list — so the
  // caller sees the borders in reverse discovery order. Verified empirically
  // against cv2 4.10 on the whole mask corpus (see tests/test_suzuki.cpp and
  // tools/verify_suzuki.py): reversing makes the shapes and the point order
  // match exactly.
  std::reverse(contours->begin(), contours->end());
}

}  // namespace ppocr
