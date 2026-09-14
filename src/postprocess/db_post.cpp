// pp-ocr-mnn — DB postprocess implementation.
// Owner: post. No OpenCV; pure C++17. No platform ifdefs.
//
// Faithful port of ppocr/postprocess/db_postprocess.py boxes_from_bitmap +
// filter_tag_det_boxes + get_sorted_boxes. Notable details:
//   - Binarize with pred > thresh.
//   - 8-connected connected components, then extract external boundary per
//     component via Moore-neighbor tracing (matches OpenCV's RETR_LIST +
//     CHAIN_APPROX_SIMPLE for outer contours of binary mask).
//   - Per contour: minAreaRect -> sort_min_area_rect_points -> box_score_fast.
//   - Unclip: ClipperLib::ClipperOffset AddPath(jtRound, etClosedPolygon),
//     distance = area * unclip_ratio / perimeter (Paddle reference uses
//     shapely: poly.area * unclip_ratio / poly.length).
//   - Re-extract minAreaRect + sort; map bitmap coords -> original image coords
//     using ratio_w, ratio_h (Paddle does /W * dest_width and /H * dest_height).
#include "ppocr/postprocess/db_post.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include "ppocr/postprocess/geometry.h"
#include "ppocr/postprocess/suzuki.h"

// ---- dev-only sub-step instrumentation (PERF-POST) -------------------------
// Enabled by PPOCR_DB_PROFILE=1; ~4 steady_clock reads per db_postprocess
// call when off (one getenv + one branch). Never affects the result.
namespace ppocr {
namespace dbprof {
struct Slot { double ms = 0; int n = 0; };
struct Bank {
  Slot binarize, ccl, hole_flood, trace, ring, contour, sort;
  bool on = false;
  Bank() { const char* v = std::getenv("PPOCR_DB_PROFILE"); on = v && *v && *v != '0'; }
  void dump(int W, int H) {
    if (!on) return;
    std::fprintf(stderr,
      "[dbprof] %dx%d binarize %.2f ccl %.2f hole_flood %.2f trace %.2f "
      "hole_ring %.2f contour %.2f sort %.2f\n",
      W, H, binarize.ms, ccl.ms, hole_flood.ms, trace.ms, ring.ms,
      contour.ms, sort.ms);
    binarize.ms = ccl.ms = hole_flood.ms = trace.ms = ring.ms = contour.ms = sort.ms = 0;
  }
};
inline Bank& bank() { static Bank b; return b; }
struct Timer {
  Slot* s;
  std::chrono::steady_clock::time_point t0;
  explicit Timer(Slot* s_) : s(s_), t0(std::chrono::steady_clock::now()) {}
  void stop() {
    if (!s) return;
    if (bank().on) {
      s->ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0).count();
      s->n++;
    }
    s = nullptr;
  }
  ~Timer() { stop(); }
};
}  // namespace dbprof
}  // namespace ppocr

// Clipper (header-only, vendored). See third_party/clipper/clipper.hpp.
#include "clipper.hpp"

namespace ppocr {
namespace {

// Paddle's DBPostProcess.min_size — boxes with a shorter side below this are
// discarded before unclip. The value now comes from the JSON config; this
// documents Paddle's reference default (M2-ROBUST swept 3 / 5 / 10 in the
// config to find a noise-robust operating point for the MNN prob map).
constexpr int kMinSizeDefault = 3;
static_assert(kMinSizeDefault == 3, "documented Paddle DBPostProcess.min_size");

// --- polygon area (signed, Green's formula) ---------------------------------
float polygon_area(const std::vector<PointF>& p) {
  if (p.size() < 3) return 0.0f;
  double s = 0.0;
  for (size_t i = 0; i < p.size(); ++i) {
    size_t j = (i + 1) % p.size();
    s += static_cast<double>(p[i].x) * p[j].y;
    s -= static_cast<double>(p[j].x) * p[i].y;
  }
  return static_cast<float>(s * 0.5);
}

// --- polygon perimeter -------------------------------------------------------
float polygon_perimeter(const std::vector<PointF>& p) {
  if (p.size() < 2) return 0.0f;
  double s = 0.0;
  for (size_t i = 0; i < p.size(); ++i) {
    size_t j = (i + 1) % p.size();
    double dx = p[j].x - p[i].x;
    double dy = p[j].y - p[i].y;
    s += std::sqrt(dx * dx + dy * dy);
  }
  return static_cast<float>(s);
}

// --- fillPoly: rasterize a polygon into a mask ------------------------------
// Bit-exact port of the PaddleX box_score_fast rasterization:
//   cv2.fillPoly(mask, box.astype(np.int32), 1)
// Callers pass vertices already truncated to int (numpy astype semantics).
// Reproduces OpenCV's CollectPolyEdges (LINE_8 outline stroke + 16.16
// fixed-point fill edges) and FillEdgeCollection (even-odd span drawing),
// modules/imgproc/src/drawing.cpp @ 4.x, INCLUDING the pathological
// mutation of the tmp sentinel (x=0,dx=0) when it is drawn as an endpoint.
namespace {
struct CvEdge { int y0, y1; int64_t x, dx; };

// cv::LineIterator connectivity=8, leftToRight=false.
void bres_line(std::vector<uint8_t>& m, int W, int H,
               int x1, int y1, int x2, int y2) {
  if (x1 < 0 || y1 < 0 || x1 >= W || y1 >= H) return;
  if (x2 < 0 || y2 < 0 || x2 >= W || y2 >= H) return;
  int d_x = 1, d_y = 1;
  int dx = x2 - x1, dy = y2 - y1;
  if (dx < 0) { dx = -dx; d_x = -1; }
  if (dy < 0) { dy = -dy; d_y = -1; }
  bool vert = dy > dx;
  if (vert) { std::swap(dx, dy); std::swap(d_x, d_y); }
  int err = dx - dy - dy;
  int plusDelta = dx + dx;
  int minusDelta = -(dy + dy);
  int minusShift = d_x, plusShift = 0;
  int minusStep = 0, plusStep = d_y;
  if (vert) {
    std::swap(plusStep, plusShift);
    std::swap(minusStep, minusShift);
  }
  int px = x1, py = y1;
  for (int i = 0; i <= dx; ++i) {
    m[static_cast<size_t>(py) * W + px] = 1;
    int mk = err < 0 ? -1 : 0;
    err += minusDelta + (plusDelta & mk);
    px += minusShift + (plusShift & mk);
    py += minusStep + (plusStep & mk);
  }
}
}  // namespace

std::vector<uint8_t> fill_polygon_mask(const std::vector<PointF>& poly_in, int W,
                                      int H) {
  std::vector<uint8_t> mask(W * H, 0);
  if (poly_in.size() < 3) return mask;
  const int n = static_cast<int>(poly_in.size());
  auto clamp_x = [&](int64_t v) { return std::min(std::max(v, (int64_t)0), (int64_t)(W - 1)); };
  auto clamp_y = [&](int64_t v) { return std::min(std::max(v, (int64_t)0), (int64_t)(H - 1)); };
  std::vector<int> vx(n), vy(n);          // rounded pixel coords (stroke uses these)
  std::vector<int64_t> vfx(n);            // <<16 fixed point for fill edges
  for (int i = 0; i < n; ++i) {
    vx[i] = static_cast<int>(clamp_x(static_cast<int64_t>(poly_in[i].x)));
    vy[i] = static_cast<int>(clamp_y(static_cast<int64_t>(poly_in[i].y)));
    vfx[i] = vx[i] << 16;
  }
  // ---- CollectPolyEdges ----
  std::vector<CvEdge> edges;
  edges.reserve(n);
  for (int i = 0; i < n; ++i) {
    const int j = (i + n - 1) % n;
    bres_line(mask, W, H, vx[j], vy[j], vx[i], vy[i]);      // LINE_8 stroke
    if (vy[j] == vy[i]) continue;                            // horizontal edge: no fill
    CvEdge e;
    const int64_t num = vfx[i] - vfx[j];                     // p1cx - p0cx
    e.dx = num / (vy[i] - vy[j]);                            // C++ trunc-toward-zero div
    if (vy[j] < vy[i]) { e.y0 = vy[j]; e.y1 = vy[i]; e.x = vfx[j]; }
    else               { e.y0 = vy[i]; e.y1 = vy[j]; e.x = vfx[i]; }
    edges.push_back(e);
  }
  if (edges.empty()) return mask;
  std::sort(edges.begin(), edges.end(), [](const CvEdge& a, const CvEdge& b) {
    if (a.y0 != b.y0) return a.y0 < b.y0;
    if (a.x != b.x) return a.x < b.x;
    return a.dx < b.dx;
  });
  // OpenCV appends a {y0=INT_MAX} sentinel AFTER sorting: the scan loop's
  // advance condition reads e->y0 for the NEXT un-inserted edge even when
  // i == total, relying on the sentinel to make (e->y0 > y) true. Without
  // it we would read past the array (UB) and the fill stops after row 0 —
  // this was the en/01,04,06 box-count bug (only contour pixels scored).
  CvEdge sentinel{};
  sentinel.y0 = INT_MAX;
  edges.push_back(sentinel);
  const int total = static_cast<int>(edges.size()) - 1;  // real edge count
  int y_max_v = INT_MIN;
  for (int k = 0; k < total; ++k) y_max_v = std::max(y_max_v, edges[k].y1);
  if (y_max_v < 0) return mask;
  y_max_v = std::min(y_max_v, H);
  constexpr int64_t kDelta = ((int64_t)1 << 16) - 1;

  // ---- FillEdgeCollection ----
  // Active list: nodes indexed 1:1 by edge index (node never outlives its
  // edge), kept as an explicit singly linked list through node_next[].
  // kSent (-2) is the 'tmp' head sentinel; per the C++ original it is a
  // real PolyEdge{x=0,dx=0} that gets mutated in draw steps, so we track
  // sent_x/sent_dx separately.
  constexpr int kSent = -2;
  std::vector<int> node_next(total, -1);
  int tmp_next = -1;
  int64_t sent_x = 0, sent_dx = 0;
  int ei = 0;
  for (int y = edges.front().y0; y < y_max_v; ++y) {
    bool clipline = y < 0;
    int draw = 0;
    int prelast = kSent;
    int last = tmp_next;
    while (last != -1 || (ei < total && edges[ei].y0 == y)) {
      if (last != -1 && edges[last].y1 == y) {
        // exclude edge
        if (prelast == kSent) tmp_next = node_next[last];
        else                  node_next[prelast] = node_next[last];
        last = node_next[last];
        continue;
      }
      int keep_prelast = prelast;
      if (last != -1 && (edges[ei].y0 > y || edges[last].x < edges[ei].x)) {
        prelast = last;
        last = node_next[last];
      } else if (ei < total) {
        int nn = ei++;
        if (prelast == kSent) tmp_next = nn;
        else                  node_next[prelast] = nn;
        node_next[nn] = last;
        prelast = nn;
      } else {
        break;
      }
      if (draw) {
        if (!clipline) {
          int64_t xa = keep_prelast == kSent ? sent_x : edges[keep_prelast].x;
          int64_t xb = prelast == kSent ? sent_x : edges[prelast].x;
          int x1, x2;
          if (xa > xb) { x1 = (int)((xb + kDelta) >> 16); x2 = (int)(xa >> 16); }
          else         { x1 = (int)((xa + kDelta) >> 16); x2 = (int)(xb >> 16); }
          if (x1 < W && x2 >= 0) {
            if (x1 < 0) x1 = 0;
            if (x2 >= W) x2 = W - 1;
            uint8_t* row = mask.data() + static_cast<size_t>(y) * W;
            std::memset(row + x1, 1, static_cast<size_t>(x2 - x1 + 1));
          }
        }
        if (keep_prelast != kSent) edges[keep_prelast].x += edges[keep_prelast].dx;
        else                       sent_x += sent_dx;
        if (prelast != kSent)      edges[prelast].x += edges[prelast].dx;
        else                       sent_x += sent_dx;
      }
      draw ^= 1;
    }
    // bubble pass keeping active list ascending in x
    bool inverted = true;
    while (inverted) {
      inverted = false;
      int pl = kSent;
      int l = tmp_next;
      while (l != -1) {
        int t = node_next[l];
        if (t != -1 && edges[l].x > edges[t].x) {
          inverted = true;
          int ln = node_next[t];
          if (pl == kSent) tmp_next = t;
          else             node_next[pl] = t;
          node_next[t] = l;
          node_next[l] = ln;
          pl = t;
        } else {
          pl = l;
        }
        l = t;
      }
    }
  }
  return mask;
}

}  // namespace

std::vector<DetBox> db_postprocess(const float* prob, int prob_h, int prob_w,
                                   int src_w, int src_h, float ratio_w,
                                   float ratio_h, const DetConfig& cfg) {
  std::vector<DetBox> out;
  if (!prob || prob_h <= 0 || prob_w <= 0) return out;
  const int H = prob_h;
  const int W = prob_w;
  dbprof::Bank& prof = dbprof::bank();
  dbprof::Slot dummy_slot;
  dbprof::Slot* bin_slot = prof.on ? &prof.binarize : &dummy_slot;
  int processed_cand = 0;
  // 1) Binarize.
  std::vector<uint8_t> mask(W * H, 0);
  {
    dbprof::Timer _t(bin_slot);
    for (int i = 0; i < W * H; ++i) {
      mask[i] = (prob[i] > cfg.thresh) ? 1 : 0;
    }
  }
  // Paddle's dilation off by default. Toggling would require a 2x2 dilate on
  // the binary mask; intentionally not wired because DetConfig has no flag
  // for it in CONTRACT.md.
  (void)0;

  // 2) Contours: one Suzuki-Abe pass over the binary mask.
  //
  // This replaces the previous "8-connected CCL -> global hole flood-fill ->
  // Moore-neighbour trace" trio, which was the dominant cost of db_post
  // (67.8 -> 27 ms had already been won by killing its O(N*n_comp) and
  // O(holes * W*H) hot spots; the tracer itself is now O(border pixels)).
  //
  // The tracer reproduces cv::findContours(RETR_LIST, CHAIN_APPROX_NONE)
  // exactly — value-, sign- and order-wise (validated on 37 synthetic masks
  // plus 5 real prob maps against cv2 4.10 and cv2 5.0, see
  // tools/verify_suzuki.py). That is a strict improvement in fidelity over
  // the old approximation, which sorted components by pixel count and
  // synthesized hole rings by wall-following, and separately had to be
  // trusted to match cv2's ordering.
  std::vector<std::vector<IntPoint>> contours;
  {
    dbprof::Timer _t(prof.on ? &prof.trace : &dummy_slot);
    suzuki_borders(mask.data(), W, H, &contours);
  }

  // Paddle's DBPostProcess iterates `contours[:max_candidates]` in cv2's
  // emitted order. cv2's order is reverse discovery order (see suzuki.cpp),
  // so "top N" is NOT the N largest components — we must not re-sort.
  // max_candidates is 1000 (docs/DET_GEOMETRY.md) and real prob maps yield
  // tens to a few hundred borders, so the cap is a backstop, not a filter.
  const size_t max_cand = static_cast<size_t>(std::max(1, cfg.max_candidates));
  const size_t n_use = std::min(contours.size(), max_cand);
  (void)processed_cand;

  // Per-candidate funnel shared by external borders and synthesized hole
  // rings: minibox -> score -> unclip -> re-minibox -> sside -> map.
  auto process_contour = [&](const std::vector<PointF>& boundary) -> bool {

    // Note: Paddle's boxes_from_bitmap (box_type="quad", the default used by
    // PaddleOCR 3.x paddlex pipeline) does NOT apply approxPolyDP. It feeds
    // the raw contour from cv::findContours directly into GetMiniBoxes. We
    // do the same: skip dp_simplify here. (ppocr's polygons_from_bitmap
    // path uses approxPolyDP, but that's the poly box_type we don't emit.)

    // Compute mini-box (minAreaRect -> sort_min_area_rect_points). Paddle's
    // get_mini_boxes also returns the short side length for sside filter.
    std::vector<PointF> pts(boundary.begin(), boundary.end());
    PointF box4[4];
    if (!min_area_rect(pts.data(), pts.size(), box4)) { return false; }
    sort_min_area_rect_points(box4);
    // Convert to flat poly (8 floats) for box_score_fast.
    std::vector<PointF> sorted_box4(box4, box4 + 4);

    // box_score_fast: mean of pred inside the polygon.
    auto poly_mask = fill_polygon_mask(sorted_box4, W, H);
    // Bounding box of the polygon.
    int xmin = W - 1, xmax = 0, ymin = H - 1, ymax = 0;
    for (const auto& p : sorted_box4) {
      int xi = static_cast<int>(std::floor(p.x));
      int yi = static_cast<int>(std::floor(p.y));
      if (xi < xmin) xmin = xi;
      if (xi > xmax) xmax = xi;
      if (yi < ymin) ymin = yi;
      if (yi > ymax) ymax = yi;
    }
    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax > W - 1) xmax = W - 1;
    if (ymax > H - 1) ymax = H - 1;
    if (xmax < xmin || ymax < ymin) { return false; }
    double sum_prob = 0.0;
    int sum_count = 0;
    for (int yy = ymin; yy <= ymax; ++yy) {
      const uint8_t* mrow = poly_mask.data() + yy * W;
      const float* prow = prob + yy * W;
      for (int xx = xmin; xx <= xmax; ++xx) {
        if (mrow[xx]) {
          sum_prob += prow[xx];
          ++sum_count;
        }
      }
    }
    if (sum_count == 0) { return false; }
    float score = static_cast<float>(sum_prob / sum_count);
    if (cfg.box_thresh > score) { return false; }

    // Unclip: distance = area * unclip_ratio / perimeter.
    float area = polygon_area(sorted_box4);
    if (area <= 0.0f) { return false; }
    float perim_b = polygon_perimeter(sorted_box4);
    if (perim_b <= 0.0f) return false;
    float distance = area * cfg.unclip_ratio / perim_b;

    // Build clipper path. Paddle's pyclipper (and Paddle's C++ unclip
    // implementation) takes the path and distance in image coords directly;
    // no SCALE=10000 multiplication is needed. (The SCALE factor is only
    // used by CropByPolys for IoU computation with Clipper, which uses
    // integer arithmetic and is sensitive to roundoff.)
    ClipperLib::Path path;
    path.reserve(sorted_box4.size());
    for (const auto& p : sorted_box4) {
      // Paddle's ppocr/postprocess/db_postprocess.py unclip passes the box
      // directly to pyclipper.PyclipperOffset.AddPath(...). pyclipper's
      // Cython wrapper (`_to_clipper_point`) constructs the C++ IntPoint
      // as `IntPoint(py_point[0], py_point[1])` — a direct C++ struct
      // construction that **truncates** Python floats to int64. We must
      // match that here. `std::llround` would round .5 toward +∞ and
      // shifts each vertex by 0–1 px on every odd-fractional coord, which
      // is the dominant source of the 1–7 px systematic offset we saw
      // when comparing our C++ clipper output to pyclipper (see
      // tests/verify_unclip.py).
      path << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(p.x),
                                   static_cast<ClipperLib::cInt>(p.y));
    }
    ClipperLib::ClipperOffset co;
    co.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    ClipperLib::Paths solution;
    co.Execute(solution, distance);
    if (solution.size() != 1) { return false; }

    std::vector<PointF> expanded;
    expanded.reserve(solution[0].size());
    for (const auto& ip : solution[0]) {
      expanded.push_back({static_cast<float>(ip.X), static_cast<float>(ip.Y)});
    }
    if (expanded.size() < 4) { return false; }

    // Re-mini-box and sort.
    PointF box2[4];
    if (!min_area_rect(expanded.data(), expanded.size(), box2)) { return false; }
    sort_min_area_rect_points(box2);
    std::vector<PointF> final_box(box2, box2 + 4);

    // sside filter: short side of the min rect.
    float w1 = std::sqrt((final_box[0].x - final_box[1].x) *
                             (final_box[0].x - final_box[1].x) +
                         (final_box[0].y - final_box[1].y) *
                             (final_box[0].y - final_box[1].y));
    float w2 = std::sqrt((final_box[2].x - final_box[3].x) *
                             (final_box[2].x - final_box[3].x) +
                         (final_box[2].y - final_box[3].y) *
                             (final_box[2].y - final_box[3].y));
    float h1 = std::sqrt((final_box[0].x - final_box[3].x) *
                             (final_box[0].x - final_box[3].x) +
                         (final_box[0].y - final_box[3].y) *
                             (final_box[0].y - final_box[3].y));
    float h2 = std::sqrt((final_box[1].x - final_box[2].x) *
                             (final_box[1].x - final_box[2].x) +
                         (final_box[1].y - final_box[2].y) *
                             (final_box[1].y - final_box[2].y));
    float sside = std::min({w1, w2, h1, h2});
    if (sside < static_cast<float>(cfg.min_size + 2)) { return false; }

    // Map bitmap coords -> original image coords, VERBATIM Paddle:
    //   boxes[:, 0] = (boxes[:, 0] * (dst_w / W)).round()  [clip 0..dst_w-1]
    //   where the division happens first in float64, and np.round is
    //   half-to-EVEN. Matching this exactly removes a systematic
    //   +/-1px discrepancy on some boxes (half-away rounding + float32
    //   premultiply vs float64 divide-then-round-even).
    DetBox db;
    for (int k = 0; k < 4; ++k) {
      auto map_axis = [&](double bxy, int bitmap_len,
                          int dst_len) -> double {
        double scaled = bxy * dst_len / bitmap_len;
        // np.round: round-half-to-even on .5 boundaries
        double fl = std::floor(scaled);
        double frac = scaled - fl;
        double r;
        if (frac > 0.5)
          r = fl + 1.0;
        else if (frac < 0.5)
          r = fl;
        else
          r = (std::fmod(fl, 2.0) == 0.0) ? fl : fl + 1.0;
        return std::min(std::max(r, 0.0), static_cast<double>(dst_len - 1));
      };
      db.poly[k * 2 + 0] = static_cast<float>(
          map_axis(static_cast<double>(final_box[k].x), prob_w, src_w));
      db.poly[k * 2 + 1] = static_cast<float>(
          map_axis(static_cast<double>(final_box[k].y), prob_h, src_h));
    }
    db.score = score;
    out.push_back(db);
    return true;
  };

  // 3) Run every contour through the funnel: minibox -> score -> unclip ->
  //    re-minibox -> sside -> map to original image coords.
  for (size_t ci = 0; ci < n_use; ++ci) {
    const auto& c = contours[ci];
    if (c.size() < 4) continue;
    std::vector<PointF> boundary;
    boundary.reserve(c.size());
    for (const auto& q : c) {
      boundary.push_back({static_cast<float>(q.x), static_cast<float>(q.y)});
    }
    dbprof::Timer _t(prof.on ? &prof.contour : &dummy_slot);
    if (process_contour(boundary)) ++processed_cand;
  }
  prof.dump(W, H);
  return out;
}

void sort_quad_boxes_reading_order(std::vector<DetBox>& boxes) {
  // Faithful port of Paddle C++ ComponentsProcessor::SortQuadBoxes
  // (deploy/cpp_infer/src/common/processors.cc:590-611).
  //   1) std::sort by (a.poly[1] < b.poly[1]) || (== && a.poly[0] < b.poly[0])
  //      — primary key is the y of the first vertex (the TL corner of the
  //        mini-box that db_postprocess emits, since we sort_min_area_rect_points
  //        puts TL at poly[0]).
  //   2) Bubble pass: for each i in [0..N-2), walk j from i+1 down to 1; if
  //      |y_j - y_{j-1}| < 10 && x_j < x_{j-1} then swap, else break. This
  //      re-orders boxes that landed in the wrong row bucket (e.g. when two
  //      rows have a y difference < 10 px the primary sort may group them
  //      together; the bubble pass enforces left-to-right within that group).
  // The strict < ordering matches cv::Point2f's float comparison in Paddle
  // (no epsilon, NaN propagates the same way).
  if (boxes.size() < 2) return;
  std::sort(boxes.begin(), boxes.end(),
            [](const DetBox& a, const DetBox& b) {
              return (a.poly[1] < b.poly[1]) ||
                     (a.poly[1] == b.poly[1] && a.poly[0] < b.poly[0]);
            });
  for (size_t i = 0; i + 1 < boxes.size(); ++i) {
    for (size_t j = i + 1; j > 0; --j) {
      if (std::fabs(boxes[j].poly[1] - boxes[j - 1].poly[1]) < 10.0f &&
          boxes[j].poly[0] < boxes[j - 1].poly[0]) {
        std::swap(boxes[j], boxes[j - 1]);
      } else {
        break;
      }
    }
  }
}

}  // namespace ppocr
