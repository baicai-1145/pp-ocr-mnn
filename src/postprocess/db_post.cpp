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
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include "ppocr/postprocess/geometry.h"

// Clipper (header-only, vendored). See third_party/clipper/clipper.hpp.
#include "clipper.hpp"

namespace ppocr {
namespace {

// Paddle's DBPostProcess.min_size — boxes with a shorter side below this are
// discarded before unclip. Now cfg-driven; the constexpr is the fallback
// used when the JSON config does not specify min_size (Paddle's reference
// default is 3). M2-ROBUST sweeps 3 / 5 / 10 in the JSON config to find
// a noise-robust operating point for the MNN prob map.
constexpr int kMinSizeDefault = 3;

// --- connected components: 2-pass 8-connectivity union-find ------------------
struct UnionFind {
  std::vector<int> parent;
  std::vector<int> rank;
  void init(int n) {
    parent.resize(n);
    rank.assign(n, 0);
    for (int i = 0; i < n; ++i) parent[i] = i;
  }
  int find(int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }
  void unite(int a, int b) {
    int ra = find(a), rb = find(b);
    if (ra == rb) return;
    if (rank[ra] < rank[rb]) std::swap(ra, rb);
    parent[rb] = ra;
    if (rank[ra] == rank[rb]) ++rank[ra];
  }
};

// --- Moore-neighbor external boundary trace ---------------------------------
// Given a label `target`, walk the external boundary of the component starting
// at (sx, sy) using Moore-neighbor 8-direction traversal. The boundary is the
// set of (x,y) such that pixel(x,y) == target AND at least one 4-neighbor is
// not target (or out-of-bounds).
//
// Returns the ordered boundary as a list of (x,y). The first point is (sx,sy).
// On failure, returns an empty list.
std::vector<PointF> trace_boundary(const std::vector<int>& mask, int W, int H,
                                   int target, int sx, int sy) {
  // 8 directions, clockwise starting from "right".
  static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
  static const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

  auto at = [&](int x, int y) -> int {
    if (x < 0 || y < 0 || x >= W || y >= H) return 0;
    return mask[static_cast<size_t>(y) * W + x];
  };

  std::vector<PointF> out;
  out.reserve(64);
  int cx = sx, cy = sy;
  // Standard Moore-neighbor initial backtrack: for an external boundary
  // starting at the first non-zero pixel (scanned row-major), the
  // "previous" boundary pixel is the one to the WEST. Direction index 4
  // is W (dx=-1, dy=0). The next-neighbor scan starts at (prev+1) mod 8
  // = NW (index 5) and proceeds clockwise.
  int prev_dir = 4;
  bool first = true;
  int safety = W * H * 4 + 16;
  while (safety-- > 0) {
    out.push_back({static_cast<float>(cx), static_cast<float>(cy)});
    int start_dir = (prev_dir + 1) % 8;
    bool moved = false;
    for (int step = 0; step < 8; ++step) {
      int d = (start_dir + step) % 8;
      int nx = cx + dx[d];
      int ny = cy + dy[d];
      if (at(nx, ny) == target) {
        prev_dir = (d + 4) % 8;  // backtrack is opposite of the move
        cx = nx;
        cy = ny;
        moved = true;
        break;
      }
    }
    if (!moved) break;  // isolated pixel
    if (cx == sx && cy == sy && !first) break;
    first = false;
  }
  return out;
}

// --- Hole-contour synthesis (RETR_LIST parity) ------------------------------
// PaddleX's boxes_from_bitmap iterates cv2.findContours(RETR_LIST) output:
// hole borders are independent candidates. Our legacy extractor keeps only
// external borders. We synthesize each hole border as the ring of
// foreground pixels that 8-adjacent the hole region, walked clockwise from
// its top-left-most pixel. The downstream funnel consumes only the
// minAreaRect vertices of the ring, which is exactly what cv2's hole
// contour yields (validated against cv2 on the corpus fuzz set).
namespace {

std::vector<PointF> synthesize_hole_ring(const std::vector<int>& comp,
                                         const std::vector<uint8_t>& hole,
                                         int W, int H, int lbl) {
  // Ring: pixels of this component adjacent (8-conn) to the hole region.
  auto atc = [&](int x, int y) { return comp[static_cast<size_t>(y) * W + x] == lbl; };
  auto ath = [&](int x, int y) -> int {
    if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return 0;
    return static_cast<int>(hole[static_cast<size_t>(y) * W + x]);
  };
  std::vector<char> ring(static_cast<size_t>(W) * H, 0);
  std::vector<std::pair<int,int>> seeds;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (!ath(x, y)) continue;
      static const int dx8[8]={1,1,0,-1,-1,-1,0,1};
      static const int dy8[8]={-1,0,-1,-1,1,0,1,1};   // N NE NW .. SE S SW etc
      static const int ddx[8]={1,1,0,-1,-1,-1,0,1};
      static const int ddy[8]={0,-1,-1,-1,0,1,1,1};
      (void)dx8; (void)dy8;
      for (int d = 0; d < 8; ++d) {
        int nx = x + ddx[d], ny = y + ddy[d];
        if ((unsigned)nx < (unsigned)W && (unsigned)ny < (unsigned)H &&
            atc(nx, ny)) {
          size_t idx = static_cast<size_t>(ny) * W + nx;
          if (!ring[idx]) { ring[idx] = 1; seeds.emplace_back(nx, ny); }
        }
      }
    }
  if (seeds.size() < 4) return {};
  // top-left-most seed = min (y, x)
  std::pair<int,int> start = seeds[0];
  for (auto& s : seeds)
    if (s.second < start.second || (s.second == start.second && s.first < start.first))
      start = s;
  // clockwise walk with background-hand preference (wall follower on hole)
  std::vector<PointF> out;
  auto ith = [&](int x, int y) -> int {
    if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return 0;
    return static_cast<int>(hole[static_cast<size_t>(y) * W + x]);
  };
  auto isr = [&](int x, int y) -> int {
    if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return 0;
    return static_cast<int>(ring[static_cast<size_t>(y) * W + x] != 0);
  };
  int cx = start.first, cy = start.second;
  int dir = 7;                                   // start heading SW
  const int DX[8]={1,1,0,-1,-1,-1,0,1}, DY[8]={0,-1,-1,-1,0,1,1,1};
  size_t guard = static_cast<size_t>(W) * H * 8 + 16;
  bool first = true;
  std::pair<int,int> prev_dir_pt{-1,-1};
  while (guard--) {
    out.push_back({static_cast<float>(cx), static_cast<float>(cy)});
    if (!first && cx == start.first && cy == start.second) break;
    first = false;
    int found = -1;
    for (int k = 0; k < 8; ++k) {
      int nd = ((dir - 1) & 7);                   // turn left around hole
      nd = (dir + k) % 8;
      int nx = cx + DX[nd], ny = cy + DY[nd];
      if (isr(nx, ny) && ath(cx, cy)) {           // stay on ring pixels that touch hole
        found = nd; break;
      }
      nd = (dir + k) % 8;
      nx = cx + DX[nd]; ny = cy + DY[nd];
      if (isr(nx, ny) && !ith(nx, ny)) { found = nd; break; }
    }
    if (found < 0) break;
    dir = found;
    cx += DX[dir]; cy += DY[dir];
  }
  (void)prev_dir_pt;
  return out;
}

}  // namespace

// --- scan a label mask for the first boundary pixel of a given label --------
bool find_first_boundary(const std::vector<int>& mask, int W, int H, int target,
                         int* sx, int* sy) {
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (mask[static_cast<size_t>(y) * W + x] != target) continue;
      // Check 4-neighbors: a boundary pixel has at least one non-target
      // neighbor (or is on the image edge).
      bool is_border = false;
      if (x == 0 || x == W - 1 || y == 0 || y == H - 1) {
        is_border = true;
      } else {
        if (mask[static_cast<size_t>(y) * W + (x - 1)] != target ||
            mask[static_cast<size_t>(y) * W + (x + 1)] != target ||
            mask[static_cast<size_t>((y - 1)) * W + x] != target ||
            mask[static_cast<size_t>((y + 1)) * W + x] != target) {
          is_border = true;
        }
      }
      if (is_border) {
        *sx = x;
        *sy = y;
        return true;
      }
    }
  }
  return false;
}

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
  // 1) Binarize.
  std::vector<uint8_t> mask(W * H, 0);
  for (int i = 0; i < W * H; ++i) {
    mask[i] = (prob[i] > cfg.thresh) ? 1 : 0;
  }
  // Paddle's dilation off by default. Toggling would require a 2x2 dilate on
  // the binary mask; intentionally not wired because DetConfig has no flag
  // for it in CONTRACT.md.
  (void)0;

  // 2) 8-connected CCL via 2-pass union-find.
  UnionFind uf;
  uf.init(W * H);
  std::vector<int> label(W * H, 0);
  int next_label = 1;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (!mask[y * W + x]) continue;
      int my = next_label++;
      std::vector<int> neighbors;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) continue;
          int xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
          if (label[yy * W + xx] > 0) neighbors.push_back(label[yy * W + xx]);
        }
      if (neighbors.empty()) label[y * W + x] = my;
      else {
        int min_lbl = *std::min_element(neighbors.begin(), neighbors.end());
        label[y * W + x] = min_lbl;
        for (int n : neighbors) uf.unite(min_lbl, n);
      }
    }
  }
  std::vector<int> comp(W * H, 0);
  std::vector<int> comp_size;
  comp_size.push_back(0);
  for (int i = 0; i < W * H; ++i) {
    if (label[i] == 0) continue;
    int root = uf.find(label[i]);
    auto it = std::find(comp_size.begin() + 1, comp_size.end(), root);
    int idx;
    if (it == comp_size.end()) {
      comp_size.push_back(root);
      idx = static_cast<int>(comp_size.size()) - 1;
    } else {
      idx = static_cast<int>(it - comp_size.begin());
    }
    comp[i] = idx;
  }
  int n_comp = static_cast<int>(comp_size.size()) - 1;

  // 3) For each component, trace external boundary and process.
  int max_cand = std::max(1, cfg.max_candidates);
  int max_cand_n = std::min(n_comp, max_cand);

  // Sort components by size desc, like cv::findContours ordering (approx).
  std::vector<int> order(n_comp);
  std::iota(order.begin(), order.end(), 1);
  std::vector<int> comp_pixel_count(n_comp + 1, 0);
  for (int i = 0; i < W * H; ++i)
    if (comp[i] > 0) ++comp_pixel_count[comp[i]];
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return comp_pixel_count[a] > comp_pixel_count[b];
  });

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

  // ---- Global hole precompute ------------------------------------------
  // One pass over the background: every 4-connected bg region gets flooded
  // once; regions touching the image border or bordered by more than one
  // component are not holes. Result: holes_of[c] = list of hole pixel sets
  // fully enclosed by component c (RETR_LIST treats their borders as
  // independent contour candidates).
  std::unordered_map<int, std::vector<std::vector<std::pair<int,int>>>> holes_of;
  {
    std::vector<int> region_id(W * H, -1);
    int cur_region = 0;
    static const int DX[4] = {1,-1,0,0};
    static const int DY[4] = {0,0,1,-1};
    for (int yy = 0; yy < H; ++yy)
      for (int xx = 0; xx < W; ++xx) {
        size_t idx = static_cast<size_t>(yy) * W + xx;
        if (comp[idx] != 0 || region_id[idx] >= 0) continue;
        std::vector<std::pair<int,int>> stack{{xx,yy}};
        std::vector<std::pair<int,int>> seen;
        bool escapes = false;
        int surround = -1;               // -1 = none seen yet; -2 = multiple
        region_id[idx] = cur_region;
        while (!stack.empty()) {
          auto [qx,qy] = stack.back(); stack.pop_back();
          seen.emplace_back(qx,qy);
          for (int d = 0; d < 4; ++d) {
            int ux = qx + DX[d], uy = qy + DY[d];
            if ((unsigned)ux >= (unsigned)W || (unsigned)uy >= (unsigned)H) {
              escapes = true;
              continue;
            }
            size_t uidx = static_cast<size_t>(uy) * W + ux;
            int c2 = comp[uidx];
            if (c2 == 0) {
              if (region_id[uidx] < 0) { region_id[uidx] = cur_region; stack.emplace_back(ux,uy); }
              continue;
            }
            if (surround == -1) surround = c2;
            else if (surround != c2) surround = -2;
          }
        }
        ++cur_region;
        if (!escapes && surround >= 1)
          holes_of[surround].push_back(std::move(seen));
      }
  }

  int processed_cand = 0;
  for (int ci = 0; ci < max_cand_n; ++ci) {
    int lbl = order[ci];
    if (comp_pixel_count[lbl] < 1) continue;

    int sx = -1, sy = -1;
    if (!find_first_boundary(comp, W, H, lbl, &sx, &sy)) continue;
    std::vector<PointF> boundary = trace_boundary(comp, W, H, lbl, sx, sy);
    if (boundary.size() >= 4 && process_contour(boundary)) ++processed_cand;

    // ---- RETR_LIST parity: use precomputed hole regions -----------------
    auto hit = holes_of.find(lbl);
    if (hit == holes_of.end()) continue;
    bool emitted_hole_box = false;
    for (const auto& reg : hit->second) {
      std::vector<uint8_t> hole(W * H, 0);
      for (auto& s : reg) hole[static_cast<size_t>(s.second) * W + s.first] = 1;
      std::vector<PointF> ring = synthesize_hole_ring(comp, hole, W, H, lbl);
      if (ring.size() >= 4 && process_contour(ring)) {
        ++processed_cand;
        emitted_hole_box = true;
      }
    }
    (void)emitted_hole_box;
  }
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
