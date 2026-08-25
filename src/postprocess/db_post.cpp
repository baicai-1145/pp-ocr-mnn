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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ppocr/postprocess/geometry.h"

// Clipper (header-only, vendored). See third_party/clipper/clipper.hpp.
#include "clipper.hpp"

namespace ppocr {
namespace {

// Paddle's DBPostProcess.min_size — boxes with a shorter side below this are
// discarded before unclip.
constexpr int kMinSize = 3;

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

// --- Douglas-Peucker polyline simplification --------------------------------
void dp_simplify(std::vector<PointF>& pts, float eps) {
  if (pts.size() < 3 || eps <= 0.0f) return;
  std::vector<uint8_t> keep(pts.size(), 0);
  keep.front() = 1;
  keep.back() = 1;
  // Iterative stack to avoid recursion depth issues on long contours.
  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(0, static_cast<int>(pts.size()) - 1);
  while (!stack.empty()) {
    auto [a, b] = stack.back();
    stack.pop_back();
    if (b - a < 2) continue;
    const PointF& p1 = pts[a];
    const PointF& p2 = pts[b];
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    float norm = std::sqrt(dx * dx + dy * dy);
    float max_d = 0.0f;
    int max_i = -1;
    for (int i = a + 1; i < b; ++i) {
      const PointF& p = pts[i];
      float d;
      if (norm < 1e-6f) {
        d = std::sqrt((p.x - p1.x) * (p.x - p1.x) +
                      (p.y - p1.y) * (p.y - p1.y));
      } else {
        d = std::fabs(dy * p.x - dx * p.y + p2.x * p1.y - p2.y * p1.x) / norm;
      }
      if (d > max_d) {
        max_d = d;
        max_i = i;
      }
    }
    if (max_i >= 0 && max_d > eps) {
      keep[max_i] = 1;
      stack.emplace_back(a, max_i);
      stack.emplace_back(max_i, b);
    }
  }
  std::vector<PointF> out;
  out.reserve(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    if (keep[i]) out.push_back(pts[i]);
  }
  pts.swap(out);
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
// Standard scanline fill. Returns a mask of size WxH (row-major, 0/1).
std::vector<uint8_t> fill_polygon_mask(const std::vector<PointF>& poly_in, int W,
                                      int H) {
  std::vector<uint8_t> mask(W * H, 0);
  if (poly_in.size() < 3) return mask;
  // Build a closed integer polygon. Clip to image bounds.
  std::vector<std::pair<int, int>> p;
  p.reserve(poly_in.size() + 1);
  for (const auto& pt : poly_in) {
    int x = static_cast<int>(std::lround(pt.x));
    int y = static_cast<int>(std::lround(pt.y));
    if (x < 0) x = 0;
    if (x > W - 1) x = W - 1;
    if (y < 0) y = 0;
    if (y > H - 1) y = H - 1;
    p.emplace_back(x, y);
  }
  // Remove consecutive duplicates.
  std::vector<std::pair<int, int>> p2;
  p2.reserve(p.size() + 1);
  for (size_t i = 0; i < p.size(); ++i) {
    size_t j = (i + 1) % p.size();
    if (p[i].first == p[j].first && p[i].second == p[j].second) continue;
    p2.push_back(p[i]);
  }
  if (p2.size() < 3) return mask;
  p2.push_back(p2.front());
  int ymin = H, ymax = -1;
  for (const auto& q : p2) {
    if (q.second < ymin) ymin = q.second;
    if (q.second > ymax) ymax = q.second;
  }
  if (ymin < 0) ymin = 0;
  if (ymax > H - 1) ymax = H - 1;
  for (int y = ymin; y <= ymax; ++y) {
    std::vector<int> xs;
    for (size_t i = 0; i + 1 < p2.size(); ++i) {
      int y1 = p2[i].second, y2 = p2[i + 1].second;
      int x1 = p2[i].first, x2 = p2[i + 1].first;
      if ((y1 <= y && y2 > y) || (y2 <= y && y1 > y)) {
        double xi = x1 + (static_cast<double>(y - y1) / (y2 - y1)) * (x2 - x1);
        xs.push_back(static_cast<int>(std::lround(xi)));
      }
    }
    if (xs.size() < 2) continue;
    std::sort(xs.begin(), xs.end());
    for (size_t k = 0; k + 1 < xs.size(); k += 2) {
      int xa = xs[k];
      int xb = xs[k + 1];
      if (xa < 0) xa = 0;
      if (xb > W - 1) xb = W - 1;
      if (xa > xb) continue;
      uint8_t* row = mask.data() + y * W;
      std::memset(row + xa, 1, xb - xa + 1);
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
  // First pass: assign provisional labels.
  std::vector<int> label(W * H, 0);
  int next_label = 1;
  std::vector<int> nbr_offsets;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      nbr_offsets.push_back(dy * W + dx);
    }
  }
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (!mask[y * W + x]) continue;
      int my = next_label++;
      std::vector<int> neighbors;
      for (int off : nbr_offsets) {
        int yy = y + off / W;
        int xx = x + off % W;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        if (label[yy * W + xx] > 0) neighbors.push_back(label[yy * W + xx]);
      }
      if (neighbors.empty()) {
        label[y * W + x] = my;
      } else {
        int min_lbl = *std::min_element(neighbors.begin(), neighbors.end());
        label[y * W + x] = min_lbl;
        for (int n : neighbors) uf.unite(min_lbl, n);
      }
    }
  }
  // Second pass: flatten labels.
  std::vector<int> comp(W * H, 0);
  std::vector<int> comp_size;
  comp_size.push_back(0);  // index 0 unused
  for (int i = 0; i < W * H; ++i) {
    if (label[i] == 0) continue;
    int root = uf.find(label[i]);
    int idx;
    auto it = std::find(comp_size.begin() + 1, comp_size.end(), root);
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

  for (int ci = 0; ci < max_cand_n; ++ci) {
    int lbl = order[ci];
    if (comp_pixel_count[lbl] < 1) continue;

    int sx = -1, sy = -1;
    if (!find_first_boundary(comp, W, H, lbl, &sx, &sy)) {
      continue;
    }
    std::vector<PointF> boundary = trace_boundary(comp, W, H, lbl, sx, sy);
    if (boundary.size() < 4) continue;

    // Simplify with DP using epsilon = 0.002 * perimeter (Paddle uses
    // approxPolyDP with this formula).
    float perim = polygon_perimeter(boundary);
    dp_simplify(boundary, 0.002f * perim);
    if (boundary.size() < 4) continue;

    // Compute mini-box (minAreaRect -> sort_min_area_rect_points). Paddle's
    // get_mini_boxes also returns the short side length for sside filter.
    std::vector<PointF> pts(boundary.begin(), boundary.end());
    PointF box4[4];
    if (!min_area_rect(pts.data(), pts.size(), box4)) continue;
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
    if (xmax < xmin || ymax < ymin) continue;
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
    if (sum_count == 0) continue;
    float score = static_cast<float>(sum_prob / sum_count);
    if (cfg.box_thresh > score) continue;

    // Unclip: distance = area * unclip_ratio / perimeter.
    float area = polygon_area(sorted_box4);
    if (area <= 0.0f) continue;
    float perim_b = polygon_perimeter(sorted_box4);
    if (perim_b <= 0.0f) continue;
    float distance = area * cfg.unclip_ratio / perim_b;

    // Build clipper path. Paddle's pyclipper (and Paddle's C++ unclip
    // implementation) takes the path and distance in image coords directly;
    // no SCALE=10000 multiplication is needed. (The SCALE factor is only
    // used by CropByPolys for IoU computation with Clipper, which uses
    // integer arithmetic and is sensitive to roundoff.)
    ClipperLib::Path path;
    path.reserve(sorted_box4.size());
    for (const auto& p : sorted_box4) {
      path << ClipperLib::IntPoint(static_cast<ClipperLib::cInt>(std::llround(p.x)),
                                   static_cast<ClipperLib::cInt>(std::llround(p.y)));
    }
    ClipperLib::ClipperOffset co;
    co.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    ClipperLib::Paths solution;
    co.Execute(solution, distance);
    if (solution.size() != 1) continue;  // Paddle: "if len(box) > 1: continue"

    std::vector<PointF> expanded;
    expanded.reserve(solution[0].size());
    for (const auto& ip : solution[0]) {
      expanded.push_back({static_cast<float>(ip.X), static_cast<float>(ip.Y)});
    }
    if (expanded.size() < 4) continue;

    // Re-mini-box and sort.
    PointF box2[4];
    if (!min_area_rect(expanded.data(), expanded.size(), box2)) continue;
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
    if (sside < static_cast<float>(kMinSize + 2)) continue;

    // Map bitmap coords -> original image coords. Paddle: x = round(bx / W *
    // dest_w); y = round(by / H * dest_h). ratio_w = dest_w / W, ratio_h =
    // dest_h / H, so equivalently bx * ratio_w, by * ratio_h.
    DetBox db;
    for (int k = 0; k < 4; ++k) {
      float ox = final_box[k].x * ratio_w;
      float oy = final_box[k].y * ratio_h;
      // Clip to [0, src].
      if (ox < 0.0f) ox = 0.0f;
      if (oy < 0.0f) oy = 0.0f;
      if (ox > src_w - 1) ox = static_cast<float>(src_w - 1);
      if (oy > src_h - 1) oy = static_cast<float>(src_h - 1);
      db.poly[k * 2 + 0] = std::round(ox);
      db.poly[k * 2 + 1] = std::round(oy);
    }
    db.score = score;
    out.push_back(db);
  }
  return out;
}

}  // namespace ppocr
