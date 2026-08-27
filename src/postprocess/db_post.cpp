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

// M3-PERF4: bbox-restricted rasterization, bit-exact with
// fill_polygon_mask restricted to the polygon's clipped-vertex bbox:
// identical integer polygon (same lround + clip), identical scanline
// crossings (same even-odd test and lround(xi)), identical span writes
// (memset of [xa, xb] inclusive) — only the buffer is bw x bh instead
// of W x H, and spans are offset by (x0, y0). The caller iterates the
// same bbox rows/columns, so scored pixels and accumulation order are
// unchanged. This avoids a full W*H zero-init per candidate (the
// dominant cost of box_score_fast on dense maps: 384 candidates x 1.2MB).
struct FilledPolyBBox {
  std::vector<uint8_t> mask;  // bw x bh, row-major
  int x0 = 0, y0 = 0, bw = 0, bh = 0;
};

static FilledPolyBBox fill_polygon_mask_bbox(const std::vector<PointF>& poly_in,
                                             int W, int H) {
  FilledPolyBBox fp;
  if (poly_in.size() < 3) return fp;
  std::vector<std::pair<int, int>> p;
  p.reserve(poly_in.size() + 1);
  int pxmin = W, pxmax = 0, pymin = H, pymax = 0;
  for (const auto& pt : poly_in) {
    int x = static_cast<int>(std::lround(pt.x));
    int y = static_cast<int>(std::lround(pt.y));
    if (x < 0) x = 0;
    if (x > W - 1) x = W - 1;
    if (y < 0) y = 0;
    if (y > H - 1) y = H - 1;
    p.emplace_back(x, y);
    if (x < pxmin) pxmin = x;
    if (x > pxmax) pxmax = x;
    if (y < pymin) pymin = y;
    if (y > pymax) pymax = y;
  }
  // Remove consecutive duplicates.
  std::vector<std::pair<int, int>> p2;
  p2.reserve(p.size() + 1);
  for (size_t i = 0; i < p.size(); ++i) {
    size_t j = (i + 1) % p.size();
    if (p[i].first == p[j].first && p[i].second == p[j].second) continue;
    p2.push_back(p[i]);
  }
  if (p2.size() < 3) return fp;
  p2.push_back(p2.front());
  fp.x0 = pxmin;
  fp.y0 = pymin;
  fp.bw = pxmax - pxmin + 1;
  fp.bh = pymax - pymin + 1;
  if (fp.bw <= 0 || fp.bh <= 0) { fp.bw = fp.bh = 0; return fp; }
  fp.mask.assign(static_cast<size_t>(fp.bw) * fp.bh, 0);
  for (int y = pymin; y <= pymax; ++y) {
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
    const int row = y - fp.y0;
    uint8_t* rowp = fp.mask.data() + static_cast<size_t>(row) * fp.bw;
    for (size_t k = 0; k + 1 < xs.size(); k += 2) {
      int xa = xs[k];
      int xb = xs[k + 1];
      if (xa < 0) xa = 0;
      if (xb > W - 1) xb = W - 1;
      if (xa > xb) continue;
      std::memset(rowp + (xa - fp.x0), 1, xb - xa + 1);
    }
  }
  return fp;
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
  // M3-PERF2: neighbor offsets precomputed ONCE (dy*W+dx flat, as
  // before). NOTE: the flat-offset arithmetic below is deliberately
  // preserved bit-for-bit from the original code — off/W and off%W use
  // C truncation semantics, which for dy=-1 folds those three
  // "up-row" neighbors onto the SAME row (a latent quirk the M2
  // matrix was validated against). "Fixing" it to true
  // 8-connectivity changes the component merge order and shifts
  // boxes on dense maps (en/08: 302 -> 301 lines) — do NOT change
  // without re-running the full CER matrix.
  int nbr_offsets[8];
  int n_off = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0) continue;
      nbr_offsets[n_off++] = dy * W + dx;
    }
  }
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (!mask[y * W + x]) continue;
      int my = next_label++;
      // M3-PERF2: fixed 8-slot stack array instead of a heap vector
      // per pixel (this loop runs over every text pixel; the malloc
      // churn was measurable on dense maps). Same semantics as the
      // original: neighbors collected in the same offset order, min
      // chosen with the same first-min tie-breaking.
      int neighbors[8];
      int n_nbr = 0;
      for (int k = 0; k < n_off; ++k) {
        const int off = nbr_offsets[k];
        int yy = y + off / W;
        int xx = x + off % W;
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const int l = label[yy * W + xx];
        if (l > 0) neighbors[n_nbr++] = l;
      }
      if (n_nbr == 0) {
        label[y * W + x] = my;
      } else {
        int min_lbl = neighbors[0];
        for (int k = 1; k < n_nbr; ++k) {
          if (neighbors[k] < min_lbl) min_lbl = neighbors[k];
        }
        label[y * W + x] = min_lbl;
        for (int k = 0; k < n_nbr; ++k) uf.unite(min_lbl, neighbors[k]);
      }
    }
  }
  // Second pass: flatten labels.
  // M3-PERF2: root->component-index lookup was a linear std::find over
  // comp_size per pixel (O(labels) per pixel, O(n^2) overall — the
  // single hottest line on dense maps). Replace with a direct
  // label-indexed vector built in the same order (first-seen root gets
  // the next index), which preserves the EXACT same component ids.
  std::vector<int> comp(W * H, 0);
  std::vector<int> root_to_idx(next_label, 0);  // labels are 1..next_label-1
  std::vector<int> comp_size;
  comp_size.push_back(0);  // index 0 unused
  for (int i = 0; i < W * H; ++i) {
    if (label[i] == 0) continue;
    const int root = uf.find(label[i]);
    int idx = root_to_idx[root];
    if (idx == 0) {
      comp_size.push_back(root);
      idx = static_cast<int>(comp_size.size()) - 1;
      root_to_idx[root] = idx;
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

  // M3-PERF2: one row-major sweep collecting each component's first
  // boundary pixel (same 4-neighbor border test as
  // find_first_boundary). Replaces the per-component full-map scan
  // (O(n_comp x W x H) — 95 ms of the 101 ms db_post on the 283-box
  // en/08 map) with a single O(W x H) pass. The first boundary pixel
  // found per label in row-major order is identical to what
  // find_first_boundary returned, so seeds (and thus boundaries, boxes
  // and scores) are bit-identical.
  std::vector<int> first_bx(n_comp + 1, -1), first_by(n_comp + 1, -1);
  {
    int remaining = n_comp;
    for (int y = 0; y < H && remaining > 0; ++y) {
      const int rowoff = y * W;
      for (int x = 0; x < W && remaining > 0; ++x) {
        const int lbl = comp[rowoff + x];
        if (lbl <= 0 || first_bx[lbl] >= 0) continue;
        bool is_border = false;
        if (x == 0 || x == W - 1 || y == 0 || y == H - 1) {
          is_border = true;
        } else {
          if (comp[rowoff + x - 1] != lbl ||
              comp[rowoff + x + 1] != lbl ||
              comp[(y - 1) * W + x] != lbl ||
              comp[(y + 1) * W + x] != lbl) {
            is_border = true;
          }
        }
        if (is_border) {
          first_bx[lbl] = x;
          first_by[lbl] = y;
          --remaining;
        }
      }
    }
  }

  for (int ci = 0; ci < max_cand_n; ++ci) {
    int lbl = order[ci];
    if (comp_pixel_count[lbl] < 1) continue;

    int sx = -1, sy = -1;
    if (first_bx[lbl] < 0) {
      continue;
    }
    sx = first_bx[lbl];
    sy = first_by[lbl];
    std::vector<PointF> boundary = trace_boundary(comp, W, H, lbl, sx, sy);
    if (boundary.size() < 4) continue;

    // Note: Paddle's boxes_from_bitmap (box_type="quad", the default used by
    // PaddleOCR 3.x paddlex pipeline) does NOT apply approxPolyDP. It feeds
    // the raw contour from cv::findContours directly into GetMiniBoxes. We
    // do the same: skip dp_simplify here. (ppocr's polygons_from_bitmap
    // path uses approxPolyDP, but that's the poly box_type we don't emit.)

    // Compute mini-box (minAreaRect -> sort_min_area_rect_points). Paddle's
    // get_mini_boxes also returns the short side length for sside filter.
    std::vector<PointF> pts(boundary.begin(), boundary.end());
    PointF box4[4];
    if (!min_area_rect(pts.data(), pts.size(), box4)) continue;
    sort_min_area_rect_points(box4);
    // Convert to flat poly (8 floats) for box_score_fast.
    std::vector<PointF> sorted_box4(box4, box4 + 4);

    // box_score_fast: mean of pred inside the polygon.
    // M3-PERF4: rasterize into the polygon bbox (bit-exact; see
    // fill_polygon_mask_bbox). The outer bbox loop bounds below are the
    // polygon's floor()-bbox as before; the mask covers the (smaller)
    // lround()-clipped bbox, and reads outside it are zero, matching the
    // old full-image mask semantics exactly.
    const FilledPolyBBox fp = fill_polygon_mask_bbox(sorted_box4, W, H);
    if (fp.bw <= 0 || fp.bh <= 0) continue;
    // Bounding box of the polygon (floor, as before).
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
      const int my = yy - fp.y0;
      const uint8_t* mrow =
          (my >= 0 && my < fp.bh) ? fp.mask.data() + static_cast<size_t>(my) * fp.bw
                                  : nullptr;
      const float* prow = prob + yy * W;
      for (int xx = xmin; xx <= xmax; ++xx) {
        const int mx = xx - fp.x0;
        if (mrow && mx >= 0 && mx < fp.bw && mrow[mx]) {
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
    if (sside < static_cast<float>(cfg.min_size + 2)) continue;

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
