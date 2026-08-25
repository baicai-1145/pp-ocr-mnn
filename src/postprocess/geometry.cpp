// pp-ocr-mnn — geometry helpers implementation.
// Owner: post. No OpenCV; pure C++17. No platform ifdefs.
#include "ppocr/postprocess/geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ppocr {
namespace {

// --- convex hull (Andrew's monotone chain) ----------------------------------
std::vector<PointF> convex_hull(std::vector<PointF> pts) {
  std::sort(pts.begin(), pts.end(), [](const PointF& a, const PointF& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  pts.erase(std::unique(pts.begin(), pts.end(),
                        [](const PointF& a, const PointF& b) {
                          return a.x == b.x && a.y == b.y;
                        }),
            pts.end());
  const int n = static_cast<int>(pts.size());
  if (n <= 1) return pts;
  std::vector<PointF> h(2 * n);
  int k = 0;
  auto cross = [](const PointF& o, const PointF& a, const PointF& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  };
  for (int i = 0; i < n; ++i) {
    while (k >= 2 && cross(h[k - 2], h[k - 1], pts[i]) <= 0.0f) --k;
    h[k++] = pts[i];
  }
  for (int i = n - 2, t = k + 1; i >= 0; --i) {
    while (k >= t && cross(h[k - 2], h[k - 1], pts[i]) <= 0.0f) --k;
    h[k++] = pts[i];
  }
  h.resize(k - 1);
  return h;
}

// --- rotating calipers --------------------------------------------------------
// Compute the minimum-area enclosing rectangle of a convex polygon (cw or ccw).
// Returns 4 corners in (u_min, v_min), (u_max, v_min), (u_max, v_max), (u_min,
// v_max) world coords, where (u, v) is the basis derived from the chosen edge.
// Equivalent to cv::minAreaRect(points).points convention.
bool min_area_rect_convex(const std::vector<PointF>& h, PointF out[4],
                          float* out_area) {
  const int n = static_cast<int>(h.size());
  if (n < 3) return false;

  // Pre-compute edge unit vectors and lengths.
  std::vector<float> edx(n), edy(n);
  for (int i = 0; i < n; ++i) {
    int j = (i + 1) % n;
    float dx = h[j].x - h[i].x;
    float dy = h[j].y - h[i].y;
    float L = std::sqrt(dx * dx + dy * dy);
    if (L > 0.0f) {
      edx[i] = dx / L;
      edy[i] = dy / L;
    } else {
      edx[i] = 0.0f;
      edy[i] = 0.0f;
    }
  }

  // Try every edge direction; the optimum has one side collinear with an edge.
  float best_area = std::numeric_limits<float>::infinity();
  float b_ux = 0, b_uy = 0, b_vx = 0, b_vy = 0;
  float b_w = 0, b_h = 0, b_cx = 0, b_cy = 0;

  for (int i = 0; i < n; ++i) {
    float ux = edx[i];
    float uy = edy[i];
    float vx = -uy;
    float vy = ux;

    float u_min = h[0].x * ux + h[0].y * uy;
    float u_max = u_min;
    float v_min = h[0].x * vx + h[0].y * vy;
    float v_max = v_min;
    for (int k = 1; k < n; ++k) {
      float pu = h[k].x * ux + h[k].y * uy;
      float pv = h[k].x * vx + h[k].y * vy;
      if (pu < u_min) u_min = pu;
      if (pu > u_max) u_max = pu;
      if (pv < v_min) v_min = pv;
      if (pv > v_max) v_max = pv;
    }
    float w = u_max - u_min;
    float hh = v_max - v_min;
    float area = w * hh;
    if (area < best_area) {
      best_area = area;
      b_ux = ux;
      b_uy = uy;
      b_vx = vx;
      b_vy = vy;
      b_w = w;
      b_h = hh;
      b_cx = u_min * ux + v_min * vx;
      b_cy = u_min * uy + v_min * vy;
    }
  }

  if (!std::isfinite(best_area) || best_area <= 0.0f) return false;

  float cu[4] = {0.0f, b_w, b_w, 0.0f};
  float cv[4] = {0.0f, 0.0f, b_h, b_h};
  for (int k = 0; k < 4; ++k) {
    out[k].x = b_cx + cu[k] * b_ux + cv[k] * b_vx;
    out[k].y = b_cy + cu[k] * b_uy + cv[k] * b_vy;
  }
  if (out_area) *out_area = best_area;
  return true;
}

// --- bicubic (OpenCV INTER_CUBIC, a = -0.75) ---------------------------------
inline float cubic_weight(float t) {
  const float a = -0.75f;
  float at = std::fabs(t);
  if (at < 1.0f) {
    return ((a + 2.0f) * at - (a + 3.0f)) * at * at + 1.0f;
  }
  if (at < 2.0f) {
    return ((a * at - 5.0f * a) * at + 8.0f * a) * at - 4.0f * a;
  }
  return 0.0f;
}

inline uint8_t sample_bicubic_replicate_plane(const uint8_t* plane, int w, int h,
                                               int stride, int ch_stride, float x,
                                               float y) {
  int x0 = static_cast<int>(std::floor(x));
  int y0 = static_cast<int>(std::floor(y));
  float fx = x - x0;
  float fy = y - y0;
  int xs[4], ys[4];
  for (int i = 0; i < 4; ++i) {
    int xx = x0 + (i - 1);
    int yy = y0 + (i - 1);
    if (xx < 0) xx = 0;
    if (xx > w - 1) xx = w - 1;
    if (yy < 0) yy = 0;
    if (yy > h - 1) yy = h - 1;
    xs[i] = xx;
    ys[i] = yy;
  }
  float wx[4], wy[4];
  for (int i = 0; i < 4; ++i) {
    wx[i] = cubic_weight(fx - (i - 1));
    wy[i] = cubic_weight(fy - (i - 1));
  }
  float sum = 0.0f;
  for (int j = 0; j < 4; ++j) {
    float row_w = 0.0f;
    const uint8_t* row = plane + ys[j] * stride;
    for (int i = 0; i < 4; ++i) {
      // Each pixel is `ch_stride` bytes wide; sample the channel at offset
      // xs[i] * ch_stride.
      row_w += static_cast<float>(row[xs[i] * ch_stride]) * wx[i];
    }
    sum += row_w * wy[j];
  }
  if (sum < 0.0f) sum = 0.0f;
  if (sum > 255.0f) sum = 255.0f;
  return static_cast<uint8_t>(sum + 0.5f);
}

}  // namespace

bool min_area_rect(const PointF* pts, size_t n, PointF out[4]) {
  if (!pts || n < 3 || !out) return false;
  std::vector<PointF> v(pts, pts + n);
  auto hull = convex_hull(std::move(v));
  if (hull.size() < 3) return false;
  float area = 0;
  return min_area_rect_convex(hull, out, &area);
}

void sort_min_area_rect_points(PointF box[4]) {
  // PaddleOCR canonical order (port of order_points_clockwise +
  // GetMinAreaRectPoints in deploy/cpp_infer/src/common/processors.cc):
  //   sort by x (asc), split into two pairs of two;
  //   per pair, the point with smaller y is the "top" corner.
  //   Output: [top-left, top-right, bottom-right, bottom-left].
  std::sort(box, box + 4, [](const PointF& a, const PointF& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  size_t index_a = 0, index_d = 1;
  if (box[1].y > box[0].y) {
    index_a = 0;
    index_d = 1;
  } else {
    index_a = 1;
    index_d = 0;
  }
  size_t index_b = 2, index_c = 3;
  if (box[3].y > box[2].y) {
    index_b = 2;
    index_c = 3;
  } else {
    index_b = 3;
    index_c = 2;
  }
  PointF sorted[4] = {box[index_a], box[index_b], box[index_c], box[index_d]};
  for (int k = 0; k < 4; ++k) box[k] = sorted[k];
}

Image warp_perspective_quad(const Image& src, const PointF quad_in[4],
                            int dst_w, int dst_h) {
  Image dst;
  if (src.c <= 0 || src.w <= 0 || src.h <= 0 || dst_w <= 0 || dst_h <= 0 ||
      !quad_in) {
    return dst;
  }
  if (src.c != 1 && src.c != 3 && src.c != 4) {
    return dst;
  }
  // Normalize quad to Paddle canonical order: TL, TR, BR, BL.
  PointF q[4];
  for (int k = 0; k < 4; ++k) q[k] = quad_in[k];

  float TLx = q[0].x, TLy = q[0].y;
  float TRx = q[1].x, TRy = q[1].y;
  float BLx = q[3].x, BLy = q[3].y;
  float w = static_cast<float>(dst_w - 1);
  float h = static_cast<float>(dst_h - 1);
  if (w < 1.0f) w = 1.0f;
  if (h < 1.0f) h = 1.0f;

  // For an axis-aligned rectangle in dst, the warp is affine:
  //   x(i,j) = a*i + b*j + c
  //   y(i,j) = d*i + e*j + f
  // Boundary conditions: (0,0)->TL, (w,0)->TR, (0,h)->BL.
  float a = (TRx - TLx) / w;
  float b = (BLx - TLx) / h;
  float c = TLx;
  float d = (TRy - TLy) / w;
  float e = (BLy - TLy) / h;
  float f = TLy;

  dst.w = dst_w;
  dst.h = dst_h;
  dst.c = src.c;
  dst.data.assign(static_cast<size_t>(dst_w) * dst_h * src.c, 0);
  int sstride = src.w * src.c;
  int dstride = dst_w * src.c;

  for (int j = 0; j < dst_h; ++j) {
    uint8_t* drow = dst.data.data() + j * dstride;
    for (int i = 0; i < dst_w; ++i) {
      float sx = a * i + b * j + c;
      float sy = d * i + e * j + f;
      for (int ch = 0; ch < src.c; ++ch) {
        // Pass ch_stride = src.c so the sampler indexes
        // (xs[i] * src.c + ch) per row, matching HWC layout.
        drow[i * src.c + ch] = sample_bicubic_replicate_plane(
            src.data.data() + ch, src.w, src.h, sstride, src.c, sx, sy);
      }
    }
  }
  return dst;
}

}  // namespace ppocr
