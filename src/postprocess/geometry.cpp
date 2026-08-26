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

// --- 3x3 inverse (Gauss-Jordan on the doubled matrix) ----------------------
// Used by warp_perspective_quad below. Single-purpose: invert one 3x3.
static bool invert3x3(const double m[9], double out[9]) {
  double a[3][3] = {{m[0], m[1], m[2]}, {m[3], m[4], m[5]}, {m[6], m[7], m[8]}};
  double inv[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int k = 0; k < 3; ++k) {
    int piv = k;
    double maxv = std::fabs(a[k][k]);
    for (int i = k + 1; i < 3; ++i) {
      if (std::fabs(a[i][k]) > maxv) { piv = i; maxv = std::fabs(a[i][k]); }
    }
    if (maxv < 1e-12) return false;
    if (piv != k) {
      for (int j = 0; j < 3; ++j) std::swap(a[k][j], a[piv][j]);
      for (int j = 0; j < 3; ++j) std::swap(inv[k][j], inv[piv][j]);
    }
    for (int i = 0; i < 3; ++i) {
      if (i == k) continue;
      double f = a[i][k] / a[k][k];
      for (int j = 0; j < 3; ++j) {
        a[i][j] -= f * a[k][j];
        inv[i][j] -= f * inv[k][j];
      }
    }
    double d = a[k][k];
    for (int j = 0; j < 3; ++j) { a[k][j] /= d; inv[k][j] /= d; }
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i * 3 + j] = inv[i][j];
  return true;
}

// --- perspective transform solver (8x8 Gauss-Jordan) -----------------------
// Computes M (3x3 row-major) such that [u_i, v_i, 1]^T ~ M * [x_i, y_i, 1]^T
// for i = 0..3. Matches cv2.getPerspectiveTransform bit-for-bit (same
// 8-equation form, same pivot ordering, double precision). Paddle's
// `get_rotate_crop_image` uses the same OpenCV call, so we mirror it here.
static bool get_perspective_transform(const double src[8], const double dst[8],
                                       double M[9]) {
  // 8 equations, 8 unknowns (a, b, c, d, e, f, g, h), with M[2,2] = 1.
  //   a*x_i + b*y_i + c - g*x_i*u_i - h*y_i*u_i = u_i
  //   d*x_i + e*y_i + f - g*x_i*v_i - h*y_i*v_i = v_i
  double A[8][8] = {{0}};
  double b[8] = {0};
  for (int i = 0; i < 4; ++i) {
    double x = src[i * 2], y = src[i * 2 + 1];
    double u = dst[i * 2], v = dst[i * 2 + 1];
    A[i][0] = x; A[i][1] = y; A[i][2] = 1;
    A[i][6] = -x * u; A[i][7] = -y * u;
    b[i] = u;
    A[i + 4][3] = x; A[i + 4][4] = y; A[i + 4][5] = 1;
    A[i + 4][6] = -x * v; A[i + 4][7] = -y * v;
    b[i + 4] = v;
  }
  // Augmented matrix M_aug[i][8] = b[i].
  double aug[8][9];
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) aug[i][j] = A[i][j];
    aug[i][8] = b[i];
  }
  for (int k = 0; k < 8; ++k) {
    int piv = k;
    double maxv = std::fabs(aug[k][k]);
    for (int i = k + 1; i < 8; ++i) {
      if (std::fabs(aug[i][k]) > maxv) { piv = i; maxv = std::fabs(aug[i][k]); }
    }
    if (maxv < 1e-15) return false;
    if (piv != k) for (int j = 0; j < 9; ++j) std::swap(aug[k][j], aug[piv][j]);
    for (int i = 0; i < 8; ++i) {
      if (i == k) continue;
      double f = aug[i][k] / aug[k][k];
      for (int j = k; j < 9; ++j) aug[i][j] -= f * aug[k][j];
    }
  }
  double x[8];
  for (int i = 0; i < 8; ++i) x[i] = aug[i][8] / aug[i][i];
  M[0] = x[0]; M[1] = x[1]; M[2] = x[2];
  M[3] = x[3]; M[4] = x[4]; M[5] = x[5];
  M[6] = x[6]; M[7] = x[7]; M[8] = 1.0;
  return true;
}

// --- bicubic (OpenCV INTER_CUBIC, a = -0.75) ---------------------------------
// Port of OpenCV's interpolateCubic: 3 explicit weights + 1 residual. The 4
// samples are at offsets -1, 0, 1, 2 from x0 (where x0 = floor(sample)). The
// residual guarantees partition-of-unity exactly (sum to 1.0) and matches
// OpenCV's rounding pattern bit-for-bit in most cases.
inline void cubic_weights(float fx, float w[4]) {
  const float A = -0.75f;
  // OpenCV's parameterization: x in [0, 1) is the fractional offset.
  // c[0] = kernel at |t| = x+1   (sample at x0-1)
  // c[1] = kernel at |t| = x     (sample at x0)
  // c[2] = kernel at |t| = 1-x   (sample at x0+1)
  // c[3] = 1 - c[0] - c[1] - c[2] (sample at x0+2)
  w[0] = ((A * (fx + 1.0f) - 5.0f * A) * (fx + 1.0f) + 8.0f * A) *
             (fx + 1.0f) -
         4.0f * A;
  w[1] = ((A + 2.0f) * fx - (A + 3.0f)) * fx * fx + 1.0f;
  w[2] = ((A + 2.0f) * (1.0f - fx) - (A + 3.0f)) * (1.0f - fx) *
             (1.0f - fx) +
         1.0f;
  w[3] = 1.0f - w[0] - w[1] - w[2];
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
  cubic_weights(fx, wx);
  cubic_weights(fy, wy);
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

  // Paddle's `get_rotate_crop_image` (paddlex 3.x,
  // paddlex/inference/pipelines/components/common/crop_image_regions.py)
  // builds the destination rectangle as `pts_std = [[0,0], [W,0],
  // [W,H], [0,H]]` with W = int(max(norm(p0-p1), norm(p2-p3))) and
  // H = int(max(norm(p0-p3), norm(p1-p2))). It then calls
  // `cv2.warpPerspective(img, M, (W, H), INTER_CUBIC, BORDER_REPLICATE)`.
  // The dst mapping therefore spans pixel indices [0, W-1] in i and
  // [0, H-1] in j, but the *range* of i/j fed into the affine
  // coefficients is [0, W] / [0, H], not [0, W-1] / [0, H-1]. Using
  // (W-1) here would scale the perspective by (W-1)/W and shift the
  // warp slightly.
  double w = static_cast<double>(dst_w);
  double h = static_cast<double>(dst_h);
  if (w < 1.0) w = 1.0;
  if (h < 1.0) h = 1.0;

  // POST-7: use a full 8-parameter perspective transform (cv2's
  // getPerspectiveTransform), not the 6-parameter affine approximation
  // that the previous version used. The affine form assumed the
  // source quad was a perfect parallelogram; for general quads (and
  // especially for vertical / rotated text where the top and bottom
  // edges are not parallel — e.g. ja/00 box0 with TL=(509,0),
  // TR=(550,0), BR=(543,213), BL=(502,212)) the affine map misses
  // the perspective component and the warped crop is shifted by up
  // to ~30 px in src coordinates. That systematic shift is what made
  // vert-text CER zero. Mirroring cv2's 8x8 linear solve restores
  // pixel-level parity.
  const double src_pts[8] = {
      q[0].x, q[0].y,  q[1].x, q[1].y,  q[2].x, q[2].y,  q[3].x, q[3].y,
  };
  const double dst_pts[8] = {0.0, 0.0,  w, 0.0,  w, h,  0.0, h};
  double M[9];
  if (!get_perspective_transform(src_pts, dst_pts, M)) {
    // Degenerate quad (collinear points etc.) — fall back to zero
    // image. Caller checks the returned image via .data.empty().
    return dst;
  }
  double M_inv[9];
  if (!invert3x3(M, M_inv)) {
    return dst;
  }
  // Pre-extract the inverse entries (M_inv * dst_h = src_h) for the
  // inner loop. Using doubles here keeps the per-pixel arithmetic
  // well-conditioned; we cast back to float when feeding the bicubic
  // sampler.
  const double m00 = M_inv[0], m01 = M_inv[1], m02 = M_inv[2];
  const double m10 = M_inv[3], m11 = M_inv[4], m12 = M_inv[5];
  const double m20 = M_inv[6], m21 = M_inv[7], m22 = M_inv[8];

  dst.w = dst_w;
  dst.h = dst_h;
  dst.c = src.c;
  dst.data.assign(static_cast<size_t>(dst_w) * dst_h * src.c, 0);
  int sstride = src.w * src.c;
  int dstride = dst_w * src.c;

  for (int j = 0; j < dst_h; ++j) {
    uint8_t* drow = dst.data.data() + j * dstride;
    for (int i = 0; i < dst_w; ++i) {
      // Inverse-warp: src_h = M_inv * (i, j, 1). The last row
      // carries the perspective denominator. With our (W, H)
      // corners convention, dst (0, 0) lands on src TL exactly
      // (M_inv[0,2] = TLx, M_inv[1,2] = TLy), matching cv2.
      const double w_denom = m20 * i + m21 * j + m22;
      const double sx = (m00 * i + m01 * j + m02) / w_denom;
      const double sy = (m10 * i + m11 * j + m12) / w_denom;
      for (int ch = 0; ch < src.c; ++ch) {
        // Pass ch_stride = src.c so the sampler indexes
        // (xs[i] * src.c + ch) per row, matching HWC layout.
        drow[i * src.c + ch] = sample_bicubic_replicate_plane(
            src.data.data() + ch, src.w, src.h, sstride, src.c,
            static_cast<float>(sx), static_cast<float>(sy));
      }
    }
  }
  return dst;
}

Image polar_unwrap(const Image& src, float cx, float cy, float r_max,
                   int dst_h, int dst_w) {
  Image dst;
  if (src.c <= 0 || src.w <= 0 || src.h <= 0 || dst_h <= 0 || dst_w <= 0) {
    return dst;
  }
  if (r_max <= 0.0f) r_max = 1.0f;
  const float two_pi = 6.283185307f;
  dst.w = dst_w;
  dst.h = dst_h;
  dst.c = src.c;
  dst.data.assign(static_cast<size_t>(dst_w) * dst_h * src.c, 0);
  const int W = src.w, H = src.h, C = src.c;
  for (int y = 0; y < dst_h; ++y) {
    // (y + 0.5) / dst_h * 2*pi -> theta in [0, 2pi), CCW from +x axis.
    const float theta = (static_cast<float>(y) + 0.5f) / dst_h * two_pi;
    const float cos_t = std::cos(theta);
    const float sin_t = std::sin(theta);
    uint8_t* drow = dst.data.data() + static_cast<size_t>(y) * dst_w * C;
    for (int x = 0; x < dst_w; ++x) {
      const float r = (static_cast<float>(x) + 0.5f) / dst_w * r_max;
      const float sx = cx + r * cos_t;
      const float sy = cy + r * sin_t;
      if (sx < 0.0f || sy < 0.0f || sx > W - 1.0f || sy > H - 1.0f) {
        continue;
      }
      const int ix = static_cast<int>(sx);
      const int iy = static_cast<int>(sy);
      const int ix1 = (ix + 1 < W) ? ix + 1 : ix;
      const int iy1 = (iy + 1 < H) ? iy + 1 : iy;
      const float dx = sx - static_cast<float>(ix);
      const float dy = sy - static_cast<float>(iy);
      const uint8_t* p00 = src.data.data() + (static_cast<size_t>(iy)  * W + ix)  * C;
      const uint8_t* p01 = src.data.data() + (static_cast<size_t>(iy)  * W + ix1) * C;
      const uint8_t* p10 = src.data.data() + (static_cast<size_t>(iy1) * W + ix)  * C;
      const uint8_t* p11 = src.data.data() + (static_cast<size_t>(iy1) * W + ix1) * C;
      const float w00 = (1.0f - dx) * (1.0f - dy);
      const float w01 = dx * (1.0f - dy);
      const float w10 = (1.0f - dx) * dy;
      const float w11 = dx * dy;
      uint8_t* dp = drow + static_cast<size_t>(x) * C;
      for (int ch = 0; ch < C; ++ch) {
        const float v = w00 * p00[ch] + w01 * p01[ch] + w10 * p10[ch] + w11 * p11[ch];
        dp[ch] = static_cast<uint8_t>(std::lround(v));
      }
    }
  }
  return dst;
}


Image polar_unwrap_band(const Image& src, float cx, float cy,
                        float r_inner, float r_outer,
                        int dst_h, int dst_w) {
  Image dst;
  if (src.c <= 0 || src.w <= 0 || src.h <= 0 || dst_h <= 0 || dst_w <= 0) {
    return dst;
  }
  if (r_inner < 0.0f) r_inner = 0.0f;
  if (r_outer <= r_inner) r_outer = r_inner + 1.0f;
  const float two_pi = 6.283185307f;
  const float r_span = r_outer - r_inner;
  dst.w = dst_w;
  dst.h = dst_h;
  dst.c = src.c;
  dst.data.assign(static_cast<size_t>(dst_w) * dst_h * src.c, 255);
  const int W = src.w, H = src.h, C = src.c;
  for (int y = 0; y < dst_h; ++y) {
    const float theta = (static_cast<float>(y) + 0.5f) / dst_h * two_pi;
    const float cos_t = std::cos(theta);
    const float sin_t = std::sin(theta);
    uint8_t* drow = dst.data.data() + static_cast<size_t>(y) * dst_w * C;
    for (int x = 0; x < dst_w; ++x) {
      const float r = r_inner + (static_cast<float>(x) + 0.5f) / dst_w * r_span;
      const float sx = cx + r * cos_t;
      const float sy = cy + r * sin_t;
      if (sx < 0.0f || sy < 0.0f || sx > W - 1.0f || sy > H - 1.0f) {
        continue;
      }
      const int ix = static_cast<int>(sx);
      const int iy = static_cast<int>(sy);
      const int ix1 = (ix + 1 < W) ? ix + 1 : ix;
      const int iy1 = (iy + 1 < H) ? iy + 1 : iy;
      const float dx = sx - static_cast<float>(ix);
      const float dy = sy - static_cast<float>(iy);
      const uint8_t* p00 = src.data.data() + (static_cast<size_t>(iy)  * W + ix)  * C;
      const uint8_t* p01 = src.data.data() + (static_cast<size_t>(iy)  * W + ix1) * C;
      const uint8_t* p10 = src.data.data() + (static_cast<size_t>(iy1) * W + ix)  * C;
      const uint8_t* p11 = src.data.data() + (static_cast<size_t>(iy1) * W + ix1) * C;
      const float w00 = (1.0f - dx) * (1.0f - dy);
      const float w01 = dx * (1.0f - dy);
      const float w10 = (1.0f - dx) * dy;
      const float w11 = dx * dy;
      uint8_t* dp = drow + static_cast<size_t>(x) * C;
      for (int c = 0; c < C; ++c) {
        const float v = w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
        dp[c] = static_cast<uint8_t>(v + 0.5f);
      }
    }
  }
  return dst;
}

}  // namespace ppocr
