// pp-ocr-mnn — geometry helpers (owner post)
//
// Minimal port of the geometry primitives PaddleOCR uses during DB
// postprocess and rec crop warping. Real implementations land on
// branch ws/post; this header is the public contract every branch
// must already satisfy, so ws/m1 ships it as a forward declaration.
#ifndef PPOCR_POSTPROCESS_GEOMETRY_H_
#define PPOCR_POSTPROCESS_GEOMETRY_H_

#include <cstddef>
#include "ppocr/image.h"

namespace ppocr {

struct PointF { float x = 0, y = 0; };

// min-area rect via rotating calipers; out points ordered like
// cv::minAreaRect().points(). Returns false if n < 3.
bool min_area_rect(const PointF* pts, size_t n, PointF out[4]);

// Re-order 4 box points into PaddleOCR's canonical clockwise-from-tl order.
// Port of order_points_clockwise + GetMinAreaRectPoints from
// deploy/cpp_infer/src/common/processors.cc.
void sort_min_area_rect_points(PointF box[4]);

// Inverse-map bicubic warp with border-replicate, port of
// cv::warpPerspective INTER_CUBIC BORDER_REPLICATE. `src` is the source
// image; `quad` defines the source quadrilateral mapped onto the
// destination rectangle [0..dst_w-1] x [0..dst_h-1].
Image warp_perspective_quad(const Image& src, const PointF quad[4],
                            int dst_w, int dst_h);

// Polar unwrap: sample (src, cx, cy) at radius r in [0, r_max],
// into a (dst_h, dst_w) image where y -> theta, x -> r.
Image polar_unwrap(const Image& src, float cx, float cy, float r_max,
                   int dst_h, int dst_w);

// Polar unwrap: sample a band [r_inner, r_outer] of (src, cx, cy)
// into a (dst_h, dst_w) image, where y -> theta, x -> r in band.
// Pixels outside the band are set to white (255).
Image polar_unwrap_band(const Image& src, float cx, float cy,
                        float r_inner, float r_outer,
                        int dst_h, int dst_w);

} // namespace ppocr
#endif // PPOCR_POSTPROCESS_GEOMETRY_H_
