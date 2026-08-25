// pp-ocr-mnn — geometry helpers (postprocess module)
// Owner: post.
// All declarations frozen by docs/CONTRACT.md. Implementations in src/postprocess/geometry.cpp.
#ifndef PPOCR_POSTPROCESS_GEOMETRY_H_
#define PPOCR_POSTPROCESS_GEOMETRY_H_

#include <cstddef>

#include "ppocr/image.h"

namespace ppocr {

struct PointF {
  float x = 0;
  float y = 0;
};

// Minimum-area bounding rectangle of a 2D point cloud via rotating calipers on
// the convex hull. Output 4 points follow cv::minAreaRect().points convention
// (an arbitrary starting corner, then the next two along the edges).
// Returns false on degenerate input (n<3 or zero area).
bool min_area_rect(const PointF* pts, size_t n, PointF out[4]);

// Re-order 4 box points to PaddleOCR canonical reading order:
//   [top-left, top-right, bottom-right, bottom-left].
// Port of order_points_clockwise + GetMinAreaRectPoints from
// /root/PaddleOCR/deploy/cpp_infer/src/common/processors.cc.
void sort_min_area_rect_points(PointF box[4]);

// Inverse-mapped bicubic warp with BORDER_REPLICATE, matching
// cv::warpPerspective(..., INTER_CUBIC, BORDER_REPLICATE) semantics:
// the source quad (in any order — sorted internally) is mapped to the
// destination rectangle [0,0]-[dst_w-1, dst_h-1].
Image warp_perspective_quad(const Image& src, const PointF quad[4],
                            int dst_w, int dst_h);

}  // namespace ppocr

#endif  // PPOCR_POSTPROCESS_GEOMETRY_H_
