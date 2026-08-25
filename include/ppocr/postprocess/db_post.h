// pp-ocr-mnn — DB postprocess (postprocess module)
// Owner: post.
#ifndef PPOCR_POSTPROCESS_DB_POST_H_
#define PPOCR_POSTPROCESS_DB_POST_H_

#include <vector>

#include "ppocr/config.h"

namespace ppocr {

struct DetBox {
  float poly[8];  // x0,y0,...,x3,y3 in ORIGINAL image coords
  float score = 0;
};

// Faithful port of ppocr/data/postprocess/db_postprocess.py boxes_from_bitmap +
// filter_tag_det_boxes + get_sorted_boxes. Uses RETR_LIST external contours
// (matching the Paddle reference; see cpp_infer processors.cc — RETR_LIST not
// RETR_CCOMP). Dilation off by default. Score mode = "fast" (mean of pred
// inside the polygon mask). Unclip via ClipperOffset JT_ROUND ET_CLOSEDPOLYGON,
// distance = area * unclip_ratio / perimeter.
std::vector<DetBox> db_postprocess(const float* prob, int prob_h, int prob_w,
                                   int src_w, int src_h, float ratio_w,
                                   float ratio_h, const DetConfig& cfg);

}  // namespace ppocr

#endif  // PPOCR_POSTPROCESS_DB_POST_H_
