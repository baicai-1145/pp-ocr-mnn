// pp-ocr-mnn — DB postprocess (owner post)
//
// Faithful C++ port of ppocr/data/postprocess/db_postprocess.py
// (boxes_from_bitmap + filter_tag_det_boxes + get_sorted_boxes). Boxes
// returned in official reading order; coordinates are in the original
// image's pixel space (already multiplied by 1/ratio_w, 1/ratio_h).
//
// Real implementation lives on branch ws/post. This header freezes the
// ABI so ws/m1 can compile against the same signature.
#ifndef PPOCR_POSTPROCESS_DB_POST_H_
#define PPOCR_POSTPROCESS_DB_POST_H_

#include <vector>
#include "ppocr/config.h"

namespace ppocr {

struct DetBox {
  float poly[8];   // x0,y0, x1,y1, x2,y2, x3,y3  (original-image coords)
  float score = 0; // mean probability of the box interior
};

std::vector<DetBox> db_postprocess(const float* prob, int prob_h, int prob_w,
                                   int src_w, int src_h,
                                   float ratio_w, float ratio_h,
                                   const DetConfig& cfg);

} // namespace ppocr
#endif // PPOCR_POSTPROCESS_DB_POST_H_
