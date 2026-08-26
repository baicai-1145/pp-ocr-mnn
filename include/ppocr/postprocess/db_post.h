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

// Reading-order sort for detected quad boxes (top-to-bottom, then left-to-right
// within each row, with a 10 px vertical tolerance for "same row"). This is
// the Paddle C++ pipeline's ComponentsProcessor::SortQuadBoxes behavior
// (deploy/cpp_infer/src/common/processors.cc:590), applied to the [TL,TR,BR,BL]
// DetBox.poly that db_postprocess emits. The pipeline layer should call this
// after db_postprocess so that rec_texts are in reading order (and CER drops).
//
// Faithful port:
//   1) std::sort by (a.poly[1] < b.poly[1]) || (== && a.poly[0] < b.poly[0])
//   2) Bubble pass: j from i+1 down to 1; if |boxes[j].poly[1] - boxes[j-1].poly[1]| < 10
//      && boxes[j].poly[0] < boxes[j-1].poly[0] then swap, else break.
//
// Empty / single-box input is a no-op.
void sort_quad_boxes_reading_order(std::vector<DetBox>& boxes);

} // namespace ppocr
#endif // PPOCR_POSTPROCESS_DB_POST_H_
