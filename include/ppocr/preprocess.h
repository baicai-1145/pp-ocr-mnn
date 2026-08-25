// pp-ocr-mnn — preprocessing (owner m1)
//
// Mirrors PaddleOCR's data-augmentation operators verbatim. Each helper
// returns CHW float32 tensors ready to be fed into an MNN tensor via
// host()->write(). No allocations hidden behind shared state — every call
// is re-entrant and safe to call from multiple threads.
#ifndef PPOCR_PREPROCESS_H_
#define PPOCR_PREPROCESS_H_

#include <vector>
#include "ppocr/config.h"
#include "ppocr/image.h"

namespace ppocr {

// Output of `prep_det`. `chw` is the network input (NCHW with N=1):
//   N = 1
//   C = 3
//   H = in_h
//   W = in_w
// ratio_w = src_w / in_w, ratio_h = src_h / in_h — callers use these to
// unmap polygons from network space back to original-image space.
struct DetInput {
  int in_w = 0, in_h = 0;
  float ratio_w = 1, ratio_h = 1;
  std::vector<float> chw;
};

// Port of PaddleOCR's DetResizeForTest. Implemented modes:
//   LimitMin   — type 0: scale so min(src_w, src_h) == limit_side_len
//                (rounding to `stride` multiples); v6 / seal.
//   ResizeLong — type 2: scale so max(src_w, src_h) == resize_long
//                (rounding to `stride` multiples); v4 / v5.
// normalization: mean {0.485, 0.456, 0.406} std {0.229, 0.224, 0.225}
// scale 1/255, BGR channel order. Output is CHW float32.
DetInput prep_det(const Image& bgr, const DetResizeConfig& rc);

// One rec line:
//   1. Resize keep-ratio so h = img_h (= 48), w' = round(h * src_w / src_h).
//   2. If w' > batch_w, downscale so w' == batch_w (no further crop).
//      If w' < batch_w, left-align and zero-pad the rest of the row to batch_w.
//   3. Normalize (x/255 - 0.5) / 0.5.
//   4. Return CHW float32 of size 3*img_h*batch_w.
// valid_w is the unpadded width w' (1..batch_w).
// Reference: ppocr/data/imaug/rec_img_aug.py::resize_norm_img.
std::vector<float> prep_rec_line(const Image& line_bgr, int img_h, int batch_w, int& valid_w);

// cls: resize to cfg.h * cfg.w, ImageNet norm, CHW float32. No padding.
std::vector<float> prep_cls(const Image& bgr, const ClsConfig& cfg);

} // namespace ppocr
#endif // PPOCR_PREPROCESS_H_
