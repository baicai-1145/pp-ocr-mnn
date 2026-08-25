// pp-ocr-mnn — preprocessing implementation
//
// Three preprocessors matching PaddleOCR's data augmentation:
//   1. prep_det  — DetResizeForTest type 0 (LimitMin) / type 2 (ResizeLong)
//                 + ImageNet normalize + CHW float32 BGR.
//   2. prep_rec_line — keep-ratio resize to (h=48, w'=round(48*ratio)) +
//                 (x/255-0.5)/0.5 + zero-pad to batch_w. CHW float32 BGR.
//   3. prep_cls  — fixed (80,160) + ImageNet normalize + CHW float32 BGR.
//
// All resize operations are hand-written bilinear (PaddleOCR uses
// INTER_LINEAR in cv2.resize, which is the same algorithm). Avoiding
// MNN::ImageProcess here keeps the prep path branch-free; we only
// touch MNN in src/mnn_session.cpp.
#include "ppocr/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ppocr {

namespace {

// Bilinear resize of an 8-bit BGR HxWx3 image to dst_h x dst_w.
// Output buffer is uninitialized; caller owns the storage.
void resize_bilinear_bgr(const uint8_t* src, int src_w, int src_h,
                         uint8_t* dst, int dst_w, int dst_h) {
  if (dst_w <= 0 || dst_h <= 0 || src_w <= 0 || src_h <= 0) {
    return;
  }
  // Same scaling for x and y (cv2.resize default).
  const double sx = static_cast<double>(src_w) / dst_w;
  const double sy = static_cast<double>(src_h) / dst_h;
  for (int y = 0; y < dst_h; ++y) {
    // PaddleOCR uses half-pixel center, matching cv2.resize INTER_LINEAR.
    const double fy = (y + 0.5) * sy - 0.5;
    int y0 = static_cast<int>(std::floor(fy));
    if (y0 < 0) y0 = 0;
    if (y0 >= src_h - 1) y0 = src_h - 2;
    const double dy = fy - y0;
    for (int x = 0; x < dst_w; ++x) {
      const double fx = (x + 0.5) * sx - 0.5;
      int x0 = static_cast<int>(std::floor(fx));
      if (x0 < 0) x0 = 0;
      if (x0 >= src_w - 1) x0 = src_w - 2;
      const double dx = fx - x0;
      const uint8_t* p00 = src + (static_cast<size_t>(y0)     * src_w + x0)     * 3;
      const uint8_t* p01 = src + (static_cast<size_t>(y0)     * src_w + x0 + 1) * 3;
      const uint8_t* p10 = src + (static_cast<size_t>(y0 + 1) * src_w + x0)     * 3;
      const uint8_t* p11 = src + (static_cast<size_t>(y0 + 1) * src_w + x0 + 1) * 3;
      uint8_t* po = dst + (static_cast<size_t>(y) * dst_w + x) * 3;
      for (int c = 0; c < 3; ++c) {
        const double v =
            (1 - dy) * ((1 - dx) * p00[c] + dx * p01[c]) +
            dy       * ((1 - dx) * p10[c] + dx * p11[c]);
        int iv = static_cast<int>(std::lround(v));
        if (iv < 0) iv = 0;
        if (iv > 255) iv = 255;
        po[c] = static_cast<uint8_t>(iv);
      }
    }
  }
}

// Pad an integer up to the next multiple of `m` (PaddleOCR uses Python's
// built-in round; we use std::round for parity, then clamp to >= m).
int round_up_to_stride(int v, int m) {
  if (m <= 1) return v;
  int q = (v + m - 1) / m;
  int r = q * m;
  // PaddleOCR keeps a minimum of `m` so a fully degenerate image still
  // produces a non-empty network input.
  return r < m ? m : r;
}

// Convert HxWx3 BGR uint8 to CHW float32 with the given per-channel
// scale, mean, std. scale is applied first: y = (x*scale - mean) / std.
void hwc_bgr_to_chw_float(const uint8_t* src, int w, int h,
                          float* dst,
                          float scale,
                          const float* mean_bgr,
                          const float* std_bgr) {
  const size_t plane = static_cast<size_t>(w) * h;
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = src + static_cast<size_t>(y) * w * 3;
    for (int x = 0; x < w; ++x) {
      const uint8_t* px = row + x * 3;
      // BGR order, matching PaddleOCR's NormalizeImage (channel_num=3,
      // order='' means default [0,1,2] which equals BGR for BGR input).
      const float b = (static_cast<float>(px[0]) * scale - mean_bgr[0]) / std_bgr[0];
      const float g = (static_cast<float>(px[1]) * scale - mean_bgr[1]) / std_bgr[1];
      const float r = (static_cast<float>(px[2]) * scale - mean_bgr[2]) / std_bgr[2];
      dst[0 * plane + y * w + x] = b;
      dst[1 * plane + y * w + x] = g;
      dst[2 * plane + y * w + x] = r;
    }
  }
}

} // namespace

DetInput prep_det(const Image& bgr, const DetResizeConfig& rc) {
  if (bgr.c != 3 || bgr.w <= 0 || bgr.h <= 0) {
    throw std::runtime_error("prep_det: invalid input image");
  }
  int limit_side = rc.limit_side_len;
  int resize_long = rc.resize_long;
  int max_side = rc.max_side_limit > 0 ? rc.max_side_limit : 4000;
  int stride = rc.stride > 0 ? rc.stride : 32;

  int w = bgr.w;
  int h = bgr.h;

  // PaddleOCR DetResizeForTest::limit_side_len type0 ("min") vs type2 (long)
  // is encoded as a string in the original config; here we branch on
  // DetResizeConfig::Mode. The result is the target H/W before stride rounding.
  double ratio = 1.0;
  if (rc.mode == DetResizeConfig::Mode::LimitMin) {
    // type 0: keep aspect, scale so min(w, h) == limit_side_len.
    if (std::min(w, h) <= 0) ratio = 1.0;
    else ratio = static_cast<double>(limit_side) / std::min(w, h);
  } else {
    // type 2: keep aspect, scale so max(w, h) == resize_long.
    if (std::max(w, h) <= 0) ratio = 1.0;
    else ratio = static_cast<double>(resize_long) / std::max(w, h);
  }
  int resize_w = static_cast<int>(std::round(w * ratio));
  int resize_h = static_cast<int>(std::round(h * ratio));

  // max_side_limit cap (PaddleOCR clips the long side to max_side_limit).
  if (std::max(resize_w, resize_h) > max_side) {
    if (resize_w > resize_h) {
      resize_w = max_side;
    } else {
      resize_h = max_side;
    }
  }
  // Snap to stride (PaddleOCR uses `resize_h = max(round(...), stride)`).
  resize_h = round_up_to_stride(resize_h, stride);
  resize_w = round_up_to_stride(resize_w, stride);
  if (resize_h < stride) resize_h = stride;
  if (resize_w < stride) resize_w = stride;

  std::vector<uint8_t> resized(static_cast<size_t>(resize_w) * resize_h * 3);
  resize_bilinear_bgr(bgr.data.data(), bgr.w, bgr.h,
                      resized.data(), resize_w, resize_h);

  DetInput out;
  out.in_w = resize_w;
  out.in_h = resize_h;
  out.ratio_w = static_cast<float>(static_cast<double>(resize_w) / bgr.w);
  out.ratio_h = static_cast<float>(static_cast<double>(resize_h) / bgr.h);
  out.chw.assign(3 * static_cast<size_t>(resize_w) * resize_h, 0.f);

  // ImageNet mean/std in BGR order. PaddleOCR's NormalizeImage order='' for
  // BGR input keeps the default [0,1,2] which equals BGR. scale 1/255.
  const float mean_bgr[3] = {0.406f * 255.f / 255.f,  // will be normalized
                             0.456f * 255.f / 255.f,
                             0.485f * 255.f / 255.f};
  const float std_bgr[3]  = {0.225f, 0.224f, 0.229f};
  // (x/255 - mean) / std
  const float real_mean[3] = {0.406f, 0.456f, 0.485f};
  (void)mean_bgr;
  hwc_bgr_to_chw_float(resized.data(), resize_w, resize_h,
                       out.chw.data(), 1.f / 255.f,
                       real_mean, std_bgr);
  return out;
}

std::vector<float> prep_rec_line(const Image& line_bgr, int img_h, int batch_w,
                                 int& valid_w) {
  if (line_bgr.c != 3 || line_bgr.w <= 0 || line_bgr.h <= 0) {
    throw std::runtime_error("prep_rec_line: invalid input image");
  }
  if (img_h <= 0 || batch_w <= 0) {
    throw std::runtime_error("prep_rec_line: invalid target size");
  }
  // Keep-ratio resize so h = img_h. PaddleOCR uses
  //   ratio = img_h / src_h
  //   w'   = round(src_w * ratio)
  // and then:
  //   if w' > batch_w: rescale w' down to batch_w (preserving aspect)
  //   else:            zero-pad the rest of the row to batch_w.
  const double ratio = static_cast<double>(img_h) / line_bgr.h;
  int w = static_cast<int>(std::round(line_bgr.w * ratio));
  if (w <= 0) w = 1;
  if (w > batch_w) w = batch_w;
  valid_w = w;

  std::vector<uint8_t> resized(static_cast<size_t>(batch_w) * img_h * 3, 0);
  if (w > 0 && img_h > 0) {
    // The first `w` columns of each row get the resized pixels; the rest
    // stay zero (matches the zero-padded CHW tensor in resize_norm_img).
    std::vector<uint8_t> tmp(static_cast<size_t>(w) * img_h * 3);
    resize_bilinear_bgr(line_bgr.data.data(), line_bgr.w, line_bgr.h,
                        tmp.data(), w, img_h);
    for (int y = 0; y < img_h; ++y) {
      std::memcpy(resized.data() + static_cast<size_t>(y) * batch_w * 3,
                  tmp.data() + static_cast<size_t>(y) * w * 3,
                  static_cast<size_t>(w) * 3);
    }
  }

  // Normalize (x/255 - 0.5) / 0.5 = (x - 127.5) / 127.5. Padding stays 0,
  // so padded pixels normalize to (-0.5)/0.5 = -1. PaddleOCR's
  // NormalizeImage with scale=1/255, mean=[0.5]*3, std=[0.5]*3 matches.
  const float scale = 1.f / 255.f;
  const float mean[3] = {0.5f, 0.5f, 0.5f};
  const float std[3]  = {0.5f, 0.5f, 0.5f};
  std::vector<float> chw(3 * static_cast<size_t>(img_h) * batch_w, 0.f);
  hwc_bgr_to_chw_float(resized.data(), batch_w, img_h, chw.data(),
                       scale, mean, std);
  return chw;
}

std::vector<float> prep_cls(const Image& bgr, const ClsConfig& cfg) {
  if (bgr.c != 3 || bgr.w <= 0 || bgr.h <= 0) {
    throw std::runtime_error("prep_cls: invalid input image");
  }
  if (cfg.w <= 0 || cfg.h <= 0) {
    throw std::runtime_error("prep_cls: invalid cfg size");
  }
  std::vector<uint8_t> resized(static_cast<size_t>(cfg.w) * cfg.h * 3);
  resize_bilinear_bgr(bgr.data.data(), bgr.w, bgr.h, resized.data(),
                      cfg.w, cfg.h);

  std::vector<float> chw(3 * static_cast<size_t>(cfg.w) * cfg.h, 0.f);
  const float scale = 1.f / 255.f;
  // cfg.mean / cfg.std are length-3; PaddleOCR stores them in RGB order
  // in the inference.yml. We feed BGR pixels, so we re-map to BGR.
  float mean_bgr[3] = {
      cfg.mean.size() > 0 ? cfg.mean[2] : 0.485f,
      cfg.mean.size() > 1 ? cfg.mean[1] : 0.456f,
      cfg.mean.size() > 2 ? cfg.mean[0] : 0.406f,
  };
  float std_bgr[3] = {
      cfg.std.size() > 0 ? cfg.std[2] : 0.229f,
      cfg.std.size() > 1 ? cfg.std[1] : 0.224f,
      cfg.std.size() > 2 ? cfg.std[0] : 0.225f,
  };
  hwc_bgr_to_chw_float(resized.data(), cfg.w, cfg.h, chw.data(),
                       scale, mean_bgr, std_bgr);
  return chw;
}

} // namespace ppocr
