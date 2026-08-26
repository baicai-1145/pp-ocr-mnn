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

// Snap an integer to the nearest multiple of `m` using banker's
// rounding (half-to-even), exactly like Python's built-in `round()`.
// PaddleOCR's `DetResizeForTest` is implemented in NumPy and uses
// `int(round(... / m) * m)`; numpy.round is half-to-even on IEEE 754
// floats. We use `std::nearbyint` which defaults to FE_TONEAREST (the
// same half-to-even rule) and so matches NumPy bit-for-bit on the
// PaddleOCR det resize path.
//
// Why this matters: at the PaddleX baseline geometry (limit_min 64,
// stride 32), a 720-pixel-tall image becomes `round(720/32)*32 = 704`
// with Python's `round`. If we use ceiling instead (the previous
// `(v + m - 1)/m * m`) the resize would land on 736, which is 32 px
// taller in the prob map, and the DB contours would land on a
// different pixel row than the baseline — box position error of
// 10-30 px and CER > 0.5.
//
// Test cases (see tests/test_preprocess.cpp::test_stride_align):
// Convert HxWx3 BGR uint8 to CHW float32 with the given per-channel
// scale, mean, std. scale is applied first: y = (x*scale - mean) / std.
// The mean/std arrays are aligned with the BGR channel order PaddleOCR
// uses (mean[0] for B, mean[2] for R); no axis swap is performed here.
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
      // px[0] = B, px[1] = G, px[2] = R. mean_bgr/std_bgr are in the
      // same BGR order, so channel i gets mean_bgr[i].
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

// Snap an integer to the nearest multiple of `m` using banker's
// rounding (half-to-even), exactly like Python's built-in `round()`.
// PaddleOCR's `DetResizeForTest` is implemented in NumPy and uses
// `int(round(... / m) * m)`; numpy.round is half-to-even on IEEE 754
// floats. We use `std::nearbyint` which defaults to FE_TONEAREST (the
// same half-to-even rule) and so matches NumPy bit-for-bit on the
// PaddleOCR det resize path.
//
// Why this matters: at the PaddleX baseline geometry (limit_min 64,
// stride 32), a 720-pixel-tall image becomes `round(720/32)*32 = 704`
// with Python's `round`. If we use ceiling instead (the previous
// `(v + m - 1)/m * m`) the resize would land on 736, which is 32 px
// taller in the prob map, and the DB contours would land on a
// different pixel row than the baseline — box position error of
// 10-30 px and CER > 0.5.
//
// Test cases (see tests/test_preprocess.cpp::test_stride_align):
//   720/32 = 22.5  -> 22*32 = 704  (half-to-even: 22 even -> 22)
//   736/32 = 23    -> 23*32 = 736
//   944/32 = 29.5  -> 30*32 = 960  (half-to-even: 30 even -> 30)
//   976/32 = 30.5  -> 30*32 = 960  (half-to-even: 30 even -> 30)
//   992/32 = 31    -> 31*32 = 992
int round_up_to_stride(int v, int m) {
  if (m <= 1) return v;
  double q = static_cast<double>(v) / static_cast<double>(m);
  long q_int = static_cast<long>(std::nearbyint(q));  // FE_TONEAREST, half-to-even
  long r = q_int * m;
  // PaddleOCR keeps a minimum of `m` so a fully degenerate image still
  // produces a non-empty network input.
  return r < m ? m : static_cast<int>(r);
}

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

  // PaddleOCR DetResizeForTest modes:
  //   type 0 ("limit_min")   — keep aspect, scale so min(w, h) == limit_side_len.
  //   type 2 ("resize_long") — keep aspect, scale so max(w, h) == resize_long.
  //   null  ("no_resize")    — image is fed at native resolution; only the
  //                            model's /4 stride shrinks the prob map. PaddleOCR
  //                            uses this for v6 det (`DetResizeForTest: null`)
  //                            and our M2 ground-truth baseline was generated
  //                            with that path. The current v6_tiny_det .mnn
  //                            has a partially-dynamic input that fails
  //                            MNN's shape inference for native-resolution
  //                            inputs (Broadcast error, dim1=46, dim2=45);
  //                            we fall back to "limit_min" mode here so the
  //                            model still runs, accepting a small accuracy
  //                            loss vs the reference. The M2-CER gate is
  //                            still passed when the resize stays close to
  //                            the reference.
  int resize_w = w;
  int resize_h = h;
  if (rc.mode == DetResizeConfig::Mode::LimitMin) {
    // PaddleOCR type0 limit_type="min": only UPSCALE when min(h,w) < limit.
    // Images already >= limit keep ratio 1.0 (this is the PaddleX baseline
    // path: limit=64 never downscales real photos; zh/04 1280x720 stays
    // 1280x720 then stride-snaps to 1280x704).
    double ratio = (std::min(w, h) > 0 && std::min(w, h) < limit_side)
        ? static_cast<double>(limit_side) / std::min(w, h) : 1.0;
    resize_w = static_cast<int>(std::round(w * ratio));
    resize_h = static_cast<int>(std::round(h * ratio));
  } else if (rc.mode == DetResizeConfig::Mode::ResizeLong) {
    double ratio = (std::max(w, h) > 0)
        ? static_cast<double>(resize_long) / std::max(w, h) : 1.0;
    resize_w = static_cast<int>(std::round(w * ratio));
    resize_h = static_cast<int>(std::round(h * ratio));
  } else {
    // NoResize: feed native resolution (model is /32-dynamic per
    // tools/DET_DYNAMIC.md). Legacy fallback scaled like limit_min — kept
    // only for pathological inputs; normal path never lands here.
    double ratio = (std::min(w, h) > 0 && std::min(w, h) < limit_side)
        ? static_cast<double>(limit_side) / std::min(w, h) : 1.0;
    resize_w = static_cast<int>(std::round(w * ratio));
    resize_h = static_cast<int>(std::round(h * ratio));
  }

  // max_side_limit cap (PaddleOCR clips the long side to max_side_limit).
  if (std::max(resize_w, resize_h) > max_side) {
    if (resize_w > resize_h) {
      resize_w = max_side;
    } else {
      resize_h = max_side;
    }
  }
  // Snap to stride (PaddleOCR uses `resize_h = max(round(...), stride)`).
  // Skipped for NoResize: PaddleOCR doesn't align when no resize is
  // applied — the network input is the raw image.
//   PaddleOCR's type-0 path uses `int(round(resize / 32) * 32)`
//   (banker's rounding, half-to-even). Type-2 (resize_long) uses
//   `(resize + 127) // 128 * 128` (ceiling). We model the difference
//   here so the resize shape matches PaddleOCR's numpy output bit for
//   bit — at limit_min=64, stride=32 (the PaddleX baseline path) a
//   720-tall image lands on 704 (= round(720/32)*32), not 736.
  if (rc.mode == DetResizeConfig::Mode::ResizeLong) {
    resize_h = ((resize_h + stride - 1) / stride) * stride;
    resize_w = ((resize_w + stride - 1) / stride) * stride;
    if (resize_h < stride) resize_h = stride;
    if (resize_w < stride) resize_w = stride;
  } else if (rc.mode != DetResizeConfig::Mode::NoResize) {
    resize_h = round_up_to_stride(resize_h, stride);
    resize_w = round_up_to_stride(resize_w, stride);
    if (resize_h < stride) resize_h = stride;
    if (resize_w < stride) resize_w = stride;
  }

  std::vector<uint8_t> resized;
  if (resize_w == bgr.w && resize_h == bgr.h) {
    // NoResize: avoid the bilinear copy and reuse the source buffer.
    resized.assign(bgr.data.begin(), bgr.data.end());
  } else {
    resized.assign(static_cast<size_t>(resize_w) * resize_h * 3, 0);
    resize_bilinear_bgr(bgr.data.data(), bgr.w, bgr.h,
                        resized.data(), resize_w, resize_h);
  }

  DetInput out;
  out.in_w = resize_w;
  out.in_h = resize_h;
  out.ratio_w = static_cast<float>(static_cast<double>(resize_w) / bgr.w);
  out.ratio_h = static_cast<float>(static_cast<double>(resize_h) / bgr.h);
  out.chw.assign(3 * static_cast<size_t>(resize_w) * resize_h, 0.f);

  // PaddleOCR DetPreProcess: DecodeImage(BGR) -> DetResizeForTest ->
  // NormalizeImage(order='chw', scale=1/255, mean=[0.485,0.456,0.406],
  // std=[0.229,0.224,0.225]) -> ToCHWImage. The mean/std arrays are
  // broadcast along the channel axis in the order PaddleOCR stores
  // them, which is BGR=0,1,2. Our input is BGR, our output is CHW with
  // channel 0 = B, 1 = G, 2 = R. No reordering is required: just
  // apply cfg.det.thresh-mean / std per channel position.
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float std [3] = {0.229f, 0.224f, 0.225f};
  hwc_bgr_to_chw_float(resized.data(), resize_w, resize_h,
                       out.chw.data(), 1.f / 255.f, mean, std);
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
  // Keep-ratio resize so h = img_h. paddlex `resize_norm_img`:
  //   ratio = w / float(h)                 (h = warped crop height)
  //   resized_w = int(math.ceil(imgH * ratio))   (Python `int` is
  //         truncate-toward-zero, but `math.ceil` is applied first so
  //         the value is the small integer >= the real resize width)
  // We replicate that exactly: `static_cast<int>(std::ceil(...))`
  // which is the C++ equivalent of `int(math.ceil(...))`.
  const double ratio = static_cast<double>(img_h) / line_bgr.h;
  int w = static_cast<int>(std::ceil(line_bgr.w * ratio));
  if (w <= 0) w = 1;
  if (w > batch_w) w = batch_w;
  valid_w = w;

  std::vector<uint8_t> resized(static_cast<size_t>(batch_w) * img_h * 3, 0);
  std::vector<uint8_t> tmp;
  if (w > 0 && img_h > 0) {
    // The first `w` columns of each row get the resized pixels; the rest
    // stay zero (matches the zero-padded CHW tensor in resize_norm_img).
    tmp.assign(static_cast<size_t>(w) * img_h * 3, 0);
    resize_bilinear_bgr(line_bgr.data.data(), line_bgr.w, line_bgr.h,
                        tmp.data(), w, img_h);
    for (int y = 0; y < img_h; ++y) {
      std::memcpy(resized.data() + static_cast<size_t>(y) * batch_w * 3,
                  tmp.data() + static_cast<size_t>(y) * w * 3,
                  static_cast<size_t>(w) * 3);
    }
  }

  // Normalize the valid `w` columns only, then place into the CHW
  // tensor. The remaining `batch_w - w` columns are left at the CHW
  // init value of 0.0f — paddlex 3.x `resize_norm_img` does the same
  // (`padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)`
  // after the normalized image has been placed at
  // `padding_im[:, :, 0:resized_w]`). Normalized zero corresponds
  // to the original uint8 value 127.5 (mid-gray), but the rec
  // network has been trained against paddlex's convention so we
  // reproduce it exactly. (The previous M1-PIPE behavior normalized
  // uint8 zeros to -1.0, which is a different semantic and shifts
  // the model's predictions.)
  const float scale = 1.f / 255.f;
  const float mean[3] = {0.5f, 0.5f, 0.5f};
  const float std[3]  = {0.5f, 0.5f, 0.5f};
  std::vector<float> chw(3 * static_cast<size_t>(img_h) * batch_w, 0.f);
  if (w > 0) {
    // tmp is a (w, img_h, 3) uint8 buffer of the kept-aspect resize.
    // Normalize it in-place into a (3, img_h, w) float buffer, then
    // copy into the first `w` columns of the per-row CHW layout.
    std::vector<float> tmp_f(static_cast<size_t>(w) * img_h * 3);
    hwc_bgr_to_chw_float(tmp.data(), w, img_h, tmp_f.data(),
                         scale, mean, std);
    // tmp_f is CHW (3, img_h, w). Scatter into chw (3, img_h, batch_w).
    const size_t plane_in  = static_cast<size_t>(w)   * img_h;
    const size_t plane_out = static_cast<size_t>(batch_w) * img_h;
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < img_h; ++y) {
        std::memcpy(chw.data() + c * plane_out + y * batch_w,
                    tmp_f.data() + c * plane_in + y * w,
                    static_cast<size_t>(w) * sizeof(float));
      }
    }
  }
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
  // PaddleClas cls pipeline: DecodeImage(BGR) -> Resize -> NormalizeImage
  // (order='chw', mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) ->
  // ToCHWImage. The mean/std arrays map onto BGR channel positions in
  // the order they are stored; we feed BGR and emit CHW(BGR), so we
  // apply cfg.mean[i] / cfg.std[i] directly to channel i. No
  // reordering (this was the M1 bug — the inference.yml values are
  // already BGR-aligned, not RGB).
  const float default_mean[3] = {0.485f, 0.456f, 0.406f};
  const float default_std [3] = {0.229f, 0.224f, 0.225f};
  float mean[3] = {
      cfg.mean.size() > 0 ? cfg.mean[0] : default_mean[0],
      cfg.mean.size() > 1 ? cfg.mean[1] : default_mean[1],
      cfg.mean.size() > 2 ? cfg.mean[2] : default_mean[2],
  };
  float stdv[3] = {
      cfg.std.size() > 0 ? cfg.std[0] : default_std[0],
      cfg.std.size() > 1 ? cfg.std[1] : default_std[1],
      cfg.std.size() > 2 ? cfg.std[2] : default_std[2],
  };
  hwc_bgr_to_chw_float(resized.data(), cfg.w, cfg.h, chw.data(),
                       scale, mean, stdv);
  return chw;
}

} // namespace ppocr
