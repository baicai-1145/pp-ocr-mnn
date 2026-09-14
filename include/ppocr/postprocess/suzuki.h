// pp-ocr-mnn — Suzuki-Abe single-pass border following (cv::findContours core).
// Owner: post. No OpenCV; pure C++17. No platform ifdefs.
//
// Drop-in replacement for the old "CCL + global hole flood + Moore trace"
// trio inside db_post.cpp. It reproduces OpenCV 4.x's *new* contour scanner
// (modules/imgproc/src/contours_new.cpp, struct ContourScanner_) for
//     mode   = RETR_LIST
//     method = CHAIN_APPROX_NONE
// which is what `cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)`
// hands to PaddleX's `boxes_from_bitmap` (ppocr/postprocess/db_postprocess.py
// passes the raw mask; the `cv2.findContours` call in that file takes the
// defaults, method=CHAIN_APPROX_SIMPLE, but PaddleX's pipeline path — the one
// the baselines were generated with — feeds the raw contour into
// get_mini_boxes, so every point participates in minAreaRect).
//
// Semantics ported 1:1 from contours_new.cpp:
//
//   * `image` is the 0/1 mask with a 1-pixel zero border, exactly as
//     cv::findContours builds via copyMakeBorder. The caller's `mask` buffers
//     already satisfy this because the prob map is padded by prep; we
//     therefore index the raw mask and clamp the scan window to
//     x = 1..W-2, y = 1..H-2, i.e. treat the outermost ring as the border.
//     (A foreground pixel touching the image edge can therefore not start a
//     contour — same as an all-zero cv2 border ring.)
//
//   * findNextX: SIMD-accelerated run scan; we use a plain byte scan over
//     `image[y][x] != prev` (no branch on the value).
//
//   * contourScan classify:
//         prev == 0 && p == 1                 -> outer border
//         p != 0 && prev != 0                 -> resume scan
//         prev & 0xFE                         -> hole border, lnbd.x = x-1
//     `prev` is image[y][x-1], `p` is image[y][x].
//
//   * is_hole borders start at (x-1, y); outer borders start at (x, y).
//
//   * icvFetchContourEx: 8-direction border follower with
//         s_end = is_hole ? 0 : 4
//     starting search s = s_end-1, s_end-2, ...  Probe order after arriving at
//     a pixel is s = (previous_s + 4) & 7, then ++s while < 8.
//     `image` is marked in place: nbd=0x02 (NEW) for a fresh border, or'd with
//     0x80 (RIGHT) when the border is not directly above the next one.
//     Because the marks share the array with the 0/1 foreground, the follower
//     tests `(*i4 & 0x7F) != 0` (Trait<schar>::checkValue) instead of `!= 0`,
//     and `isVal(i3)` becomes `*i3 == 0x02`.
//
//   * RETR_LIST is `isSimple()`: no parent search, no hierarchy — every
//     border is emitted, in the order the raster scan finds it. That order
//     is what the old code approximated by sorting components by pixel count;
//     here it comes out for free.
//
// The returned contours are vectors of (x, y) with the 1px border NOT
// subtracted (the caller passes a mask whose usable area is 1..W-2/1..H-2,
// and db_postprocess maps coordinates in mask space directly).
#ifndef PPOCR_POSTPROCESS_SUZUKI_H_
#define PPOCR_POSTPROCESS_SUZUKI_H_

#include <cstdint>
#include <vector>

namespace ppocr {

struct IntPoint {
  int x = 0;
  int y = 0;
};

// Find all borders (outer + hole) of the binary mask `image` (row-major,
// `w` x `h`, values 0/1) in cv::findContours(RETR_LIST, CHAIN_APPROX_NONE)
// order. `image` is modified in place (marker bits 0x02/0x80 are written);
// pass a scratch copy if the caller needs the mask afterwards.
//
// A 1-pixel zero border is applied internally, exactly as cv2 does via
// `copyMakeBorder(image, image, 1, 1, 1, 1, BORDER_CONSTANT | BORDER_ISOLATED)`,
// and the returned coordinates are shifted back by (-1, -1) — i.e. they are
// in the caller's original pixel space. A foreground pixel touching the edge
// therefore CAN start a contour, and the contour may contain x == -1 / w or
// y == -1 / h, matching cv2.
void suzuki_borders(const uint8_t* image, int w, int h,
                    std::vector<std::vector<IntPoint>>* contours);

}  // namespace ppocr

#endif  // PPOCR_POSTPROCESS_SUZUKI_H_
