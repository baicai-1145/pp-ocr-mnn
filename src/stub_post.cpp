// pp-ocr-mnn — postprocess stubs (TEMP STUB - remove when ws/post merges)
//
// ws/m1 implements the public C ABI and det MNN session; the postprocess
// owner is the `ws/post` branch. To keep `ws/m1` independently
// compilable, this file provides no-op implementations that return
// empty / default-initialized values, so:
//   - db_postprocess  returns an empty vector (no det boxes) — but
//                     M1 ships a MINIMAL FALLBACK that binarizes the
//                     probability map and emits axis-aligned bounding
//                     boxes around the largest connected component.
//                     This is enough to prove the det MNN session is
//                     real (lines in the CLI JSON are non-empty with
//                     coordinates back in original image space), but it
//                     is NOT a faithful port of PaddleOCR's DB. The
//                     real box_thresh / unclip / clipper pipeline lands
//                     on ws/post and replaces this entire file.
//   - ctc_decode      returns an empty string with score 0 (M1 only).
//   - geometry helpers return false / leave outputs untouched (M2+).
//
// The decision-maker removes this file on merge: real implementations
// land in src/ on ws/post, which adds these .cpp files (geometry.cpp,
// db_post.cpp, ctc_decode.cpp) and removes the symbol conflicts.
//
// Do NOT add full DB semantics here. If ws/m1 needs a partial postprocess
// (e.g. to make the v6-tiny CER gate pass), coordinate with the
// decision-maker to either port it into ws/post or merge ws/post
// first.
#include "ppocr/postprocess/ctc_decode.h"
#include "ppocr/postprocess/db_post.h"
#include "ppocr/postprocess/geometry.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <vector>

namespace ppocr {

bool min_area_rect(const PointF* pts, size_t n, PointF out[4]) {
  (void)pts; (void)n;
  if (out) for (int i = 0; i < 4; ++i) { out[i].x = 0; out[i].y = 0; }
  return false;
}

void sort_min_area_rect_points(PointF box[4]) {
  (void)box;
}

Image warp_perspective_quad(const Image& src, const PointF quad[4],
                            int dst_w, int dst_h) {
  (void)src; (void)quad;
  Image out;
  out.w = dst_w; out.h = dst_h; out.c = 3;
  if (dst_w > 0 && dst_h > 0) {
    out.data.assign(static_cast<size_t>(dst_w) * dst_h * 3, 0);
  }
  return out;
}

std::vector<DetBox> db_postprocess(const float* prob, int prob_h, int prob_w,
                                   int src_w, int src_h,
                                   float ratio_w, float ratio_h,
                                   const DetConfig& cfg) {
  if (!prob || prob_h <= 0 || prob_w <= 0) return {};
  // M1 fallback: binarize at `thresh`, find the largest connected
  // component, emit its axis-aligned bbox in original-image coords.
  // This is intentionally NOT a PaddleOCR port — it exists so the
  // CLI emits non-empty JSON that proves the MNN det session is real.
  const float thr = cfg.thresh;
  std::vector<uint8_t> mask(static_cast<size_t>(prob_w) * prob_h, 0);
  for (int i = 0; i < prob_w * prob_h; ++i) {
    mask[i] = (prob[i] >= thr) ? 1 : 0;
  }
  // BFS to find the largest connected component.
  std::vector<int> comp(static_cast<size_t>(prob_w) * prob_h, -1);
  std::vector<std::pair<int,int>> seeds;
  seeds.reserve(64);
  int largest = -1;
  size_t largest_size = 0;
  int label = 0;
  std::deque<int> q;
  for (int y = 0; y < prob_h; ++y) {
    for (int x = 0; x < prob_w; ++x) {
      int idx = y * prob_w + x;
      if (!mask[idx] || comp[idx] >= 0) continue;
      // BFS
      int size = 0;
      int xmin = x, xmax = x, ymin = y, ymax = y;
      double sum = 0.0;
      comp[idx] = label; q.push_back(idx);
      while (!q.empty()) {
        int cur = q.front(); q.pop_front();
        int cy = cur / prob_w, cx = cur - cy * prob_w;
        ++size;
        sum += prob[cur];
        if (cx < xmin) xmin = cx;
        if (cx > xmax) xmax = cx;
        if (cy < ymin) ymin = cy;
        if (cy > ymax) ymax = cy;
        const int nbrs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
        for (auto& n : nbrs) {
          int nx = cx + n[0], ny = cy + n[1];
          if (nx < 0 || nx >= prob_w || ny < 0 || ny >= prob_h) continue;
          int ni = ny * prob_w + nx;
          if (mask[ni] && comp[ni] < 0) {
            comp[ni] = label;
            q.push_back(ni);
          }
        }
      }
      if (static_cast<size_t>(size) > largest_size) {
        largest_size = static_cast<size_t>(size);
        largest = label;
      }
      ++label;
    }
  }
  if (largest < 0) return {};
  // Recompute bbox of `largest` comp (we discarded intermediate stats).
  int xmin = prob_w, xmax = -1, ymin = prob_h, ymax = -1;
  double sum = 0.0; int cnt = 0;
  for (int y = 0; y < prob_h; ++y) {
    for (int x = 0; x < prob_w; ++x) {
      int idx = y * prob_w + x;
      if (comp[idx] != largest) continue;
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
      sum += prob[idx]; ++cnt;
    }
  }
  if (xmax < 0) return {};
  // Map back to original-image pixel coords using inverse ratios.
  // Note: prob space is HxW, original is src_h x src_w, so we scale
  // x by 1/ratio_w and y by 1/ratio_h. ratio_w = in_w / src_w.
  const float inv_rw = ratio_w > 0 ? 1.f / ratio_w : 1.f;
  const float inv_rh = ratio_h > 0 ? 1.f / ratio_h : 1.f;
  float x0 = xmin * inv_rw;
  float y0 = ymin * inv_rh;
  float x1 = xmax * inv_rw;
  float y1 = ymax * inv_rh;
  // Clamp to image bounds.
  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;
  if (x1 > src_w) x1 = src_w;
  if (y1 > src_h) y1 = src_h;
  DetBox b;
  b.poly[0] = x0; b.poly[1] = y0;
  b.poly[2] = x1; b.poly[3] = y0;
  b.poly[4] = x1; b.poly[5] = y1;
  b.poly[6] = x0; b.poly[7] = y1;
  b.score = static_cast<float>(sum / std::max(1, cnt));
  return {b};
}

RecOut ctc_decode(const float* logits, int timesteps, int num_classes,
                  const RecConfig& cfg) {
  (void)logits; (void)timesteps; (void)num_classes; (void)cfg;
  return RecOut{};
}

} // namespace ppocr
