// SE-pool rewrite variant selection (task-7).
//
// tools/rewrite_se_pool.py replaces the SE-block global average poolings of
// a det graph (Pooling3D{isGlobal:true, type:AVEPOOL}) with an equivalent
// Reduction{MEAN, dim=[2,3], keepDims} node. It is a pure graph
// re-serialization - weights are untouched - and it is numerically inert
// (PP-OCRv5_mobile, Metal, 5 languages: full-res MLC delta 0.0e+00 vs the
// original; prob-map meanabs 7.3e-6, binarized agreement 100%).
//
// The benefit is Metal-specific: MNN's Metal backend dispatches the rewritten
// Reduction better than the global pool it replaces. The rewritten graph is
// 2-6% SLOWER on CPU, so the variant is only ever selected for Metal.
//
// This header holds the *policy* (which det models have a useful variant) as
// a dependency-free pure function so it can be unit tested without any .mnn
// files present. The I/O half of the decision (does the file exist, is the
// backend Metal, is the knob disabled) stays in src/ppocr.cpp.
#ifndef PPOCR_SE_REWRITE_H
#define PPOCR_SE_REWRITE_H

#include <string>

namespace ppocr {

// Det models whose SE-block global poolings are `group=1x1x1` single
// threadgroup W*H serial scans under Metal (the actual Metal det bottleneck),
// i.e. the ones where the rewrite measured a win.
//
// Deliberately excluded, on measured evidence:
//   - PP-OCRv4_server_det: global pooling is 0.36% of Metal GPU time
//     (MNN_METAL_OP_PROFILE) and already grouped-dispatched
//     (group=1x1x{2,4,6,8}), so the rewrite has nothing to win and regresses
//     it 1141 -> 3541 ms.
//   - PP-OCRv6_medium_det / PP-OCRv5_server_det: no global Pooling3D ops, so
//     rewrite_se_pool.py is a no-op.
//   - the *_seal_det models: same reasoning as their backbone counterparts,
//     and they are not in the verified gate matrix.
inline bool det_has_se_rewrite_variant(const std::string& det_name) {
  return det_name == "PP-OCRv6_tiny_det" ||
         det_name == "PP-OCRv6_small_det" ||
         det_name == "PP-OCRv5_mobile_det" ||
         det_name == "PP-OCRv4_mobile_det";
}

// The variant's file name for a model, e.g. "PP-OCRv5_mobile_det.red.mnn".
// Callers must still check this is not an allowlisted model and that the file
// exists on disk before using it.
inline std::string se_rewrite_variant_name(const std::string& det_name) {
  return det_name + ".red.mnn";
}

}  // namespace ppocr

#endif  // PPOCR_SE_REWRITE_H
