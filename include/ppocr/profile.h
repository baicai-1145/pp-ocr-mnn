// pp-ocr-mnn — per-stage profiling (M3-PERF1)
//
// A flat POD record of the wall-clock cost of each pipeline stage for
// ONE ppocr_run_file / ppocr_run call. Populated by the engine only
// when ppocr_config.profile is non-zero (zero cost when off: the
// instrumentation is a handful of steady_clock::now() calls per stage,
// ~100 ns total, but we keep the default off anyway so the frozen ABI
// stays deterministic).
//
// Stage breakdown (single image):
//
//   decode_ms      stb_image JPEG/PNG decode -> BGR8 buffer
//   det_prep_ms    prep_det: resize (limit_min/stride 32) + HWC->CHW + normalize
//   det_run_ms     det MNN session run (network forward only)
//   db_post_ms     DB postprocess: prob->bitmap->contours->minAreaRect->unclip->order sort
//   crop_warp_ms   per-box GetRotateCropImage perspective warps (+90 rot) (rec path)
//   rec_prep_ms    prep_rec_line per crop: keep-ratio resize + normalize + CHW pad
//   rec_run_ms     rec MNN session run(s), all batches
//   ctc_decode_ms  ctc_decode over all batch outputs (+UTF-8 reverse in seal mode)
//   cls_ms         textline orientation classifier (0 when cls off)
//
//   e2e_ms         wall clock of the whole run (>= sum of stages; the
//                  difference is glue: JSON build, line copy, etc.)
//
// create_ms is the engine cold-start cost (ppocr_create: model mmap/
// read + session create, NOT the first run). first_run_ms mirrors
// e2e_ms for the FIRST run on the engine (includes MNN kernel JIT /
// CUDA context init / first-touch allocations); subsequent runs are
// warm. The bench harness reports first_run separately from the warm
// mean/std.
#ifndef PPOCR_PROFILE_H_
#define PPOCR_PROFILE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ppocr_profile {
  float decode_ms;
  float det_prep_ms;
  float det_run_ms;
  float db_post_ms;
  float crop_warp_ms;
  float rec_prep_ms;
  float rec_run_ms;
  float ctc_decode_ms;
  float cls_ms;
  float e2e_ms;
  float create_ms;
  float first_run_ms;
  int   n_boxes;              // det boxes found this run (rec batch load)
  int   rec_batches;          // number of rec session runs this pass
  int   threads;              // engine num_threads actually applied (0=auto)
  char  backend[16];          // resolved backend of the det session
} ppocr_profile;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // PPOCR_PROFILE_H_
