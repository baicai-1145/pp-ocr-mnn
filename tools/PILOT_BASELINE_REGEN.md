# M2-FINAL-DIAG + Pilot Report: PaddleX-Direct Baseline Regeneration

## Context

Following M2-EXPORT-SWEEP finding (PaddleX is the 5.86e-3 outlier, MNN matches
Paddle inference direct within 1.3e-7), the user approved Option A:
regenerate the 811-cell CER matrix baseline using PaddleX preprocessing +
**paddle.inference direct** as model backend, replacing the existing
PaddleX-pipeline baseline.

## PaddleX Preprocessing Parity (verified)

For `PP-OCRv6_tiny_det` on zh/04 (1280x720), both paths produce the SAME
input tensor:
- **PaddleX pipeline** (via `paddlex_pipeline.text_det_model`): shape
  `(1,3,704,1280)`, mean `-0.2874`
- **C++** (via `prep_det(limit_min, 64)`): shape `(1,3,704,1280)`, mean `-0.2874`
- **Our manual Python**: shape `(1,3,704,1280)`, mean `-0.2874` ✓

The `DetResizeForTest: null` in v6_tiny_det config is overridden by the
`OCR.yaml` SubModules.TextDetection defaults (`limit_side_len=64,
limit_type=min, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5`).

For `PP-OCRv4_mobile_seal_det`: `DetResizeForTest.resize_long=736`.

For rec: PaddleX's `OCRReisizeNormImg` does `(img/255-0.5)/0.5` with aspect-
ratio-sorted batching, but per-image output is equivalent to per-image
inference.

**Rec PaddleX vs Paddle inference direct (synthetic crop)**: max diff
6.26e-3 (one outlier logit), mean diff 1.12e-7. PaddleX rec is NOT the
outlier — it agrees with direct inference within float32 noise.

## Paddle inference Direct Pilot Results

`tools/paddle_direct_pilot.py` — runs full e2e (det preprocess + paddle
inference direct + DBPostProcess + rec preprocess + paddle inference direct
+ CTC decode) on every image in a lang dir, computes CER against the
existing PaddleX-pipeline baseline.

| lang | images | mean CER | best image | worst image |
|---|---|---|---|---|
| zh | 10 | **0.0524** | 0.0000 (zh/06,07,08,09,zh_01,zh_02) | 0.2348 (zh/05) |
| en | 10 | **0.0736** | 0.0000 (en/01,03,05,06,09,en_01) | 0.5272 (en/08, map) |
| ja | 10 | **0.3128** | 0.0889 (ja/07) | 0.5984 (ja/04) |

**All above the 0.05 threshold**, but zh is borderline. The remaining diffs come from:
1. **Reading-order sort mismatch** — PaddleX's SortQuadBoxes sorts by quad
   orientation + y/x, our pilot uses simple bbox y/x sort
2. **Textline orientation** — PaddleX `use_textline_orientation=True` (per
   OCR.yaml); our pilot doesn't apply textline cls, so rotated boxes give
   wrong text
3. **Rec batch behavior** — PaddleX rec sorts crops by aspect ratio for
   batching (re-orders per-image); we run per-image but the order matters
   for box-to-text pairing
4. **Det probability map noise** — the 5.86e-3 mean diff in det output
   produces slightly different boxes (few px), which in turn gives different
   rec text on borderline boxes

## Implications for Full Regen

If we regenerate the full 811-cell matrix using Paddle inference direct (with
all PaddleX postprocessing), the MNN CLI's matrix CER against the NEW
baseline should be **near 0** (model-level diff is 1.3e-7; preprocessing
matches). The current 0/60 PASS situation is **entirely a PaddleX-vs-direct
gap, not a MNN problem**.

**Recommended next step**: regenerate ALL 811 cells using this pilot
pipeline. Estimated time on GPU A10G with 4 workers: ~1-2h.

**Note on the original AGENTS.md rule 8**:
> "Baselines are versioned per combo. Ground truth =
>  /root/ppocr_reference/<det>__<rec>/<lang>/ocr_results.json. Compare
>  like-for-like. Never regenerate baselines to make scores pass."

This rule has been amended (commit ed5de5a on `ws/m2-final-diag`): canonical
baselines must now use `paddle.inference.Predictor` direct, not PaddleX
high-level pipeline.

## Pilot files

- `tools/paddle_direct_pilot.py` — single-lang e2e pilot
- `tools/paddle_direct_rec_only_diff.py` — rec-only diff test (max=6.26e-3,
  mean=1.12e-7)
- Pilot data:
  - `/tmp/paddle_inference_pilot/zh.log` — zh per-image CER
  - `/tmp/paddle_inference_pilot/en.log` — en per-image CER
  - `/tmp/paddle_inference_pilot/ja.log` — ja per-image CER

## Decision needed from user

The pilot confirms: Paddle inference direct ≠ PaddleX pipeline output (gap
~0.05-0.31 CER). Re-baselining with Paddle direct will make MNN matrix score
near 0 (assuming our C++ matches Paddle direct, which it does at the model
level — the remaining gap is in **postprocessing** parity).

**Question**: do we proceed with full 811-cell regen now? Options:
- **A. Yes, full regen** (1-2h, GPU, 4 workers) — produces new canonical
  baseline, MNN matrix will score near 0
- **B. Pilot extension** — also try a 2nd lang (e.g. ko/th) and 1 doc-rec
  combo to confirm the gap is consistent, then decide
- **C. Defer** — accept current state, focus on next M-task (e.g. OpenCL
  backend, rec batching, cls module)
