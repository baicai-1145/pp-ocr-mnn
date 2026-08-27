# Seal Recognition Pipeline (M4)

## TL;DR

A seal (印章) is a circular stamp on a Chinese government / corporate /
notary document. The text forms a ring around the seal's outer edge,
and (optionally) a star or block of text in the center. The det
network sees the ring as a single closed polygon (often 50-200 points
along the curve) and emits 1-3 polygons per image (outer ring, inner
ring, central star).

The recognition step is a plain `PP-OCRvN_*_rec` model applied to
each polygon's bounding rectangle. There is no special rec model for
seals; the same zh / en / ja / ru rec models used for normal text
line recognition are used. The rec score threshold is 0.0 (not the
0.5 default for normal text) so all rec output is kept.

## PaddleOCR reference chain

Source files (read while building M4):
- `/root/.local/pytools/lib/python3.12/site-packages/paddlex/configs/pipelines/seal_recognition.yaml`
  — `SealOCR` sub-pipeline config (model + params)
- `/root/.local/pytools/lib/python3.12/site-packages/paddlex/inference/pipelines/ocr/pipeline.py` — `text_type="seal"` branch
- `/root/.local/pytools/lib/python3.12/site-packages/paddlex/inference/pipelines/components/common/crop_image_regions.py` — `CropByPolys(det_box_type="poly")` + `get_poly_rect_crop`
- `/root/PaddleOCR/deploy/cpp_infer/src/common/processors.h/cc` — C++ port
- `/root/PaddleOCR/configs/det/PP-OCRv4/PP-OCRv4_mobile_seal_det.yml` — training-time config (DB postprocess params)

End-to-end flow:

```
image ─► det (PP-OCRv4_mobile_seal_det, DB 0.2/0.6/0.5) ─►
  polygons (N=50..200 each, circular contour) ─►
  per poly:
    min_area_rect + sort_min_area_rect_points ─► 4-corner quad ─►
    GetRotateCropImage: 4-point perspective warp ─►
    (rotate 90° CCW if H/W >= 1.5) ─►
    rec (PP-OCRvN_*_rec, score_thresh=0) ─►
    text + score
join (order-agnostic: ring text has no reading order)
```

The Python `AutoRectifier` / `CurveTextRectifier` is the Plan-B path
used when the polygon's IoU with its min-area-rect is < 0.7. For the
seal test set we measured all 33 images have IoU > 0.7, so the simple
min-area-rect crop is the right operating point (and matches the C++
`CropByPolys::GetPolyRectCrop` fallback).

## PaddleXR parameters (verified against paddle source)

| param | value | source |
|---|---|---|
| `limit_side_len` | 736 | `seal_text_detection.yaml`, `TextDetection.limit_side_len` |
| `limit_type`    | "min" | `seal_text_detection.yaml`, `TextDetection.limit_type` |
| `max_side_limit`| 4000 | `seal_text_detection.yaml`, `TextDetection.max_side_limit` |
| `thresh`        | 0.2 | `seal_text_detection.yaml`, DBPostProcess |
| `box_thresh`    | 0.6 | `seal_text_detection.yaml`, DBPostProcess |
| `unclip_ratio`  | 0.5 | `seal_text_detection.yaml`, DBPostProcess |
| `box_type`      | "poly" | `seal_text_detection.yaml`, DBPostProcess |
| `rec_score_thresh` | 0 | `seal_recognition.yaml`, `TextRecognition.score_thresh` |
| `use_textline_orientation` | False | `seal_recognition.yaml`, `SealOCR.use_textline_orientation` |

The seal det's prob map stride is 4 (DB net /4 downsample) and the
input is square after the 736-min resize (matches v6 family).

## The C++ port in our tree

`third_party/MNN` is unchanged. The seal pipeline is **entirely in
our postprocess + C ABI** layer:

### `db_postprocess` (src/postprocess/db_post.cpp)

Same code path as text det. The seal prob map is fed through the same
`unclip → min_area_rect → sort_min_area_rect_points` chain. Output is
`DetBox{ poly[8], score }` where `poly` is the 4-corner min-area-rect
of the original N-point poly. This is **equivalent** to paddlex's
"high-IoU fast path" in `CropByPolys::GetPolyRectCrop`.

### `run_rec_sync` (src/ppocr.cpp:run_rec_sync)

Same code path. `warp_perspective_quad` is the 4-point perspective
warp (POST-7: full 8-param projective transform, not 6-param affine).
The 90° rotation if H/W >= 1.5 is the same as the normal text
pipeline.

**M4-SEAL2 update**: for seal mode the crop is NOT `warp_perspective_quad`
but `polar_unwrap_band` with an adaptive radial band derived from the
det poly's min-area-rect geometry:

```
rect_center = centroid(quad)
d_rect      = |rect_center - image_center|
h_rect      = short side of the rect
r_outer     = d_rect + h_rect / 2          # ring's outer edge
r_inner     = r_outer - max(30, 0.35*r_outer)
angular_n   = min(2*pi*r_mid, rec_w)       # rec_w = 320
radial_n    = rec_h = 48
```

Rationale: the det polys are rotated min-area rects whose VERTEX radii
w.r.t. the image center overshoot the ring (290 vs true 220 on zh_00_0);
`d_rect + h_rect/2` measured 212–216 on zh/en seals whose red-ink ring
sits at r≈205–220 (verified by ink histograms). A fixed
`[0.60·r_seal, 1.05·r_seal]` band (M4-SEAL) clipped the en rings at the
strip's bottom edge. The rec output is UTF-8-codepoint-reversed because
the unwrap samples theta CW in image coords while ring text reads CCW
from the right.

### `run_full` (src/ppocr.cpp:run_full)

**Difference from the normal text pipeline**: skip
`sort_quad_boxes_reading_order`. Ring text has no meaningful
top-to-bottom-then-left-to-right order; sorting that way actually
hurts downstream consumers who want the multiset of recognized
characters.

`is_seal` is the new Engine flag (bool, default false). The CLI
auto-detects seal mode from the det model name:

- `PP-OCRv4_mobile_seal_det` → `is_seal = true`
- `PP-OCRv4_server_seal_det` → `is_seal = true`
- anything else → `is_seal = false`

The CLI also accepts an explicit `--seal` flag for overrides (e.g.
`--det-config some_custom_seal_det.json --seal`).

### `build_lines`

No change. The rec score is preserved on the output so downstream
consumers can filter, but the pipeline does not filter itself.

## What's NOT in scope for M4

- **Polar unwrap** (the user's `(中心=多边形质心, 半径=max 顶点距, 角度逆时针展开成矩形条带)`).
  PaddleOCR's *Python* `paddlex` package has a `CurveTextRectifier`
  for severely curved text (Plan B), but the reference **C++** port
  (which is what we mirror) does NOT use it. Paddle's
  `CropByPolys::GetPolyRectCrop` falls back to the same
  min-area-rect + GetRotateCropImage for low-IoU polys too. Our
  baseline measurement (IoU of all 33 seal polys with their
  min-area-rect > 0.7) confirms the simple crop is the right
  operating point for this image set.
- **PaddleXR's `sealing_text_orientation`** (text inside the ring
  vs around it). The PP-OCRv4_mobile_seal_det output is already
  divided into outer-ring / inner-star polys by the network, so the
  crop is naturally per-region.
- **`text_type=seal` config flag** in our ppocr_config. PaddleOCR
  carries this flag because the same pipeline code is reused for
  general OCR; we don't have that reuse here, so the flag is
  implicit-from-name rather than a user-set field.

## Verification

- 33 images at `/root/ocr_test_imgs/seal/*.jpg`
- 8 languages: zh, en, ja, ko, ru, el, plus a couple of mixed-zh
  variants
- Baseline: `/root/ppocr_reference/PP-OCRv4_mobile_seal_det/seal/ocr_results.json`
  (Paddle GPU fp32 polygons + scores)
- GT: `/root/ocr_test_imgs/seal/ground_truth.json` (per-image expected text)
- Metric: **multiset intersection / GT length** (per-image). The
  rec text is split into characters, the GT text is split into
  characters, the intersection is computed as a multiset (e.g.
  "aabbc" ∩ "abbc" = "abbc" of length 4, not "abc" of length 3).
  This makes the metric order-agnostic, which is what seal
  recognition needs (ring text is cyclic).
- Target: **≥ 0.6 multiset match per image, ≥ 0.6 mean across
  images**. PaddleOCR's own seal-rec demo hits ~0.7-0.8 on
  high-quality seal images; our 736-min resize + MNN det numerics
  put us in the same ballpark.
