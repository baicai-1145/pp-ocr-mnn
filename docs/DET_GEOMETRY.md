# Det Geometry — PaddleX Baseline Contract

This document is the **authoritative geometric contract** that the
C++ preprocessing / postprocessing pipeline must reproduce to match
the 811-cell baseline at `/root/ppocr_reference/`. It was added in
M2-FIX after the empirical probe (`tools/probe_baseline_geom.py`)
established that PaddleX's runtime overrides the per-model
`inference.yml` resize and postprocess fields with a single uniform
configuration. The values below are **NOT** derived from
`inference.yml`; they come from a direct measurement of the
PaddleX baseline generator (the same code that produced the 811
cells under `/root/ppocr_reference/`).

## TL;DR — the uniform det config

All 7 main-matrix det models and the 2 seal det models use the same
PaddleX-side det config (the seal model uses a slightly different
resize because PaddleX's seal pipeline does not override it).

### 7 main det models

`PP-OCRv4_mobile_det`, `PP-OCRv4_server_det`, `PP-OCRv5_mobile_det`,
`PP-OCRv5_server_det`, `PP-OCRv6_tiny_det`, `PP-OCRv6_small_det`,
`PP-OCRv6_medium_det`:

```yaml
det:
  thresh: 0.3
  box_thresh: 0.6
  unclip_ratio: 1.5
  max_candidates: 1000
  resize:
    mode: limit_min
    limit_side_len: 64        # NOT 736 — PaddleX default, not PaddleOCR yml
    resize_long: 960          # inherited, unused at this mode
    stride: 32                # 32-align
    max_side_limit: 4000
```

### 2 seal det models

`PP-OCRv4_mobile_seal_det`, `PP-OCRv4_server_seal_det`:

```yaml
det:
  thresh: 0.2                # seal yml value, not overridden
  box_thresh: 0.6
  unclip_ratio: 0.5          # seal yml value, not overridden
  max_candidates: 1000
  resize:
    mode: resize_long
    limit_side_len: 736
    resize_long: 736
    stride: 128              # 128-align (type-2)
    max_side_limit: 4000
```

## Why this matters (root cause of M2-PIPE CER regression)

The M2-PIPE commit (`f555d93`) reproduced the `LimitMin(736,
32-align)` resize that is the **PaddleOCR v3 reference**, but the
811-cell baseline at `/root/ppocr_reference/` was generated with
**PaddleX's OCR pipeline**, which has its own YAML default
(`paddlex/configs/pipelines/OCR.yaml`):

```yaml
SubModules.TextDetection:
  limit_side_len: 64
  limit_type: min
  max_side_limit: 4000
  thresh: 0.3
  box_thresh: 0.6
  unclip_ratio: 1.5
```

That YAML is what the baseline generator (see
`tools/gen_parallel.py:run_cell`) loaded. The per-model
`inference.yml` is loaded by the model's predictor but the
`SubModules.TextDetection` block in the OCR pipeline takes
**precedence** because PaddleX's pipeline init computes
`text_det_limit_side_len = text_det_config.get("limit_side_len", 960)`
and the resulting value (64 from the default, 960 if overridden
elsewhere, or whatever the user passed via CLI) is the one that
ends up in the predictor's `self.limit_side_len`.

For PaddleOCR's `PaddleOCR()` API the CLI params default to `None`,
the structure override sets `None` in `SubModules.TextDetection`, and
PaddleX falls back to the OCR.yaml default → 64. The
`inference.yml` `DetResizeForTest` and `PostProcess` blocks are
loaded into the per-model predictor's local config but **the OCR
pipeline passes its own `limit_side_len`/`limit_type`/etc. to the
predictor ctor, overriding the per-model values for the
resize/PostProcess semantics**.

## Empirical geometry table

Measured by `tools/probe_baseline_geom.py` against
`/root/ocr_test_imgs/zh/04.jpg` (1280x720) on the 7 main det models
through the PaddleOCR API. **All 7 land on 1280x704** when
fed 1280x720, which is `round(720/32)*32 = 22*32 = 704` (banker's
rounding, half-to-even).

| det model            | limit_side_len | limit_type | max_side_limit | thresh | box_thresh | unclip_ratio | resize out (1280x720→) | n_polys |
|----------------------|----------------|------------|----------------|--------|------------|--------------|------------------------|---------|
| PP-OCRv4_mobile_det  | 64             | min        | 4000           | 0.3    | 0.6        | 1.5          | 1280x704               | 2       |
| PP-OCRv4_server_det  | 64             | min        | 4000           | 0.3    | 0.6        | 1.5          | 1280x704               | 2       |
| PP-OCRv5_mobile_det  | 64             | min        | 4000           | 0.3    | 0.6        | 1.5          | 1280x704               | 5       |
| PP-OCRv5_server_det  | 64             | min        | 4000           | 0.3    | 0.6        | 1.5          | 1280x704               | 3       |
| PP-OCRv6_tiny_det    | 64             | min        | 4000           | 0.3    | 0.6        | 1.5          | 1280x704               | 2       |
| PP-OCRv6_small_det   | 64             | min        | 4000           | 0.3    | 0.6        | 1.5          | 1280x704               | 2       |
| PP-OCRv6_medium_det  | 64             | min        | 4000           | 0.3    | 0.6        | 1.5          | 1280x704               | 3       |

Note: the yml value of `unclip_ratio=1.4` for v6 models and the
`box_thresh=0.4` are **silently overridden** by PaddleX. The
empirical v6_tiny_det output for zh/04 matches the baseline
`/root/ppocr_reference/PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec/zh/ocr_results.json`
exactly when we feed the model an input of shape `1,3,704,1280`
with the 0.3/0.6/1.5 postprocess parameters.

## Stride alignment — banker rounding (half-to-even)

PaddleOCR's `DetResizeForTest.resize_image_type0` ends with:

```python
resize_h = max(int(round(resize_h / 32) * 32), 32)
```

`numpy.round` and Python's built-in `round` both use **banker's
rounding** (FE_TONEAREST, half-to-even). We reproduce it with
`std::nearbyint` (default rounding mode is also FE_TONEAREST) and
the formula:

```cpp
int round_up_to_stride(int v, int m) {
  if (m <= 1) return v;
  long q_int = static_cast<long>(std::nearbyint(static_cast<double>(v) / m));
  long r = q_int * m;
  return r < m ? m : static_cast<int>(r);
}
```

Half-value boundaries (tested in `tests/test_preprocess.cpp`):

```
720/32 = 22.5  -> 22*32 = 704  (half-to-even: 22 even)
736/32 = 23    -> 23*32 = 736
944/32 = 29.5  -> 30*32 = 960  (half-to-even: 30 even)
976/32 = 30.5  -> 30*32 = 960  (half-to-even: 30 even)
992/32 = 31    -> 31*32 = 992
```

For `resize_long` (PaddleOCR type 2, used by the seal det) the
alignment is **ceiling**, not round:

```python
resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
```

`prep_det` switches on `rc.mode` and applies the corresponding
formula; see the `if (rc.mode == ...)` block in
`src/preprocess.cpp::prep_det`.

The previous M1 implementation used ceiling for both modes and
that caused a **32 px discrepancy in H** for any 720-pixel-tall
input — 1280x720 became 1280x736 (our path) instead of 1280x704
(baseline). DB contours then landed on a different prob-map row
than the baseline, box positions shifted by tens of pixels, and
the rec crops decoded the wrong text. The M2-PIPE mean CER was
0.78 (zh) and 0.90 (en) almost entirely from this 32 px error.

## What the C++ pipeline must do

1. `prep_det` (in `src/preprocess.cpp`) must read the `resize` block
   from the model config JSON and apply it; do not read
   `inference.yml` directly.
2. The 32-align step must use `round_up_to_stride` (banker's
   rounding) for `limit_min` and `no_resize` modes. For
   `resize_long` it must use `(v + stride - 1) / stride * stride`
   (ceiling).
3. `db_postprocess` must use `thresh=0.3`, `box_thresh=0.6`,
   `unclip_ratio=1.5` for every main det model (the values
   supplied by the new configs). The v6 yml values
   `0.2/0.4/1.4` MUST NOT be used.
4. The seal det models keep their yml values
   (`0.2/0.6/0.5` + `resize_long 736, stride 128`).

## How to regenerate the configs

The configs under `configs/*.json` are emitted by
`tools/extract_dict.py` (one mode) and `tools/extract_dict.py cls`
(for the cls). The det policy lives in
`tools/extract_dict.py::_resize_policy` and `_extract_det`. To
regenerate after a model change:

```bash
cd /root/pp-ocr-mnn
python3 -c "
import sys, hashlib, os
sys.path.insert(0, 'tools')
from extract_dict import extract, write_config
DETS = ['PP-OCRv4_mobile_det','PP-OCRv4_server_det',
        'PP-OCRv5_mobile_det','PP-OCRv5_server_det',
        'PP-OCRv6_tiny_det','PP-OCRv6_small_det','PP-OCRv6_medium_det',
        'PP-OCRv4_mobile_seal_det','PP-OCRv4_server_seal_det']
for n in DETS:
    yml = f'/root/ppocr_models/{n}/inference.yml'
    mnn = f'models/{n}.mnn'
    ext = extract(yml, name=n)
    h = hashlib.sha256()
    with open(mnn,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20), b''): h.update(chunk)
    out = write_config('configs', ext, file=f'{n}.mnn',
                       sha256=h.hexdigest(), bytes_=os.path.getsize(mnn),
                       url=f'{n}.mnn')
    print(out)
"
```

For an end-to-end regeneration including rec dicts, run
`tools/convert_models.py` (which calls the same `extract_dict.py`
internally).
