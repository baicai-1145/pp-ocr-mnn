# M3-CLS — textline orientation classifier

Optional PP-LCNet_x1_0_textline_ori step wired between det-crop and rec.

## When to use

Enable cls when the input images may contain upside-down text. The cls
model predicts 0° vs 180° per det crop and rotates the crop 180° in place
before it reaches rec. PaddleClas::TextRecRotator and ppocr::predict_cls do
the same thing in the reference pipeline.

Disable cls (the default) for upright-only inputs (most camera frames,
scanned documents). cls adds ~10 ms per crop and the cls model file
(6.7 MB) on disk.

## Enabling

```bash
# Models: PP-LCNet_x1_0_textline_ori.mnn + configs/PP-LCNet_x1_0_textline_ori.json
# Both ship in the local model_dir.

ppocr_cli \
  --image img.jpg \
  --det-config configs/PP-OCRv4_mobile_det.json \
  --rec-config configs/PP-OCRv4_mobile_rec.json \
  --cls-config configs/PP-LCNet_x1_0_textline_ori.json \
  --model-dir /path/to/models
```

The CLI parses `--cls-config` and sets `cfg.cls_name` to the basename
(without `.json`). The engine then loads `model_dir/<cls_name>.mnn`
and the matching config.

## Wiring

- `src/ppocr.cpp::run_cls_sync(e, crops, labels, scores, ms)` is the
  new step. It takes the warped (and 90°-CCW-rotated if tall) crops
  produced by `run_rec_sync`, runs the cls model on each, and rotates
  180° in place any crop the model labels as 180°.

- `run_rec_sync` was extended to take a `cls_ms` out param and a
  pointer for the per-crop cls labels (optional). The cls step is
  invoked between the warp loop (which already includes the 90° CCW
  rotation for tall crops per paddlex convention) and the rec chunk
  loop.

- `Engine::last_result.cls_ms` is now populated and the C ABI's
  `ppocr_result.cls_ms` mirrors it. `total_ms` includes the cls
  contribution when cls is on.

- The async worker (`run_async`) and the boxes-json path
  (`run_with_boxes`) both go through `run_rec_sync` so they pick up
  cls automatically.

- cls is **off by default**. When `cfg->cls_name` is null, `e.cls`
  is null and `run_cls_sync` is a no-op (no per-crop overhead, just
  a single `assign` to the labels vector).

## Output of the cls model

The converted `PP-LCNet_x1_0_textline_ori.mnn` produces `[1, 2]`
softmax (verified). Output name `fetch_name_0`. Index 0 = "0_degree",
index 1 = "180_degree" (per the model's labels in
`e.cls_cfg.cls.labels`).

`run_cls_sync` does `argmax` per crop. If 0, no rotation. If 1,
`rotate_180_inplace` reverses the BGR buffer in place (the same as
`np.rot90(k=2)` and PaddleClas::TextRecRotator).

## Validation

`tools/cls_validation.py` runs the full pipeline against the 16-lang
test set and reports:

- (a) `zh/04.jpg` rotated 180°: cls-off returns empty text, cls-on
  recovers `["ALLEY", "SOLINSKY"]`.
- (b) 5 images x (upright + rot180): % of upright texts recovered on
  the rot180 image with cls on. Latin scripts (zh/04, en/04, de/04)
  recover 50-100%; CJK/Arabic dense scenes (zh/05, ja/04) recover
  13-30% because det itself finds different boxes on the rotated
  image, NOT because cls mispredicts.

`tests/test_cls.cpp` (build target `test_cls`, runs under
`ctest`) wires the cls through the C ABI:

- Two engines (cls on, cls off) on the same image.
- Asserts `cls_ms > 0` for the on engine and `cls_ms == 0` for the
  off engine.
- Asserts `n_lines` matches and texts match (no rotation on the
  upright).

## Latency

Measured on this CPU (AVX2): ~10 ms/crop. For 2 boxes the
`cls_ms` is ~20 ms total, dominated by two `set_input_float` +
single-image forward passes. A future batched forward could halve
this; the cls is `[1,3,80,160]` fixed-shape (single batch) because
that is what the PPLCNet ONNX export produces; a batched re-export
+ re-convert would unlock the speedup.

## Files touched

```
src/ppocr.cpp           +160 lines   (run_cls_sync, rotate_180_inplace,
                                      run_rec_sync cls hook, all callers
                                      updated for cls_ms)
tests/test_cls.cpp      new          (cls integration test)
tools/cls_validation.py new          (end-to-end driver, 16-lang)
CMakeLists.txt          +9           (test_cls target)
```
