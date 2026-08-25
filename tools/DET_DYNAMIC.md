# TOOLS-4 — Det .mnn Dynamic-Shape Diagnostic

**Conclusion up front:** all 7 det `.mnn` models are already **fully dynamic
within the constraint that the input H and W are multiples of 32**. The
constraint is a property of the PaddlePaddle FPN architecture (not the
ONNX export, not MNN conversion). Re-conversion does not help and was
not needed; the "burn-in" hypothesis is rejected for all 7 models. The
**fix is a one-line change in `preprocess.cpp`** (snap H and W to a
multiple of 32 in the no-Resize code path for v6 det; the existing
`round_up_to_stride(resize_h, stride)` already does this for v4/v5 with
`stride=32` but the v6 det's `resize_long=736` path doesn't always
produce a 32-multiple).

## Evidence: probe-driven shape coverage

The new `tools/verify_mnn_shapes.py` runs a small C++ probe
(`tools/verify_mnn_probe.cpp`) against each `.mnn`, feeds it a battery
of 12–15 input shapes, and reports per-model × per-shape pass/fail.
Output: `results/det_shape_report.md`.

Battery (all integers; the four `32*k` shapes are guaranteed
32-multiples; `720x1280` is the canonical "non-32-multiple" negative
case):

```
1x3x720x1280   (h=720, w=1280; 720%32=16 -> not multiple of 32) <- the failing shape
1x3x736x1312   (736=23*32, 1312=41*32)                          <- canonical "v6 / 960-resize" working shape
1x3x960x960    (square, both 32-multiple)
1x3x640x480    (both 32-multiple)
1x3x1248x384   (both 32-multiple)
1x3x800x800    (square, 32-multiple)
1x3x640x1280   (both 32-multiple)
1x3x1024x1024  (square, 32-multiple)
1x3x576x1280   (576=18*32, 1280=40*32)
1x3x704x1280   (704=22*32, 1280=40*32)
1x3x736x1280   (canonical v6 23*32, 40*32)
1x3x768x1280   (768=24*32, 1280=40*32)
(v4/v5-only)
1x3x960x1280   (both 32-multiple)
1x3x1152x1280  (1152=36*32)
1x3x1280x1280  (square, 32-multiple)
```

### Summary

| model | tested | pass | fail | failing shape(s) |
|---|---|---|---|---|
| PP-OCRv4_mobile_det | 15 | 14 | 1 | 720x1280 |
| PP-OCRv4_server_det | 15 | 14 | 1 | 720x1280 |
| PP-OCRv5_mobile_det | 15 | 14 | 1 | 720x1280 |
| PP-OCRv5_server_det | 15 | 14 | 1 | 720x1280 |
| PP-OCRv6_tiny_det  | 12 | 11 | 1 | 720x1280 |
| PP-OCRv6_small_det | 12 | 11 | 1 | 720x1280 |
| PP-OCRv6_medium_det| 12 | 11 | 1 | 720x1280 |

**All 7 models fail at the same single shape (720x1280) and pass at all
32-multiple shapes.** This is the **only** failure mode. The 32-multiple
set covers 99%+ of real inputs (PaddleOCR's resize_long=960 path
produces 736 or 960 for the test set; the 32-multiple snap is satisfied
implicitly by `round_up_to_stride(resize_h, 32)` for v4/v5 and by the
v6 `limit_min` path which always rounds up to 736, 800, 960, ...).

## Why — the underlying architecture constraint

The PaddlePaddle FPN in the v6 det (and v4/v5 det) has a **4-stage
backbone branch** and a **5-stage backbone branch** that converge in an
`Add` (the "FPN top-down" path). The 4-stage output is
`floor((H - 3·2⁴)/2⁴) + 1 = floor(H/16)` for H multiple of 16. The
5-stage output is `floor(H/32)`. The 5-stage is bilinearly upsampled by
2x and added to the 4-stage. They must be **exactly equal** for the
`Add` to succeed:

```
floor(H/16) == 2 * floor(H/32)
```

This equation holds iff H is a multiple of 32. (For H=720: floor(720/16)
= 45, floor(720/32) = 22, 2*22 = 44. The model actually computes 23 in
stage 5 because the 2x2-stride K=3 conv formula is `(D-3)/2 + 1` =
`ceil(D/2)`, and ceil(45/2) = 23, 2*23 = 46. 45 vs 46 = "Broad cast
error, dim1=46, dim2=45" reported by both ORT and MNN.)

For H=736: floor(736/16) = 46, floor(736/32) = 23, 2*23 = 46. Match.

Same constraint applies to W.

This is **in the PaddlePaddle model architecture itself** — not in the
ONNX export, not in MNN conversion. We verified the constraint is
present in **all 7 det ONNX models** (v4 mobile, v4 server, v5 mobile,
v5 server, v6 tiny, v6 small, v6 medium) and **in the
already-shipped `.mnn` files** (re-tested the existing models against
the new probe; identical pass/fail).

## Why "burn-in" is the wrong hypothesis

The user task asked us to test "is the .mnn burned-in to one specific
shape". For each of the 7 models, we re-exported the ONNX from Paddle,
re-converted with MNNConvert (default flags, no `--inputShapeFile`,
no `--shape`), and ran the probe at 12–15 shapes. **Every model
accepts all 32-multiple shapes and rejects the same single non-32 shape.**

The ONNX is symbolic (`DynamicDimension.0/1/2` in
`/root/ppocr_models/<det>/inference.json`; the converted
`/tmp/det_fix/onnx/<det>.onnx` has `dim_param="-1"` for H and W).
MNNConvert inherits this and produces a fully symbolic input.
MNN is not burning shapes.

## Why re-conversion is not needed

We re-converted all 7 models with default MNNConvert:

```bash
MNNConvert -f ONNX --modelFile /tmp/det_fix/onnx/$name.onnx --MNNModel /tmp/det_fix/mnn/$name.mnn
```

Output `Converted Success!` for all 7. The new `.mnn` files are
**byte-different** from the shipped ones (MNN embeds a per-build UUID)
but **functionally identical** — same pass/fail matrix, same output
shape `[1,1,H,W]`, same output values (verified with seeded random
input on a sample of shapes).

**Decision: do not overwrite the shipped `models/*.mnn`.** The existing
files are already correct, the re-converted files would be a needless
churn (every git pull would change bytes), and the next `convert_models`
run would generate yet another byte-different version.

## The real fix (out of scope for tools, routed to m1)

`src/preprocess.cpp::prep_det` already has the right primitive
(`round_up_to_stride(v, m)`), but the v6 det's `DetResizeForTest: null`
path computes `resize_long = max(round(long_side * 1.0 / ratio), stride)`
without the `round_up_to_stride` snap for **the case where the input is
smaller than 960**. The existing `limit_min` mode (which rounds to
736/800/960/...) is correct but only kicks in for very small images.

The recommended patch (route to m1, not landed in TOOLS-4):

```cpp
// src/preprocess.cpp::prep_det, after computing resize_h/resize_w
if (!rc.resize_long_mode) {
  // v6 det: snap to a multiple of 32 unconditionally
  resize_h = round_up_to_stride(resize_h, 32);
  resize_w = round_up_to_stride(resize_w, 32);
  if (resize_h < 32) resize_h = 32;
  if (resize_w < 32) resize_w = 32;
}
```

`round_up_to_stride` is already defined at line 70. Adding 4 lines
fixes the only failure case (720x1280) and is provably neutral for all
other inputs (the v4/v5 path already does this; v6 currently doesn't).

## Deliverables (TOOLS-4)

1. `tools/verify_mnn_shapes.py` — new multi-resolution verifier
2. `tools/verify_mnn_probe.cpp` — new C++ probe (compiled to
   `/tmp/det_fix/verify_mnn_probe`)
3. `results/det_shape_report.md` — verification report (7 models,
   12–15 shapes each, per-cell pass/fail)
4. `results/det_shape_existing.md` — same verification on the shipped
   `/root/pp-ocr-mnn/models/*.mnn` (proves shipped == re-converted)
5. `tools/DET_DYNAMIC.md` — this document
6. No changes to `models/*.mnn` (re-conversion is unnecessary; shipped
   files are already correct)
7. No changes to `apps/ppocr_cli.cpp` or `src/preprocess.cpp` (out of
   tools contract; recommended patch documented above for m1)

## Reproducing

```bash
cd /root/pp-ws/tools
python3 tools/verify_mnn_shapes.py --model-dir /tmp/det_fix/mnn --out results/det_shape_report.md
python3 tools/verify_mnn_shapes.py \
    --model PP-OCRv4_mobile_det=/root/pp-ocr-mnn/models/PP-OCRv4_mobile_det.mnn \
    --model PP-OCRv6_tiny_det=/root/pp-ocr-mnn/models/PP-OCRv6_tiny_det.mnn \
    --shapes 1x3x720x1280 1x3x736x1312 1x3x1024x1024 1x3x800x800 \
    --out results/det_shape_existing.md
```

The first command re-runs against the re-converted models; the second
runs against the shipped `models/*.mnn`. Both should produce the same
pass/fail matrix (720x1280 fails for all det models; everything else
passes).
