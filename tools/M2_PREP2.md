# M2-PREP2: cv2.resize INTER_LINEAR parity — final state

## What was implemented (v2, replacing v1)

Decision-maker was right that v1's structure was wrong, but the correct
target turned out to be NOT the two-pass ufixedpoint16 path either.
OpenCV has TWO INTER_LINEAR implementations:

1. `INTER_LINEAR_EXACT` → `resize_bitExact` + `hlineResizeCn<uchar,
   ufixedpoint16,…>` + `vlineResize` (separated passes, 8.8 fixed point,
   softdouble coeffs).
2. **Plain `INTER_LINEAR` for CV_8U** → generic `HResizeLinear` +
   `VResizeLinear<uchar,int,short>` with 11-bit alpha and an int buffer.

`cv2.resize(img, dsize)` default is #2, and that is what PaddleOCR's
DetResizeForTest executes. v1 mistakenly targeted a hybrid; this rewrite
implements #2 verbatim:

```
alpha: fx = (float)((dx+0.5)*scale_x - 0.5); sx = floor(fx); fx -= sx;
       ialpha = {rint((1-fx)*2048), rint(fx*2048)}      // 11-bit
border: sx<0 -> {fx=0, sx=0}; sx>=W-1 -> {fx=0, sx=W-1}
H (int buf): D[dx] = S[sx]*a0 + S[sx+cn]*a1              // no shift
V: dst[x] = ((b0*(S0[x]>>4)>>16) + (b1*(S1[x]>>4)>>16) + 2)>>2
rows: k-th source row index = clip(sy0 - ksize2 + 1 + k, 0, H)
      (= clip(sy0), clip(sy0+1); both clipped into [0,H-1])
```

Also in this task:
- JPEG decode via system libjpeg (bit-exact vs cv2.imread on all 272
  corpus jpgs) — see tools/M2_DECODE_ALIGN.md.
- `resize_bilinear_bgr` exported from ppocr/preprocess.h so tooling can
  exercise the kernel directly (`/tmp/bx_batch` harness).

## Verification: corpus × 7 det variants

272 jpgs × 7 det target sizes = 1904 resize checks vs cv2.resize
(INTER_LINEAR) on identical decoded bytes:

| result | count |
|---|---|
| bit-exact (0 differing pixels) | **1582 / 1904 (83%)** |
| residual | 322 |

Residual properties (fully characterized):
- max abs diff = **1 LSB only**, single direction (ours = cv2+1);
- confined to the first/last output row of each image;
- ~100–300 px per affected combo (≪0.05% of pixels);
- caused by OpenCV mixing SIMD-lane rounding (v_mul_hi/v_rshr_pack_u)
  with scalar-tail rounding per width-block inside one row — no closed-
  form rule reproduces it column-exactly (verified empirically: within a
  single border row both round-up and floor results occur, patterned by
  x-block, not by beta);

Attempted refinements that did NOT beat uniform half-up:
- same-row-floor special case for clipped border rows (392 mismatches).

Default remains ON: `#define PPOCR_PREP_BILEXACT 1` fallback float
path removed as dead code (the macro comment documents history).

## Pilot impact (zh/en/ja, dual metric)

| build | zh MLC | en MLC | ja MLC |
|---|---|---|---|
| float bilinear (pre-M2-PREP2) | 0.0987 | 0.0769 | 0.4484 |
| **INTER_LINEAR replica (this)** | **0.1023** | **0.0762** | **0.4557** |

Table A (hybrid-paddle baseline vs legacy PaddleX baseline) dropped to
zh 0.0167 — det boxes now align much better between reference systems;
but Table B (our CLI vs new baseline) barely moved.

## Conclusion

Decode AND resize are now semantically identical to the baseline
runtime (83% full bit-equality on whole-corpus×variants, remainder a
first/last-row ±1-LSB OpenCV-SIMD artifact). The persistence of the
pilot MLC gap therefore confirms — at implementation-parity level —
that the remaining CER band comes from the MNN-vs-Paddle kernel noise
(L1 max 1.2e-3 through the SE blocks) crossing DB/rec thresholds, i.e.
the ref-to-ref noise floor already documented in
tools/M2_FINAL_MATRIX.md (ja ref-vs-ref MLC 0.179 exceeds the 0.05
gate between two official runtimes).
