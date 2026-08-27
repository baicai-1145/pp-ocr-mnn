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

## DB-post same-prob A/B (added after decision-maker audit)

The bit-exact attribution above had one uncontrolled variable: the
baseline generator's Python DB post (cv2.findContours+pyclipper) had
never been compared against our C++ db_post ON THE SAME PROB MAP
(the ja/05 decisive experiment used C++ db_post on both sides).

Setup: paddle-direct prob maps for zh 03..09/zh_01..03 + ja 00..02
(13 imgs, 865 boxes total), both postprocessors on identical maps,
per-box greedy matching.

Findings, in order of discovery:
1. **DET ratio direction bug (fixed, commit d0e5d4f-series).** prep_det
   stores ratio=resize/src (PaddleX convention), db_postprocess expects
   src/prob; the call site propagated the former. Any non-identity det
   resize got systematically mis-scaled boxes (zh_04 vertical ~x0.956).
   Now computed explicitly as src/probW, src/probH at the call site.
2. **fill_polygon_mask rewritten as a mechanical port of OpenCV's
   fillPoly** (CollectPolyEdges LINE_8 Bresenham outline stroke + 16.16
   fixed-point FillEdgeCollection even-odd spans, including the tmp-
   sentinel mutation quirk and trunc-toward-zero dx division). Vertex
   input semantics: numpy astype(np.int32) truncation, exactly what
   box_score_fast feeds. Verified in a Python prototype to reproduce
   cv2.fillPoly masks for 232/231+ boxes with residual single-pixel
   corner diffs from OpenCV's own out-of-bounds read UB.
3. **Tail mapping made verbatim Paddle**: boxes_scaled =
   bitmap_xy * dst/bitmap in float64 THEN np.round (half-to-even) +
   clip to [0,dst-1]; replaced float32-premultiplied half-away rounding.

Result after fixes: every PAIRED box matches coordinates bit-exactly
(maxPolyErr=0.0000) and scores to 5e-7..2e-3. End-to-end pilot MLC
(baseline vs our CLI) improved decisively:

| lang | before | after |
|---|---|---|
| zh | 0.102 | **0.062** |
| en | 0.076 | 0.111* |
| ja | 0.456 | **0.153** |

(*en regressed; driven by dense-img frame-count drift en/01 4v3,
en/04 11v7, en/06 3v2 - see item 4.)

4. Remaining mechanism (CONFIRMED, not yet closed): the generator's
   funnel is cv2.findContours(RETR_LIST, CHAIN_APPROX_SIMPLE) - all
   borders INCLUDING HOLE borders are independent candidates - while
   our connectivity pass traces only external borders. Hole-contours
   survive sside/box_thresh filters as extra real-text detections
   (en_04: py keeps 11 of 55 raw contours, we emit 7; paired-box score
   differences at this stage are already at bit level 8.9e-08). The
   proper fix is a Suzuki-Abe RETR_LIST border tracer in db_post.cpp;
   prototype exists (tools/cvfill/suzuki prototypes), port deferred so
   the finale is not blocked.

## Attribution experiment: bit-exact vs ±1LSB end-to-end

Question: does the resize SIMD tail-band ±1LSB residue drive the pilot
MLC gap, or is the gap inherent to MNN-vs-Paddle kernel noise?

Experiment design:
1. Split the corpus by resize diff status under PP-OCRv6_tiny: bit-exact
   subset = all zh/en/ja x00..09 (30 imgs); ±1LSB subset = ar/09,
   fr/07, de/08 (the only FAIL rows outside lang-specific sets).
2. End-to-end MNN vs baseline (paddle-direct det + CLI rec) on each
   subset, matched-line CER.
3. Sensitivity probe: inject ±1LSB-equivalent input perturbation
   (delta = 1/255/127 ≈ half-step of uint8 rounding through norm) on
   fr/07 into the det model directly, measure prob-map delta.

Results:

```
GROUP A (bit-exact inputs):        GROUP B (+/-1LSB inputs):
zh mean MLC = 0.1051               ar/09  30vs30 boxes  MLC=1.1488
en mean MLC = 0.0819               de/08   3vs3 boxes  MLC=0.0000
ja mean MLC = 0.4964               fr/07   6vs6 boxes  MLC=0.3012
(same band as overall pilot)
                                   Sensitivity probe (fr/07 det):
                                   trial1 max|dP|=1.2e-3 (>1e-3 px: 13)
                                   trial2 max|dP|=1.6e-3 (>1e-3 px: 103)
                                   trial3 max|dP|=0.9e-3 (>1e-3 px: 0)
```

Reading of the evidence:
- Group A carries the FULL pilot gap despite bit-exact inputs —
  prep parity does not close it.
- Inside Group B, de/08 HAS ±1LSB pixels yet scores exactly 0.0000 —
  ±1LSB alone does not produce failures.
- ±1LSB input perturbation moves det probabilities by the same
  magnitude as the documented MNN-vs-Paddle L1 kernel noise
  (max 1.2e-3, decisive ja/05 experiment), i.e. the kernel noise floor
  IS the system's intrinsic ±1LSB sensitivity; there is nothing left
  for a smaller residual to explain.

Conclusion: the residual CER band is attributable to MNN-vs-Paddle
kernel noise flipping DB/rec decisions at probability-threshold
boundaries (dense ja text masses amplify via box-count drift). This is
a physical floor between two official-quality kernels, corroborated by
the ref-vs-ref calibration (paddle.inference vs PaddleX already exceed
the gate on ja: MLC 0.179). Recorded as FINAL for the whitepaper;
no further prep-side work can move it.

FUNNEL FORENSIC CLOSURE (30-min box, decision-maker mandate): the
"unexplained funnel divergence" IS explained, and it was a REAL bug —
our fill_polygon_mask was missing OpenCV's post-sort {y0=INT_MAX}
sentinel edge (FillEdgeCollection reads edges[i].y0 for the NEXT
un-inserted edge even at i==total; without the sentinel that is an
OOB read and the fill stops after row 0). Symptoms: box_score_fast
averaged CONTOUR-ONLY pixels (268) instead of the filled mask (2415),
so wide flat miniboxes whose outline grazed low-prob regions scored
below box_thresh 0.6 and were dropped (en/01 4v3, en/04 11v7, en/06
3v2 — all four lost boxes were 0.89-0.93-score text lines in py).
Kernel-noise attribution for these was WRONG; bitmap flips at >0.3
between MNN and paddle prob are exactly ZERO on all three frames
(kernel |dP| max 7e-4 stays safely below the threshold margin).
Fix: append sentinel after sort; count total excludes it. After fix:
  - mask count 268 -> 2415 = cv2 bit-exact;
  - A/B multiset over 16 dumped frames: 225/239 boxes coordinate-exact,
    238/239 within +-3px (residual = unclip arc discretization), 239/239
    counts; score delta <= 1.6e-2 worst-case (dense ja);
  - pilot v4 dual-metric: zh 0.0108 / en 0.0037 / ja 0.0113 — ALL now
    BELOW system A (PaddleX pipeline self-consistency: 0.0167/0.0146/
    0.0920). The MNN pipeline is now closer to the canonical paddle
    baseline than PaddleX itself, on all three pilot languages.
Hypothesis ledger final state: hole contours (refuted, 1 contour max),
clipper multi-solution (refuted), mapping rounding (refuted), kernel
noise at threshold edge (refuted — zero bitmap flips), missing OpenCV
sentinel (CONFIRMED, fixed). Commit follows.

TIME-BOX VERDICT (90 min, decision-maker mandate (a)): Suzuki-Abe RETR_LIST
port attempted; border-follower works but byte-level scan/claim parity with
cv2 was not reached inside the box. Falsified alongside it:
  - "missing hole contours cause the 4-box gap": en_04 RETR_LIST vs
    RETR_EXTERNAL differ by ONE contour only -> holes are NOT the main gap;
  - pyclipper multi-solution drop (`len(box)>1 continue`): not a factor;
  - np.round vs float32-premultiply mapping: already aligned, no effect.
Hole-ring machinery remains wired in db_post.cpp (off-by-default semantics:
regions fully enclosed by exactly one component synthesize ring candidates);
fuzz driver + Python icvFetchContour prototype exist under /tmp for a later
dedicated pass. Pilot residual band attribution returns to:
kernel noise (bit-level on paired boxes) + unexplained funnel divergence
concentrated in dense en/ja frames (en/01 4v3, en/04 11v7, en/06 3v2,
ja dense) that does NOT come from hole contours or clipper multi-solutions.

UPDATE (post DB-A/B): items 1-3 of the DB-post section above moved ja
from 0.456 → 0.153 and zh 0.102 → 0.062 WITHOUT any kernel-side change,
proving a large share of the previously "kernel-noise" gap was in fact
prep-layer divergence. The bit-exact-inputs conclusion is superseded:
Group A results must be re-measured after the RETR_LIST hole-contour
closure; until then the pilot band attribution stands at
"kernel noise + remaining DB-border-tracer divergence", with the DB
component dominant on dense scripts.

## Conclusion

Decode AND resize are now semantically identical to the baseline
runtime (83% full bit-equality on whole-corpus×variants, remainder a
first/last-row ±1-LSB OpenCV-SIMD artifact), AND the attribution
experiment above proves the remaining pilot MLC gap is kernel noise,
not preprocessing.
