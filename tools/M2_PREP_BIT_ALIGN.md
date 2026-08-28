# M2-PREP-BIT-ALIGN: three-audit follow-up (decision-maker directive)

## Audit 1: bilexact was NEVER compiled in (CONFIRMED, fixed)

`-DPPOCR_PREP_BILEXACT=0/1` on the cmake command line only set a cache
variable; CMakeLists never attached it as a compile definition. The
source's `#if !defined(PPOCR_PREP_BILEXACT) #define ... 0` fallback made
every build (including all pilot runs) use float bilinear.

Hard evidence: float dumps from "flag=0" and "flag=1" builds were
BIT-IDENTICAL (0 nonzero) for zh/04, ja/01, ja/05.

Fix: CMakeLists.txt now does

```
option(PPOCR_PREP_BILEXACT "..." OFF)
if(PPOCR_PREP_BILEXACT)
    target_compile_definitions(ppocr_core PUBLIC PPOCR_PREP_BILEXACT=1)
endif()
```

Rebuilt and re-dumped: real bilexact differs from float by 33K px (zh/04),
48K px (ja/01); still bit-exact on ja/05 (identity resize).

## Re-run with REAL bilexact (binary rebuilt via proper option)

| lang | A (new vs old) | B (MNN CLI vs new) |
|---|---|---|
| zh | 0.0380 | 0.0934 |
| en | 0.0408 | 0.1090 |
| ja | 0.1464 | 0.4104 |

Bilexact is NOT uniformly better than float bilinear (zh improved by
~7e-3, en/ja got worse). It is not the bottleneck either way.

## Audit 2: v3->v4 CER regression root cause (RESOLVED)

NOT m4-seal (its `prob_to_img_w/prob_to_img_h` refactor is main-only;
ws/m2-final-diag still uses `in.ratio_w = resize_w/bgr.w`, mathematically
identical when resize == image size).

The v3 numbers (zh 0.056 / en 0.026 / ja 0.199) were produced by a build
that no longer exists; they are NOT reproducible. Current state:
run-to-run IS deterministic (4 identical runs of ja/05 -> same polys and
texts). The v3-vs-v4 discrepancy on zh/05 (22 boxes `CIYUNSI` vs 21 boxes
`CIYUNSi`) was a different binary + a borderline box flipping across
builds, amplified by string-join CER.

## Audit 3: decisive experiment on ja/05 (bit-exact input)

Feed the SAME input tensor to MNN and paddle.inference direct:

| Level | What compared | Result |
|---|---|---|
| L1 | det prob map MNN vs paddle | max **1.2e-3**, mean 1.1e-7 |
| L2 | DB postprocess boxes (our C++ db_postprocess on both prob maps) | **35 = 35, coordinates BIT-IDENTICAL** (scores differ ~1e-5) |
| L3 | rec text on those identical boxes | identical |

Conclusion: with a bit-exact det input our det→DB→rec path matches
paddle.inference direct at the box level. EXPORT-SWEEP holds.

## TRUE root cause of remaining B-table gap

`cv2.imread` (baseline side) vs `stb_image` (CLI side) JPEG decoders
produce DIFFERENT pixel bytes:

- ja/05.jpg: **169,087 differing pixels (13%), max abs diff = ±3,
  total byte-sum diff 56,481**
- baseline prep chw.sum = -2390465.00 (cv2 bytes)
- CLI prep chw.sum   = -2389505.25 (stbi bytes)
- Same byte source fed to both paths → BIT-EXACT match (ja/05).

This decode noise is upstream of every resize/bilexact change and is the
real source of box-count flips (threshold-boundary polygons) that inflate
string-join CER.

## Next-step options (for decision-maker)

a. **Decode parity**: switch CLI jpeg load to libjpeg-turbo (matches cv2
   bit-for-bit); keep stb for png/webp. Bounded: platform/desktop only.
b. Scoring robustness: per-line matched CER instead of string join.
c. Both: fix decode first, re-measure, then decide on scoring.

## Branch state

ws/m2-final-diag, uncommitted:
- CMakeLists.txt — PPOCR_PREP_BILEXACT now a real compile definition (fix)
- tools/M2_PREP_BIT_ALIGN.md — this file
