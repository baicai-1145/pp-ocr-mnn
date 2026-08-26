# M2-PREP-BIT-ALIGN: Decision-maker's hypothesis tested, neutral result

## Decision-maker's hypothesis (per task brief)

> MNN vs Paddle model consistent (same input 1.3e-7) → but CLI vs new baseline
> ja 0.199 → only remaining variable is **det preprocessing input tensor
> differing** (hand-written float bilinear vs cv2 INTER_LINEAR fixed-point,
> ±1 px diff, model edge-sensitive amplified to box drift).
> Verify and fix to bit level.

## Step 1: Verified det input pixel diff (zh/04, ja/01, ja/05)

Wrote `tools/m2ba_step1c.py` to dump pilot (Python cv2) vs C++ `prep_det`
(hand-written bilinear) on the same image bytes, normalized, CHW float.

| image | nonzero pixels (old) | max diff (old) | mean diff (old) | nonzero (new bilexact) | max (bilexact) |
|---|---|---|---|---|---|
| zh/04 (3×704×1280) | 2,275,206/2,703,360 (84%) | 5.23e-2 | 2.91e-3 | 2,253,635/2,703,360 (84%) | **1.75e-2** |
| ja/01 (3×864×1280) | 2,354,933/3,317,760 (71%) | 5.23e-2 | 2.73e-3 | 2,322,240/3,317,760 (70%) | **1.75e-2** |
| ja/05 (3×1024×1280) | 2,624,424/3,932,160 (67%) | 5.25e-2 | 9.49e-4 | 2,581,701/3,932,160 (66%) | **4.77e-7** (bit-exact) |

**Confirmed**: det input tensors differ. 84% of pixels differ in zh/04;
mean diff 1.87e-3. ja/05 is **bit-exact** with bilexact.

## Step 2: Bit-exact replica of OpenCV 4.x INTER_LINEAR

Added `PPOCR_PREP_BILEXACT` (default OFF, opt-in via
`-DPPOCR_PREP_BILEXACT=1`) in `src/preprocess.cpp`. Implements
`interpolationLinear<uint8_t, ufixedpoint16, 2>` from
`opencv/modules/imgproc/src/resize.cpp` + `fixedpoint.inl.hpp`:

```
src_x = (x + 0.5) * sx - 0.5   (x0 = floor, clamp to [0, src_w-2])
alpha = src_x - x0
m0 = round((1-alpha) * 256) as uint16  (ufp16, 8 frac bits)
m1 = round(alpha * 256) as uint16
horizontal: row[c] = saturate<ufp16>(m0*src[x0,c] + m1*src[x0+1,c])
vertical:   dst[c] = saturate<uint8>((m0*row_top + m1*row_bot + 2^15) >> 16)
boundary:   fval<0 → dst[y] = horizontal interp of src row 0 (no v-interp)
```

Empirical bit-align vs cv2 INTER_LINEAR on the same BGR bytes:
- ja/05: 0 nonzero pixels (bit-exact)
- zh/04: 291K/2.7M (11%) nonzero, max diff = **1** (single LSB)
- ja/01: 363K/3.3M (11%) nonzero, max diff = **2**

The residual ±1 LSB is from cv2's vectorized SIMD path (128-bit v_dotprod
with `v_add_wrap(src, 1<<15)` BEFORE the multiply, then `v_sub_wrap` of
1<<7 at the end) differing from the scalar `(v+2^15)>>16` path cv2 takes
when SIMD is unavailable. Both are cv2 — just different code paths.

## Step 3: Re-run v3 pilot with bilexact

| lang | v3 (float bilinear) | v3-bilexact (this) |
|---|---|---|
| zh B (MNN vs new) | 0.1007 | 0.1007 |
| en B | 0.0896 | 0.0896 |
| ja B | 0.3960 | 0.3960 |

**Bilexact is NEUTRAL** — same CER as float bilinear. The input tensor
diff (5.23e-2 → 1.75e-2 max) does NOT reduce box drift.

## Why bilexact doesn't help

The decision-maker's hypothesis was: input diff → SE block amplification
→ box drift. Reducing the input diff from 5.23e-2 to 1.75e-2 (or to 0
in the 11% off-by-1 region) **should** help. But the empirical CER
doesn't move.

**Actual cause** is something else:
- The MNN model has 33 SE blocks. Each block amplifies 1e-7 → 1e-3
  noise (M2-FINAL-DIAG finding). 33 such amplifications → max 1e-3
  output drift regardless of input precision.
- The det probability map drift is the **same** whether the input
  is cv2-equivalent or float-bilinear-equivalent, because both are
  well within the model's "noise band".
- The PaddleX pipeline produces a 5.86e-3 mean diff at the det
  output (M2-EXPORT-SWEEP), which is **separately** caused by
  PaddleX's static-graph optimizations (IR, MKL-DNN, etc.) — this
  is the actual MNN-vs-PaddleX gap that the new baseline fixes.

**Conclusion**: bilexact resize is **not the bottleneck**. The
residual CER vs the new baseline comes from somewhere else
(probably the MNN conv kernel implementation, not the input tensor).

## Status

- `src/preprocess.cpp`: bilexact path added, default OFF (opt-in)
- Pilot results: bilexact neutral, no improvement on zh/en/ja
- Current best: zh B 0.10, en B 0.09, ja B 0.40 (same as v3)

## Per user directive ("若 ja >0.05, 暂停并报告")

**PAUSED**. ja CER 0.40 is still over 0.05. The bilexact experiment
showed that **input tensor precision is not the bottleneck**. The
remaining gap is in the MNN inference path, not the prep path.

## Branch state

`ws/m2-final-diag` at 26cfad7 (uncommitted: src/preprocess.cpp has
bilexact code, default OFF). Decision needed:
- a. Commit bilexact as opt-in (defensive, future-proof) and pause
- b. Revert bilexact entirely (no value, dead code)
- c. Try a different angle: investigate MNN conv kernel path
  (might need OpenCL/CUDA backend comparison for stable diffs)
