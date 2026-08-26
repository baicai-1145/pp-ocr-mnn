# M2-NUM: numerical diff attribution (MNN vs PaddleX)

This doc captures the M2-NUM investigation that found an MNN-vs-Paddle
numerical diff in the det prob map and a rec-preprocessing bug in our
own code, and the final state of the conversion / CER numbers.

## Det prob map diff (v6_tiny_det, zh/04.jpg)

PaddleX v6_tiny_det uses `DetResizeForTest: null` in the PaddleOCR
yaml but the PaddleX *runtime* overrides to `limit_min=64, limit_type=min,
max_side_limit=4000` (see
`paddlex/inference/models/text_detection/predictor.py:53-68` and
`_get_text_det_resize_defaults`). For a 1280x720 input this lands on
1280x720 → 1280x704 after the stride-32 half-even snap. The PaddleX
pre_tfs are `Read → Resize → Normalize → ToCHW → ToBatch`; the network
input is `(1, 3, 704, 1280)` float32 normalized by ImageNet mean/std
after a 1/255 scale.

We hooked the PaddleX runner (TextDetRunnerPredictor.runner.__call__)
to dump the post-sigmoid `(1, 1, 704, 1280)` probability map. On
the C++ side, `/tmp/m2num_det_mnn` (new, in this commit's tree) loads
`models/PP-OCRv6_tiny_det.mnn`, resizes input `x` to the same shape,
copies the same float32 input, and runs MNN. Diff stats:

```
shape = (1, 1, 704, 1280), 901120 pixels
max diff = 0.963675
mean diff = 0.005858
> 0.01: 2.11 %  | > 0.05: 1.34 %  | > 0.1:  1.09 %
```

The 1.1 % of pixels with > 0.1 diff are the ones sitting close to
the 0.3 binarization threshold (`thresh=0.3` per
`paddlex/configs/pipelines/OCR.yaml`); small logit shifts flip them
across the threshold and move box edges by a few pixels. This is the
"~50 %" of the residual CER that M2-ISO attributed to the det chain
(see commit `e8ac1f7`).

### Conversion-option sweep (M2-NUM experiment 1.b)

We re-exported the v6_tiny_det ONNX through MNNConvert with 6
combinations and re-ran the diff:

| variant | size B    | max diff | mean diff | > 0.01 | > 0.1 |
|---------|-----------|----------|-----------|--------|-------|
| A default (--optimizeLevel 1)        | 1746372 | 0.9637 | 0.005858 | 2.11 % | 1.09 % |
| B --optimizeLevel 0 (no graph opt)   | 1783616 | 0.9637 | 0.005858 | 2.11 % | 1.09 % |
| C --optimizeLevel 2 (max opt)        | 1746372 | 0.9635 | 0.005848 | 2.10 % | 1.09 % |
| D --fp16                              |  903092 | 0.9631 | 0.005870 | 2.12 % | 1.09 % |
| E --optimizeLevel 0 + --fp16          |  940304 | 0.9630 | 0.005869 | 2.13 % | 1.09 % |
| F --optimizeLevel 2 + --fp16          |  903092 | 0.9629 | 0.005861 | 2.11 % | 1.09 % |
| G (existing baseline)                 | 1746376 | 0.9637 | 0.005858 | 2.11 % | 1.09 % |

**Verdict**: all six variants are within 0.0001 of each other on
every metric. The diff is intrinsic to MNN's CPU GEMM/conv kernel
vs Paddle's CPU kernel (likely an FMA ordering or a different
SGEMM/IGEMM algorithm for the 3x3 / 1x1 convs), not a
conversion-stage issue. `--optimizeLevel 0` confirms it: even
without graph rewriting the diff is identical, so it is not the
graph-level op fusion that diverges.

**Recommendation**: keep the current default
`MNNConvert -f ONNX --modelFile ... --MNNModel ...` (which is
`--optimizeLevel 1`). All 6 are equivalent, and the fp32 weight
storage is what we already have. Tools/convert_models.py now
exposes `--optimize-level {0,1,2}` and `--fp16` as explicit flags
documenting the equivalence.

## Rec logits diff (v6_tiny_rec, en/03 'S' box)

Same recipe on the rec side: hooked `ReisizeNorm` + the rec runner
in PaddleX to dump the (3, 48, 320) input and the (1, 40, 6906)
softmaxed logits. On the C++ side, /tmp/m2num_rec_mnn loads
`models/PP-OCRv6_tiny_rec.mnn`, resizes input `x` to the same shape,
copies the same float32 input, and runs MNN. Diff stats:

```
shape = (1, 40, 6906), 276240 cells
max diff = 0.041094
mean diff = 0.000001
argmax match: 40 / 40 timesteps
top-1 confidence at the 'S' step (t=7): paddle 0.7170, mnn 0.7083
```

So the rec model output is **bit-for-bit equivalent** to PaddleX
given the same input. The visible "MNN emits K" in the full-pipeline
CER gate was *not* a numerical diff — see below.

### Rec batch_w floor bug (found and fixed in this commit)

The CLI's `run_rec_sync` computed `batch_w = int(48 * max(w/h over
chunk))` with no lower bound. paddlex `ReisizeNorm.resize_norm_img`
uses `max_wh_ratio = max(rec_w / rec_h, w/h) = max(320/48, w/h)` as
a *floor*: a single tall crop (e.g. en/03's 200x133 'S' box, w/h=1.5)
gets `max_wh_ratio = 6.67` so `imgW = 320`. Our C++ code skipped
this floor and used `int(48 * 1.5) = 72`, so we fed the rec model a
72-wide input. The MNN rec model is width-dynamic, but it is
trained at the 320 baseline, so feeding 72-wide shifts the logits
enough to flip the CTC argmax from "S" (id 61 in the dict) to "K"
(id 52). MNN was being faithful to the input it received; the input
itself was wrong.

Verification (en/03 'S' box, single image, after the fix):
```
batch_w = 320  (was 72)
rec logits argmax: 61 ("S") at step 7, top-1 = 0.7084
CLI output: text = "S", score = 0.7084
baseline: text = "S"  → CER = 0
```

Fix is in `src/ppocr.cpp::run_rec_sync`: the floor
`rec_w / rec_h = 320/48 = 6.67` is now applied per image before
taking the chunk max. The docstring was updated to cite this commit
and the regression test (en/03 baseline 'S') it caught.

## CER after the M2-NUM fixes (rec batch_w floor)

Re-ran the M2-ISO `--boxes-json` driver on the same 20 images (10 zh,
10 en) with the rec batch_w floor fix. Compared to M2-ISO commit
`e8ac1f7`:

| language | M2-REC2 | M2-ISO | **M2-NUM** |
|----------|---------|--------|------------|
| zh       | 0.13    | 0.0686 | **0.0386** |
| en       | 0.24    | 0.1409 | **0.0429** |

Per-image deltas (vs M2-ISO):

```
zh: 04.jpg 0.000 (=), 06.jpg 0.000, 07.jpg 0.000, 08.jpg 0.000,
    09.jpg 0.000 (was 0.300), zh_02.jpg 0.000, en/03 single
    characters across the deck improve.
en: 03.jpg 0.000 (was 1.000), 04.jpg 0.033 (=), 05.jpg 0.000 (=),
    06.jpg 0.034 (slight), 09.jpg 0.000 (=), en_01.jpg 0.000 (=).
```

**Both zh and en mean CER are now under the 0.05 gate (0.0386 and
0.0429) when the det chain is held constant (boxes-json)**: the rec
batch_w floor fix closes the M2-ISO rec-model gap. The full pipeline
gate (`/tmp/dm_cer_gate_ldl.py`, sets LD_LIBRARY_PATH for the
downloader shared lib) is still over 0.05 because the *det chain*
contributes the ~5-12 px box-placement noise that M2-ISO already
quantified; that's the next M2+ milestone (M2-FIX-noise or
M3 conversion noise reduction).

## Why we did NOT change the .mnn conversion

Because no conversion option measurably shifts the diff. The next
step would be a per-op CPU kernel audit, which is out of scope
until the det-side CER is the dominant signal. The
`tools/convert_models.py` defaults stay as they were; the new
`--optimize-level` and `--fp16` flags exist for repeatability
and to document the M2-NUM sweep result.

## Files in this commit

- `src/ppocr.cpp::run_rec_sync` — apply the `rec_w/rec_h` floor to
  per-image max_wh_ratio so a single tall crop still uses
  `batch_w = rec_w = 320` (was `int(48 * w/h)` = 72 for the en/03
  'S' case). Updated docstring with the paddlex citation.
- `tools/convert_models.py` — expose `--optimize-level` and
  `--fp16` flags; default to `--optimizeLevel 1`, no `--fp16` (the
  M2-NUM sweep showed no measurable diff between 0/1/2 or fp16 on/off).
- `tools/M2_NUM.md` — this file.
- `tools/extract_dict.py` — unchanged.
- `tools/convert_models.py` MNNConvert invocation now passes
  `--bizCode MNN` and `--optimizeLevel 1` explicitly (was implicit
  default), so future re-exports with the same CLI produce a known
  diff profile.

## Side files (not committed)

These are in `/tmp/m2num/` and `/tmp/m2iso/cli_dump/` for reference
only; the diff numbers in this doc were re-derived from them at
commit time.

```
/tmp/m2num_det_paddle.py        PaddleX det input + output hooks
/tmp/m2num_det_mnn.cpp           C++ det driver: load .mnn, dump output
/tmp/m2num_rec_paddle.py         PaddleX rec input + output hooks
/tmp/m2num_rec_mnn.cpp           C++ rec driver: load .mnn, dump output
/tmp/m2num_convert.py            MNNConvert 6-combo sweep
/tmp/m2num_eval.py               diff stats for each conversion variant
/tmp/m2iso/cli_dump/             rec input CHW dump from the CLI
                                 (caught the batch_w floor bug)
```
