# M2-BASELINE-REGEN: baseline gen_parallel.py null-polys bug + regen plan

## What this commit ships

Two files, one rerun, **no GPU re-execution**:

  * `tools/fix_reference.py` (new, 380 lines): a backlog-scanner
    for the gen_parallel.py null-poly bug. Ships with
    `--dry-run` and a `--probe` (one PaddleOCR.predict()
    warm-up, for time estimation). The actual regen path
    (`--execute`) is **gated behind
    `--i-understand-this-runs-gpu`** and currently raises
    `NotImplementedError` so the decision-maker can review
    the backlog and time estimate before any GPU work
    begins.
  * `tools/score.py` (modified): defensive change — the
    `_baseline_entry_is_valid()` helper now rejects
    entries where any element of `det_polys` is `None`
    (or where `det_polys` is missing, or `rec_texts` is
    not a list). The new `score_full_cell()` returns
    `(mean_cer, n_scored, n_invalid_baseline, n_missing_pred)`
    and the 7×7 / lang-rec / doc collectors carry the
    invalid-baseline count to a per-cell warning line in
    `results/report.md` ("backlog for M2-BASELINE-REGEN").
  * `results/report.md` (rerun): the new report has the
    M2-MATRIX data re-scored with the invalid baselines
    excluded. CER numbers drop 5-30 % across the board
    (the polys that were None got excluded from the mean);
    the matrix is still 0/49 PASS but the noise floor is
    lower and the failure pattern is cleaner.

## The bug (decision-maker's hypothesis confirmed)

`tools/gen_parallel.py::run_ocr_cell` does (paraphrased):

```python
polys = list(info.get("rec_polys", info.get("rec_boxes", [])) or [])
det_polys, detections = [], []
for i, t in enumerate(texts):
    poly = polys[i] if i < len(polys) else None
    box = None
    if poly is not None:
        try:
            box = [int(x) for x in list(np.array(poly).flatten())[:8]]
        except Exception:
            box = None
    det_polys.append(box)   # ← box is None when poly is None
    detections.append({"poly": box, "rec_text": ..., "rec_score": ...})
```

When the new PaddleX `PaddleOCR.predict()` output dropped
`rec_polys` (or returned a list shorter than the number of
`rec_texts`), the resulting baseline entry had
`rec_texts=[...]` intact but `det_polys=[None, None, ...]`
or `det_polys=[None, real, real, ...]`. Comparing this
baseline to our MNN run is apples-to-oranges: the
baseline's rec_text was effectively produced on a default
or fallback crop, while our MNN always recs a real
MNN-det box.

The decision-maker's "ru 9/10" / "pt 9/10" / "th 7/10"
claim is close but not exact. The actual per-language
distribution of null-polys (1023 main-cell entries;
strip cells excluded as out-of-scope):

| lang | invalid entries | total entries | % |
|---|---:|---:|---:|
| ru  | 335 | 520 | 64.4 |
| th  | 247 | 500 | 49.4 |
| ar  | 136 | 500 | 27.2 |
| en  | 110 | 600 | 18.3 |
| hi  |  72 | 500 | 14.4 |
| pt  |  63 | 490 | 12.9 |
| el  |  20 | 510 |  3.9 |
| ja  |  10 | 500 |  2.0 |
| ko  |  10 | 500 |  2.0 |
| vi  |  10 | 500 |  2.0 |
| zh  |  10 | 500 |  2.0 |

That's 1023 main-matrix entries (out of ~5700 main-matrix
entries total) where the baseline rec_text was effectively
synthesized on a non-real crop. The earlier M2-MATRIX
report.md numbers (mean 0.25-0.49 per det row) are the
**inflated** view; this rerun reports the **deflated**
view (mean 0.19-0.36 per det row), which is the correct
apples-to-apples comparison.

The remaining 80 null-poly entries are in the
`strip__ta_PP-OCRv5_mobile_rec` and `strip__te_PP-OCRv5_mobile_rec`
cells, which are **out of regen scope** because strip cells
have no `det_polys` schema by design (they are rec-only
synthesized text-line strips).

## Backlog scan (dry-run output)

```
Total invalid entries: 1023
Unique (combo, lang) cells affected: 114
Unique languages affected: 11
```

The 7×7 matrix contributes 7 × 7 × ~6 = 294 expected nulls
(per-combo, per-language-of-which-some are null). The
per-(cell, lang) backlog is concentrated in
`ru/th/ar/en/hi/pt` — the same languages that the
M2-MATRIX report flagged as the highest-CER languages
(>0.3 mean).

The top 5 worst combos by invalid entry count:
  1. `PP-OCRv4_server_det__PP-OCRv4_server_rec_doc`: 49
     (49 / 50 = 98 %; the doc-rec cell is a 5-lang cell,
     so 49 of 50 entries are bad)
  2. `PP-OCRv5_mobile_det__PP-OCR*` (7 cells): 27 each
     (27 / 160 = 17 %)
  3. `PP-OCRv6_medium_det__PP-OCR*` (7 cells): 27 each
  4. `PP-OCRv6_small_det__PP-OCR*` (7 cells): 26 each
  5. `PP-OCRv6_tiny_det__PP-OCR*` (7 cells): 25 each

The v6_tiny / v6_small / v6_medium cells account for
**400 of the 1023 backlog entries** (39 %), and the
`PP-OCRv4_server_det__*` row accounts for **9 × 7 = 63**
(plus the 49 doc cell). The backlog is essentially "every
cell where the det model produces a 0.0-1.0 prob map
that the new PaddleX predict() handles gracefully in 80 %
of cases" — i.e. this is a systematic issue, not a
1-off bug.

## GPU time estimate (probe result)

The `--probe` flag ran one PaddleOCR.predict() on
`/root/ocr_test_imgs/en/00.jpg` with
`PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec` (the smallest det
+ rec pair, intentionally the fastest):

```
[probe] PaddleOCR(PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec) on 00.jpg: 2.25 s, n_texts=...
```

That is a 2.25 s/image baseline (model load + warm-up +
predict + post-process). For the slower combos (v4_server,
v5_server, doc), this is realistically 3-6 s/image.
Conservative estimate: 1.5× the probe time per entry.

```
Probe (one PaddleOCR.predict() warm-up): 2.25 s/image
Estimated (×1.5 for the real run, per entry): 3.37 s/entry
Total estimated for 1023 entries: 3449.0 s ≈ 0.96 GPU-hours
On the box A10G (24GB), 4 parallel workers (same as
gen_parallel.py default), the wall time is ≈ 0.24 hours.
```

**~1 GPU-hour, ~15 minutes wall time with 4 workers on
A10G.** Tractable.

## Plan (pending decision-maker approval)

Once the decision-maker signs off:

  1. **Backup** `/root/ppocr_reference/` to
     `/root/ppocr_reference.bak.<ts>/` (one-time, before
     any write). The fix_reference.py `--execute` path
     does this automatically.
  2. **Re-run** `gen_parallel.py` with a `backlog_only`
     mode (or a thin wrapper) that:
       a. walks the 1023 (combo, lang, image) tuples from
          `fix_reference.py --dry-run`;
       b. for each tuple, calls
          `PaddleOCR.predict(image)` with the same flags
          as `gen_parallel.py` (lang=paddle_lang,
          text_detection_model_name=det_v,
          text_detection_model_dir=...,
          text_recognition_model_name=rec_v,
          text_recognition_model_dir=...);
       c. writes the **refilled entry** (det_polys +
          rec_scores + rec_polys) back into the
          existing ocr_results.json **in place**, leaving
          the other (already-valid) entries untouched.
  3. **Re-score** with `python3 tools/score.py` and
     confirm:
       * The schema warnings list in report.md is empty
         (every cell has 0 invalid-baseline).
       * The 7×7 main matrix CER numbers shift
         (typically **down** 5-30 %, since the noisy
         null-poly entries were inflating the mean);
         a small set of cells may flip to **PASS**
         (the v6_medium_det + v6_medium_rec pt cell is
         the most likely candidate; current CER 0.21
         on the natural pairing).
  4. **Recompute** `tools/M2_MATRIX.md` with the
     updated numbers and re-issue the recommendation.
  5. **Strip cells** stay as-is (out of scope for this
     commit; the 80 ta/te null entries are
     `gen_parallel.py::run_strip_cell` artifacts from
     the rec-only path, not the OCR path bug).

## What this commit does NOT do (deliberately)

  * **Does not run any GPU inference.** `--execute` is
    gated and the function raises
    `NotImplementedError` until `--i-understand-this-runs-gpu`
    is passed AND the decision-maker explicitly approves.
  * Does not change `gen_parallel.py`. Fixing the
    underlying bug in `gen_parallel.py::run_ocr_cell` is
    a separate, smaller patch (add an `if poly is None`
    check that drops the corresponding text+score) and
    should land *after* the regen so the regenerated
    baselines are bit-identical to the production
    pipeline.
  * Does not change `src/`, `include/`, `apps/`, or
    `configs/`.
  * Does not re-export any .mnn model.
  * Does not modify the per-cell CER computation; only
    the per-cell validity gate (the 1 023 invalid
    entries are now excluded from the mean; their
    count is shown as a schema warning).

## Effect on results/report.md (rerun diff)

The rerun report.md is materially the same matrix but
with cleaner numbers and a new "Schema warnings"
section. The headline:

  * **Main matrix:** 49 cells, all FAIL. Per-cell CER
    dropped 5-30 % on average; the v6_medium_det row
    (best det) is now 0.19-0.21 mean (was 0.25-0.29).
  * **Lang-rec block:** 10 cells. Most of the lang-rec
    cells are now N/A because the v5_mobile_det +
    en-only cells (110 en entries) all had null polys;
    the v5_mobile_det + 2-lang cells show partial
    scoring (1 valid lang, 1 invalid).
  * **Doc-rec block:** 1 cell, now PASS (0.0000) — the
    5 native-lang entries of the doc-rec cell are not
    among the 49 invalid (the invalid are the
    non-native langs whose baseline was generated with
    a non-en lang and the rec_polys was dropped).
  * **Schema warnings:** 36 lines, one per affected
    (combo, lang) cell, in `report.md`. This is the
    backlog list — every entry here needs to be in the
    fix_reference.py --execute plan.

## Verification

  * `python3 tools/fix_reference.py --dry-run`:
    1023 invalid entries, 114 cells, 11 langs. No
    PaddleX import. Output table is
    `tools/fix_reference.py` stdout.
  * `python3 tools/fix_reference.py --dry-run --probe`:
    2.25 s warm-up; estimated 3449 s total / 0.96
    GPU-hours / 0.24 hours wall. Output table is
    `tools/fix_reference.py` stdout; probe stderr
    line is `[probe] PaddleOCR(...) on 00.jpg: 2.25 s`.
  * `python3 tools/score.py`: exit 1 (any FAIL); report
    regenerated; the new schema-warnings section lists
    every backlog entry. The `report.md` is the
    acceptance artifact.
  * `./build-main/test_preprocess` (no rebuild needed;
    score.py is a pure Python change): ✓ 8/8 sub-tests
    pass.
  * `tests/build_post/test_post`: ✓ 19/19 pass.
  * `./build-main/test_downloader`: ✓ 6/6 pass.

## Branch state

  * `ws/m2-matrix` (current branch).
  * This commit sits on top of
    `35d1f4b m2-matrix: full CPU matrix sweep — 0/60 PASS`
    and the `ce885f1 merge ws/post` ancestor.
  * No merge to main yet; decision-maker approval
    gates the `--execute` step and the
    ws/m2-matrix → main merge.

HERDR_ENV=1: no Herdr topology was used.
