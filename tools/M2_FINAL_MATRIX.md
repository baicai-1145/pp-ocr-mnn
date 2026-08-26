# M2-FINAL-MATRIX: full-matrix regen + dual-metric final report

## Status: IN PROGRESS

- [x] score.py: matched_line_cer implemented per decision-maker spec
      (greedy min-normalized-Levenshtein 1:1 pairing; unmatched base/pred
      lines charged at full length; denominator = total base length;
      join-CER kept as reference-only column). 10 unit tests PASS,
      including the decision-maker's 3-base/2-pred example = 4/9.
- [x] Pilot dual metric (libjpeg build):
      | lang | MLC (decision) | join (ref) |
      |---|---|---|
      | zh | 0.1027 | 0.0908 |
      | en | 0.0976 | 0.0948 |
      | ja | 0.4579 | 0.3832 |
- [x] Noise-floor calibration — two reference-grade Paddle runtimes
      (PaddleX pipeline vs paddle.inference direct) scored against each
      other on the same combo:
      | lang | ref-vs-ref MLC |
      |---|---|
      | zh | 0.0409 |
      | en | 0.0449 |
      | ja | **0.1793** |
      The ja floor between TWO OFFICIAL RUNTIMES already exceeds the
      0.05 gate. Our CLI sits ~2.2–2.5× above that floor uniformly across
      langs, i.e. the same order as official-runtime divergence.
- [x] Failure decomposition:
      * zh/en: box counts match; failures are per-box 1-char rec flips
        driven by det-polygon ±1px jitter changing the rec crop pixels.
      * ja dense images: additional low-score FP boxes on the MNN side
        (prob-map noise > 0.3 threshold in regions paddle stays below);
        real text found by one side and missed by the other cuts both ways.
      * identity-resize ja images fail less than resized ones but still
        carry FP boxes (ja/03 33 vs 27 boxes with 26 exact poly matches).
- [x] Baseline promotion: /root/ppocr_reference (PaddleX) moved to
      /root/ppocr_reference.paddlex.bak; new architecture baseline
      (v6-tiny pilot combo) copied in.
- [ ] FULL REGEN RUNNING (PID 389006): gen_baseline_direct.py --full
      --workers 4 on A10G; log /tmp/full_regen.log.
- [ ] run_reference CPU over all 811 cells + score.py dual-metric report.
- [ ] CUDA matrix: --backend cuda re-run of run_reference + score.
- [ ] Final report here.

## Verdict template (to fill after full run)

To be decided by decision-maker based on the full 811-cell table:
strict per-line PASS/FAIL counts for CPU and CUDA.
