# pp-ocr-mnn tools (`ws/tools` executor branch)

Python tooling for the PP-OCR → MNN port: model conversion, registry
generation, reference matrix runner, and CER scoring.

Module owners per `docs/CONTRACT.md`: `tools` = this directory.

## Files

```
tools/
  extract_dict.py     parse PaddleOCR inference.yml → dict + det/rec/cls params
  convert_models.py   full pipeline paddle→onnx→mnn for 30 models
  run_reference.py    drive ppocr_cli across the reference matrix
  score.py            CER per cell; report.md + exit code

tests/
  test_tools.py       unit + integration tests (27 tests, stdlib only)

configs/              generated per-model config JSONs + registry.json
```

## Quick start

```sh
# 1) convert all 30 models (reuses existing models/_onnx/*.onnx and .mnn)
python3 tools/convert_models.py            # 30/30 in ~7s (cached)
python3 tools/convert_models.py --force    # re-convert .mnn for every model
python3 tools/convert_models.py --only PP-OCRv4_mobile_rec

# 2) run reference matrix (requires ppocr_cli built by m1)
python3 tools/run_reference.py --cli build-tools/ppocr_cli --jobs 4
python3 tools/run_reference.py --cells '^PP-OCRv6_' --only-combo ...  # filter

# 3) score predictions
python3 tools/score.py                      # full OCR matrix
python3 tools/score.py --strip              # strip cells (ta/te)
python3 tools/score.py --seal               # emit M4 N/A stub
python3 tools/score.py --all                # all of the above
```

Exit code from `score.py` is **0 if all cells PASS** (CER ≤ 0.05), **1 if any FAIL**.

## Catalog (30 models)

7 det + 2 seal det + 20 rec + 1 cls (`PP-LCNet_x1_0_textline_ori`).

| Group | Models |
|---|---|
| det  | PP-OCRv4_{mobile,server}_det, PP-OCRv5_{mobile,server}_det, PP-OCRv6_{tiny,small,medium}_det |
| seal | PP-OCRv4_{mobile,server}_seal_det |
| rec  | PP-OCRv4_{mobile,server}_rec, PP-OCRv4_server_rec_doc, PP-OCRv5_{mobile,server}_rec, PP-OCRv6_{tiny,small,medium}_rec, en_PP-OCRv{4,5}_mobile_rec, arabic/cyrillic/devanagari/el/eslav/korean/latin/ta/te/th_PP-OCRv5_mobile_rec |
| cls  | PP-LCNet_x1_0_textline_ori (`~/.paddlex/official_models/...`) |

## Verification

```sh
python3 -m py_compile tools/extract_dict.py tools/convert_models.py \
    tools/run_reference.py tools/score.py
python3 tests/test_tools.py          # 27 tests; ALL GREEN
```

## Schema reference

Configs are emitted per `docs/CONTRACT.md` `Model config JSON` section.
Registry is a flat name → {file, sha256, bytes, url, type} map; mirror base
URL is provided at runtime via the `PPORC_MNN_MIRROR` env var.

## Notes

* All Python is stdlib-only (no numpy / no external deps).
* `convert_models.py` reuses `models/_onnx/*.onnx` when present; only the
  cls model is converted from scratch via `paddle2onnx` against
  `~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori`.
* `score.py` ships a self-implemented Wagner–Fischer Levenshtein (no numpy).
