# pp-ocr-mnn tools (`ws/tools` executor branch)

Python tooling for the PP-OCR → MNN port: model conversion, registry
generation, reference matrix runner, CER scoring, and baseline audits.

Module owners per `docs/CONTRACT.md`: `tools` = this directory.

## Files

```
tools/
  extract_dict.py     parse PaddleOCR inference.yml → dict + det/rec/cls params
  convert_models.py   full pipeline paddle→onnx→mnn for 30 models
  run_reference.py    drive ppocr_cli across the reference matrix
  score.py            CER per cell; report.md + exit code
  cer_audit.py        re-score baseline vs sibling .jpg.txt; baseline sanity
  verify_mnn_shapes.py  multi-resolution probe-driven verifier for det .mnn
  verify_mnn_probe.cpp  C++ probe (built by verify_mnn_shapes, run via MNN)

  README.md           this file
  CER_VS_OFFICIAL.md  analysis: our CER vs PaddleOCR RecMetric
  REC_ONLY.md         rec-only contract gap (CLI patch request to m1)
  DET_DYNAMIC.md      det .mnn dynamic-shape diagnostic (TOOLS-4)

tests/
  test_tools.py       unit + integration tests (60 tests, stdlib only)

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

## End-to-end run → score

Below is the canonical flow once `ppocr_cli` is built (m1 owner delivers
the binary at `build-tools/ppocr_cli`; until then `run_reference.py` will
report the missing CLI cleanly via stderr + non-zero rc).

```sh
# A) build everything (one-time)
cmake -S . -B build-tools -DCMAKE_BUILD_TYPE=Release
cmake --build build-tools -j          # produces build-tools/ppocr_cli

# B) run the full reference matrix serially (no parallelism)
python3 tools/run_reference.py --cli build-tools/ppocr_cli

# C) same, with parallel image processing (4 worker threads per combo,
#    combos themselves still run serially to keep log lines readable)
python3 tools/run_reference.py --cli build-tools/ppocr_cli --jobs 4

# D) process a subset: just the v6 tiny / chinese cell
python3 tools/run_reference.py \
    --cli build-tools/ppocr_cli \
    --only-combo PP-OCRv6_tiny_det__PP-OCRv6_tiny_rec \
    --langs zh \
    --jobs 8

# E) debug a single language across all combos
python3 tools/run_reference.py \
    --cli build-tools/ppocr_cli \
    --cells '__' \
    --langs ko \
    --jobs 2

# F) score all predictions (default writes results/report.md)
python3 tools/score.py

# G) score a subset only (e.g. en + zh) — pass an explicit --results-dir
#    if you ran run_reference.py with a non-default --results-dir.
python3 tools/score.py --results-dir results | head

# H) strip / seal / full matrix
python3 tools/score.py --strip           # ta/te rec-only cells
python3 tools/score.py --seal            # M4 stub: N/A for seal det
python3 tools/score.py --all             # combined report

# I) baseline audit (sanity check the reference itself)
python3 tools/cer_audit.py --out results/cer_audit.md
# exit 0, writes a markdown report. The per-image CER looks high because
# the .jpg.txt siblings are Wikimedia file metadata, not in-image text;
# read tools/CER_VS_OFFICIAL.md for the full caveat.
```

### `run_reference.py` flags (most-used)

| flag | default | meaning |
|---|---|---|
| `--cli PATH` | `./build-tools/ppocr_cli` | path to the binary |
| `--configs-dir DIR` | `./configs` | where `configs/<name>.json` live |
| `--results-dir DIR` | `./results` | where `<combo>/<lang>/pred.json` is written |
| `--cls-config PATH` | off | enable textline-orientation cls |
| `--backend` | `cpu` | `auto` / `cpu` / `cuda` / `opencl` / `vulkan` |
| `--threads N` | `4` | MNN intra-op threads |
| `--jobs N` | `4` | per-combo image-parallel workers |
| `--cells REGEX` | all | restrict combo names by regex |
| `--only-combo NAME` | none | restrict to a single combo (repeatable) |
| `--langs a,b,c` | all | restrict languages |
| `--rec-only` | off | rec-only mode; **CLI does not yet support this** — see `tools/REC_ONLY.md`. Implies `--include-strip`. |
| `--dry-run` | off | print plan only, do not run CLI |

### `score.py` flags (most-used)

| flag | default | meaning |
|---|---|---|
| `--results-dir DIR` | `./results` | dir containing `<combo>/<lang>/pred.json` |
| `--report PATH` | `<results-dir>/report.md` | output matrix |
| `--report-md PATH` | `<results-dir>/report.md` | alias of `--report` |
| `--threshold F` | `0.05` | per-cell mean CER cutoff |
| `--strip` | off | score `strip__*` cells vs `strip_gt.json` |
| `--seal` | off | emit M4 N/A stub |
| `--all` | off | full + lang-rec + doc + strip + seal in one report |
| exit code | — | `0` = all PASS, `1` = any FAIL |

The default `score.py` (no flag) writes a 7×7 main matrix (rows=det,
cols=rec, cell=lang-averaged CER + status) plus a lang-rec block and
a doc-rec block. Use `--all` to also include strip and seal.

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
    tools/run_reference.py tools/score.py tools/cer_audit.py
python3 tests/test_tools.py          # 30 tests; ALL GREEN
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
* `cer_audit.py` is a diagnostic; it never blocks a commit. See
  `tools/CER_VS_OFFICIAL.md` for the formula difference vs PaddleOCR
  `RecMetric` (per-image join vs per-line; `lev/len(t)` vs
  `lev/max(len(p),len(t))`; spaces preserved by default).

