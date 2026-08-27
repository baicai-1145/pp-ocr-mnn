# pp-ocr-mnn

**PP-OCR full family on MNN** — the complete PaddleOCR PP-OCR model family
(v4 / v5 / v6; mobile / server / tiny / small / medium, plus language-specific
rec models, doc-rec, seal-det and the PP-LCNet textline-orientation cls),
running through [MNN](https://github.com/alibaba/MNN) on Windows / Linux /
macOS / Android / iOS, with automatic CPU / CUDA / OpenCL / Vulkan / Metal /
CoreML / NNAPI backend selection. Exposed as a multi-instance thread-safe
C ABI with zero platform `#ifdef`s in the business code.

- 7 det models × 7 rec models × 16 languages acceptance matrix (811 cells, see
  `tools/M2_*.md`), every variant supported from one codebase
- Preprocess/postprocess hand-written in C++ (no OpenCV): DB postprocess with
  contour tracing, minAreaRect, Clipper unclip and perspective warp; CTC decode
  for 95–18708-entry dictionaries; polar unwrap for seals
- Bit-exact-where-it-matters: preprocessing reproduces PaddleX pipeline
  semantics (det resize limit-min/stride-32 banker's rounding; rec [3,48,320];
  BGR mean/std), JPEG decode is cv2-parity via libjpeg-turbo

## Quick start

### Build (Linux desktop, CPU)

```sh
git clone --recurse-submodules <repo> pp-ocr-mnn && cd pp-ocr-mnn
# prebuilt third_party/MNN/build/libMNN.a is picked up automatically;
# without it CMake falls back to a CPU-only add_subdirectory build
cmake -S . -B build-main -DCMAKE_BUILD_TYPE=Release
cmake --build build-main -j
```

Produces `ppocr_cli` and `libppocr_core.a` (+ `test_post`, `test_preprocess`,
`test_cls`, `test_downloader`).

### CLI: single image

```sh
./build-main/ppocr_cli \
  --image /path/to/img.jpg \
  --det-config configs/PP-OCRv6_tiny_det.json \
  --rec-config configs/PP-OCRv6_tiny_rec.json \
  --model-dir ./models \
  --backend auto --threads 4 --json out.json
```

`out.json` contains
`{"image":…,"backend":"cpu","lines":[{"poly":[x0,y0,…,x3,y3],"text":…,"score":…},…],"ms":{…}}`
with polys in original-image pixel coordinates. Add `--profile` for per-stage
timings, `--det-only` to skip rec, `--boxes-json boxes.json` to skip det and
rec caller-supplied polys (see `docs/USAGE.md`).

### CLI: batch a directory

```sh
./build-cuda/ppocr_cli \
  --batch-dir /path/to/imgs \
  --det-config configs/PP-OCRv6_tiny_det.json \
  --rec-config configs/PP-OCRv6_tiny_rec.json \
  --model-dir ./models --backend cuda --threads 4 \
  --workers 4 --pin-offset 2 --json out.json
```

Engine-per-worker pool; output is a JSON array plus a trailing `_bench` object
(`throughput_fps`, per-image e2e, failures). `--pin-offset 2` pins each worker
to 2 dedicated CPUs (taskset semantics; +25% throughput on a contended host).
Full flag reference: `docs/USAGE.md`.

### C ABI: three lines

```c
#include "ppocr/ppocr.h"

ppocr_config cfg = {0};                 /* zero-init = all defaults   */
cfg.model_dir = "/path/to/models";      /* .mnn files + configs/      */
ppocr_engine* e;  char err[256];
ppocr_create(&cfg, &e, err, sizeof(err));

ppocr_result* r;
ppocr_run_file(e, "img.jpg", &r);       /* r->lines[i].text / .poly   */

ppocr_destroy(e);
```

A complete compilable example: `examples/c_api_demo.c`. Defaults are
`PP-OCRv6_tiny_det` + `PP-OCRv6_tiny_rec` on CPU with 4 threads; every field
(model names, backend, threads, rec batch, cls, seal mode, profiling, warmup,
download policy) is overridable on `ppocr_config` — full field table in
`docs/USAGE.md`.

## Model support matrix

30 models total (registry `configs/registry.json`, sizes are `.mnn` bytes):

| Group | Models | Size range |
|---|---|---|
| det (7) | PP-OCRv4_{mobile,server}_det, PP-OCRv5_{mobile,server}_det, PP-OCRv6_{tiny,small,medium}_det | 1.7 MB (tiny) – 95 MB (v4 server) |
| seal det (2) | PP-OCRv4_{mobile,server}_seal_det | — |
| rec (20) | PP-OCRv4_{mobile,server}_rec, PP-OCRv4_server_rec_doc, PP-OCRv5_{mobile,server}_rec, PP-OCRv6_{tiny,small,medium}_rec, en_PP-OCRv{4,5}_mobile_rec, {arabic,cyrillic,devanagari,el,eslav,korean,latin,ta,te,th}_PP-OCRv5_mobile_rec | 4.4 – 83 MB |
| cls (1) | PP-LCNet_x1_0_textline_ori (0°/180°) | 6.7 MB |

### Backend status

| Backend | Status |
|---|---|
| CPU | ✅ validated (all rounds; M3-CUDA report: CPU numerics stable across MNN 2.9.1→3.6.1, mean diff 0.0) |
| CUDA | ✅ validated (M3-CUDA gate: CUDA vs CPU mean diff 0.000055 << 0.002; fp32/Normal precision; all PERF rounds) |
| OpenCL / Vulkan | ⚙️ compiled into the MNN build; **not validated on this host** (M3-CUDA report: OpenCL init fails → CPU fallback; Vulkan segfaults in the NVIDIA loader) |
| Metal / CoreML / NNAPI | ⚙️ enum + platform wiring shipped (`platform/ios`, `platform/android`); requires Apple/Android hardware — M5 scope |

`PPOCR_BACKEND_AUTO` maps to MNN `MNN_FORWARD_AUTO` (best available, CPU
guaranteed). Explicit requests fail with `PPOCR_ERR_BACKEND` if unavailable.
Backend selection lives only in `src/ppocr.cpp::pickBackend()` (CONTRACT rule
#6).

## Performance

EPYC 7502 (14C) + A10G 24G, PP-OCRv6_tiny, CUDA, 4 threads unless noted;
host runs co-tenant load (28–42 during PERF3–6 rounds — absolute FPS between
rounds is not directly comparable; deltas were measured interleaved):

| Round | Headline |
|---|---|
| baseline | single-image CUDA 7.50 FPS (133 ms) |
| PERF2 | single-image 9.45 FPS (+26%); db_post 207→27 ms |
| PERF3 | batch zh w4 15.9 FPS (+53% vs 1 worker) |
| PERF4 | dense-workload w4 +49%; first-image 104→81 ms (create-time warmup); rec batch 16 |
| PERF5 | JPEG decode 13.4→4.6 ms (libjpeg, 272/272 images cv2-parity); medium-model rec batch hint 8 (+24% dense) |
| PERF6 | e2e −7…−10% (zero-copy image view + fused det resize/normalize); zh w4 +25% with `--pin-offset 2` |

**Cumulative: single image 133 → 64–81 ms; batch zh (4 workers) up to
~15 FPS; dense (283-box) pages ~2.6–4.5 FPS w4.** CPU t4: 545→~517 ms
(forward-bound). Full per-round methodology and A/B data:
[`tools/PERF_BASELINE.md`](tools/PERF_BASELINE.md).

All performance changes were gated on output identity (bit-exact text+poly on
5 sentinel images; the libjpeg decode change is separately gated by decode
parity + CER).

## Architecture

Platform-agnostic core (`include/ppocr/` + `src/` — zero `#ifdef <platform>`);
platform code only under `platform/`; backend choice only in
`src/ppocr.cpp::pickBackend()`.

```
                    ┌──────────────────────────────────────────┐
                    │              ppocr_engine                │
   BGR image ──────►│ decode (stb / libjpeg, cv2-parity)       │
                    │   ├► det: prep_det ─► MNN session ─► db_post ─┐
                    │   │                                            │ boxes
                    │   │            ┌── cls (optional): rotate 180°┤
                    │   ▼            ▼                               ▼
                    │   crop+warp (perspective / polar-unwrap in seal mode)
                    │   └► rec: prep_rec_line (batch) ─► MNN session ─► ctc_decode
                    │ lines (reading-order sort; skipped in seal mode)  │
                    └──────────────────────────────────────────┘
```

| Module | Files | Role |
|---|---|---|
| C ABI / engine | `include/ppocr/ppocr.h`, `src/ppocr.cpp` | lifecycle, pipeline orchestration, async worker, profiling, `pickBackend()` |
| Preprocess | `src/preprocess.cpp` | det resize (Paddle `DetResizeForTest` ports, banker's rounding), fused resize+normalize, rec keep-ratio [3,48,320] batching, cls |
| MNN session | `src/mnn_session.cpp`, `include/ppocr/mnn_session.h` | RAII wrapper, input resize, host↔device staging (CUDA/OpenCL/Vulkan), `PPOCR_MNN_SESSION_MODE` diagnostic knob |
| DB postprocess | `src/postprocess/db_post.cpp` | prob→contours→minAreaRect→Clipper unclip→box filter→reading order; bbox-limited mask rasterization |
| CTC decode | `src/postprocess/ctc_decode.cpp` | blank=0 collapse, per-model dictionary (95–18708 entries), space append |
| Geometry | `src/postprocess/geometry.cpp` | minAreaRect, point ordering, perspective warp (cv2 INTER_CUBIC BORDER_REPLICATE port), polar unwrap (seals) |
| Image I/O | `src/image.cpp` | stb_image; JPEG via system libjpeg-turbo when available (`PPOCR_HAVE_LIBJPEG`, bit-exact with cv2.imread) |
| Config | `src/config.cpp`, `include/ppocr/config.h` | model config JSON parsing (det/rec/cls params, dict, `rec_batch_hint`) |
| Downloader | `src/downloader.cpp`, `include/ppocr/downloader.h` | registry.json, sha256, curl fetch, atomic cache (`tools/AUTO_DOWNLOAD.md`) |
| CLI | `apps/ppocr_cli.cpp` | single-image + batch-dir modes, JSON output, profiling |
| Platforms | `platform/{desktop,android,ios}` | install rules + C example; JNI wrapper (arm64-v8a, NDK r27); Objective-C / Swift Package wrapper |

MNN is an unmodified submodule (`third_party/MNN`, pinned); its own build
directory is local-only. Model weights are not committed; `tools/convert_models.py`
generates `.mnn` + per-model configs + `registry.json` from the PaddleOCR
models (paddle→onnx→mnn).

## Benchmarks & verification

- **Acceptance matrix**: 7 det × 7 rec × 16 languages + lang-rec + doc-rec +
  seal + strip cells (811 total), CER ≤ 0.05 per cell vs versioned baselines
  generated with canonical `paddle.inference` (methodology:
  [`tools/M2_BASELINE_REGEN.md`](tools/M2_BASELINE_REGEN.md),
  [`tools/M2_EXPORT_SWEEP.md`](tools/M2_EXPORT_SWEEP.md)). Current matrix
  status: `tools/M2_FINAL_MATRIX.md` (pending — final matrix run in progress
  at the time of writing; see `tools/M2_MATRIX.md` for the current sweep).
- **Known numeric property**: MNN vs Paddle det prob-map mean diff ≈ 5.9e-3,
  rooted in paddle2onnx conversion semantics, not our kernels — full per-layer
  attribution in [`tools/M2_FINAL_DIAG.md`](tools/M2_FINAL_DIAG.md),
  [`tools/M2_NUM.md`](tools/M2_NUM.md), kernel sweep in
  [`tools/M3_KERNELS.md`](tools/M3_KERNELS.md).
- **Scoring**: `tools/score.py` (image-joined CER; differences vs PaddleOCR
  `RecMetric` documented in [`tools/CER_VS_OFFICIAL.md`](tools/CER_VS_OFFICIAL.md)).
  The acceptance gate uses the baselines' own generating-runtime outputs for
  like-for-like comparison.
- **Regenerate runs**: `tools/run_reference.py` drives `ppocr_cli` over the
  matrix; `python3 tools/score.py` emits `results/report.md` (exit code 0/1).
- **Unit tests**: `tests/test_post.cpp` (DB post + geometry + CTC + polar
  unwrap, 20 asserts), `test_preprocess`, `test_cls`, `test_downloader`;
  `ctest` + `./build-*/tests/test_post` green.
- Deep-dives: det geometry ([`docs/DET_GEOMETRY.md`](docs/DET_GEOMETRY.md)),
  seal pipeline ([`docs/SEAL_PIPELINE.md`](docs/SEAL_PIPELINE.md)), cls
  ([`tools/M3_CLS.md`](tools/M3_CLS.md)), auto-download
  ([`tools/AUTO_DOWNLOAD.md`](tools/AUTO_DOWNLOAD.md)).

## License

Same as the project. MNN and PaddleOCR model licenses apply to their
respective artifacts.
