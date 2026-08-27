# pp-ocr-mnn usage reference

Everything below is sourced from code and shipped docs: `apps/ppocr_cli.cpp`,
`include/ppocr/ppocr.h`, `include/ppocr/profile.h`, `include/ppocr/config.h`,
`src/config.cpp`, `docs/CONTRACT.md`, `tools/README.md`.

## CLI (`ppocr_cli`)

Two mutually exclusive modes: single image (`--image`) and batch directory
(`--batch-dir`). Errors go to stderr with a non-zero exit code; results are a
single JSON object (single) or a JSON array + `_bench` trailer (batch) on
stdout or `--json OUT`.

### Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--det-config PATH` | (required) | det model config JSON; model name is derived from the file basename; the engine loads `<model_dir>/<name>.mnn` |
| `--rec-config PATH` | (optional) | rec model config; omit or pass `--det-only` for det-only |
| `--cls-config PATH` | off | textline-orientation cls config (`PP-LCNet_x1_0_textline_ori`); rotates upside-down crops 180° before rec |
| `--model-dir DIR` | `./models` | directory containing `.mnn` files + `configs/` |
| `--registry PATH` | `<model_dir>/configs/registry.json` | registry override |
| `--backend B` | `auto` | `auto` \| `cpu` \| `cuda` \| `opencl` \| `vulkan` \| `metal` \| `coreml` \| `nnapi` |
| `--threads N` | 0 (auto) | MNN intra-op CPU threads |
| `--batch N` | 0 | rec batch size override (see `rec_batch_hint` below) |
| `--max-side N` | 0 (from model config) | det max side limit override |
| `--det-only` | off | skip rec; lines get `text:""`, `score` = det box score |
| `--json OUT` | stdout | output file |
| `--time` | off | one-line timing summary to stderr |
| `--profile` | off | append a `"profile"` object (per-stage ms, see below) |

### Single-image mode

| Flag | Meaning |
|---|---|
| `--image IMG` | input image (jpg/png/bmp via stb; JPEG decodes through libjpeg-turbo when compiled in, bit-exact with cv2.imread) |
| `--boxes-json F` | skip det; recognize caller-supplied polygons (M2-ISO error-isolation audit). Flat JSON array, 8 ints per box (TL,TR,BR,BL), original-image pixel coords, caller-chosen order. Requires `--rec-config` |

Exit codes: 0 ok; 1 usage; 2 engine create; 3 run failed; 4 file I/O; 5 malformed boxes file.

### Batch mode (`--batch-dir`)

Engine-per-worker pool: each worker owns an independent `ppocr_engine` (the C
ABI is multi-instance thread-safe), pulls images off an atomic index, runs the
full serial pipeline per image — outputs identical to the single-image CLI.
The output array is sorted by filename and followed by
`{"_bench":{n_images,workers,wall_s,throughput_fps,mean_e2e_ms,max_e2e_ms,failures}}`.
Failed images get `{"image":…,"error":true}`; exit code 5 if any failure.

| Flag | Default | Meaning |
|---|---|---|
| `--batch-dir DIR` | — | process all `*.jpg/.jpeg/.png/.bmp` in DIR (non-recursive) |
| `--workers N` | 2 | number of engine instances |
| `--pin-offset N` | 0 (off) | pin worker *w* to N consecutive CPUs starting at `(w*N) % ncpu` (taskset semantics via `sched_setaffinity`, best-effort). On a 14-core host with 4 workers, `--pin-offset 2` measured +25% (zh) / +9% (dense) under co-tenant load; values that oversubscribe (4 workers × 3 cpus) can regress |
| `--no-warmup` | warmup on | skip the create-time dummy inference (see `warmup` below) |

### Profile output (`--profile`)

Adds `"profile":{...}` with per-stage wall-clock ms (populated only when
requested — zero instrumentation cost otherwise): `decode_ms`,
`det_prep_ms`, `det_run_ms`, `db_post_ms`, `crop_warp_ms`, `rec_prep_ms`,
`rec_run_ms`, `ctc_decode_ms`, `cls_ms`, `e2e_ms`, `create_ms`,
`first_run_ms`, plus `n_boxes`, `rec_batches`, `threads`, `backend`.
Semantics documented in `include/ppocr/profile.h` (e.g. `e2e_ms` ≥ sum of
stages; the difference is glue).

## C ABI

`include/ppocr/ppocr.h` (v1, frozen). Multi-instance thread-safe; synchronous
by default, optional async. All strings UTF-8, copied at create time;
zero-init `ppocr_config` = all defaults.

### Functions

| Function | Meaning |
|---|---|
| `ppocr_version(int*,int*,int*)` | ABI version (currently 0.1.0) |
| `ppocr_create(const ppocr_config*, ppocr_engine**, char* err, size_t)` | resolves models (auto-download if enabled), loads det/rec/cls sessions |
| `ppocr_destroy(ppocr_engine*)` | free engine |
| `ppocr_run(ppocr_engine*, const uint8_t* bgr, int w, int h, ppocr_result**)` | synchronous inference on a BGR interleaved buffer |
| `ppocr_run_file(ppocr_engine*, const char* path, ppocr_result**)` | same, decodes jpg/png/bmp |
| `ppocr_run_async(ppocr_engine*, const uint8_t* bgr, int w, int h, ppocr_callback, void* user)` | optional async; callback receives engine/status/result/user |
| `ppocr_run_with_boxes(ppocr_engine*, const uint8_t* bgr, int w, int h, const int* polys, int n_polys, ppocr_result**)` | skip det, run rec on caller-supplied polygons (8 ints per box, TL,TR,BR,BL) |
| `ppocr_last_profile(const ppocr_engine*)` | pointer to the per-stage profile of the most recent run; NULL when `cfg.profile==0` or before the first run; storage owned by the engine, valid until the next run/destroy |
| `ppocr_status_string(ppocr_status)` | status → string |

`ppocr_result` is owned by the engine and valid **until the next run on the
same engine** — copy out what you need before the next call. `lines[i].poly`
= 8 ints (x0,y0,…,x3,y3) in original-image pixel coords.

Statuses: `PPOCR_OK`, `PPOCR_ERR_PARAM`, `PPOCR_ERR_MODEL` (missing/corrupt/sha
mismatch), `PPOCR_ERR_BACKEND`, `PPOCR_ERR_DOWNLOAD`, `PPOCR_ERR_IO`,
`PPOCR_ERR_OOM`, `PPOCR_ERR_INTERNAL`.

### `ppocr_config` fields

| Field | Default (zero-init) | Meaning |
|---|---|---|
| `model_dir` | env `PPORC_MNN_MODELS` or `./models` | dir with `.mnn` + `configs/` |
| `cache_dir` | env `PPORC_MNN_CACHE` or `~/.cache/ppocr-mnn` | download cache |
| `det_name` | `"PP-OCRv6_tiny_det"` | det model name in registry |
| `rec_name` | `"PP-OCRv6_tiny_rec"` | rec model; `NULL` disables rec (det-only) |
| `cls_name` | NULL (off) | textline-orientation cls model |
| `registry_path` | `<model_dir>/configs/registry.json` | registry override |
| `mirror` | env `PPORC_MNN_MIRROR` | download base URL |
| `backend` | `PPOCR_BACKEND_AUTO` | see backend table in README |
| `num_threads` | 0 (auto) | CPU threads |
| `rec_batch` | 0 | rec batch override; see priority order below |
| `max_side` | 0 (from model config) | det max side limit override |
| `offline` | 0 | 1 = never download, fail if missing |
| `download` | 1 | 0 = disable auto-download |
| `is_seal` | 0 (auto) | 1 = force seal pipeline; 2 = disable; 0 = auto-detect from det name containing `"seal"`. Seal mode: no reading-order sort (ring text is cyclic), rec score threshold 0 |
| `profile` | 0 | 1 = collect per-stage timings for `ppocr_last_profile()` |
| `warmup` | 0 | 1 = run one dummy det/rec/cls inference at create so the first real image skips one-time backend init (~15 ms cutlass select + workspace alloc on CUDA). The CLI enables this unless `--no-warmup` |

**`rec_batch` resolution order** (fixed in PERF5): explicit
`cfg.rec_batch` > per-model `rec_batch_hint` (from the rec model config JSON)
> 16. Bigger rec models run faster at batch 8 on CUDA (the [16,3,48,320]
session resize + activation memory outweigh the GEMM win); the shipped
configs set `rec_batch_hint: 8` for `PP-OCRv6_medium_rec`,
`PP-OCRv6_small_rec`, `PP-OCRv5_server_rec`.

Note: CLI runs always set `offline=1, download=0` (hard error if a model file
is missing — useful for CI); embedders get auto-download via the defaults.

## Model config JSON (`configs/<Name>.json`)

Generated by `tools/convert_models.py`; parsed by `src/config.cpp`
(`load_model_config`). Shape per `docs/CONTRACT.md`:

```json
{
  "name": "PP-OCRv6_tiny_det", "type": "det",
  "file": "PP-OCRv6_tiny_det.mnn", "sha256": "…", "bytes": 1746376,
  "url": "PP-OCRv6_tiny_det.mnn",
  "det": { … }
}
```

Top level: `name`, `type` (`det|rec|cls`), `file`, `sha256`, `bytes`, `url`.

### `det` sub-object

| Field | Example | Meaning |
|---|---|---|
| `thresh` | 0.3 | binarization threshold on the probability map |
| `box_thresh` | 0.6 | min mean box score to keep |
| `unclip_ratio` | 1.5 | polygon expansion factor (Clipper offset) |
| `max_candidates` | 1000 | cap on contours considered |
| `resize.mode` | `limit_min` | `limit_min` \| `resize_long` \| `no_resize` (Paddle `DetResizeForTest` types) |
| `resize.limit_side_len` | 64 | LimitMin: min side cap → stride alignment; ResizeLong: long-side target |
| `resize.resize_long` | 960 | long-side cap for `resize_long` mode |
| `resize.stride` | 32 | alignment stride (32 = v6/seal; 128 = v4/v5) |
| `resize.max_side_limit` | 4000 | hard ceiling on either edge |

`min_size` (drop boxes whose unclipped shortest side is below this, default
3) is also parsed from the det sub-object when present (`src/config.cpp`),
introduced by the M2-ROBUST sweep.

**Runtime values are the PaddleX pipeline defaults, not the per-model
`inference.yml`**: all dets run `limit_min 64 / stride 32` with DB
`0.3/0.6/1.5` at inference time (empirically verified — see
`docs/DET_GEOMETRY.md`, AGENTS.md rule 4); seal dets use
`resize_long 736 / stride 128` with `0.2/0.6/0.5`.

### `rec` sub-object

| Field | Example | Meaning |
|---|---|---|
| `shape` | `[3,48,320]` | input CHW (all current rec variants) |
| `use_space` | true | append a literal space to the alphabet (official models do) |
| `dict` | `["!","\"",…]` | character dictionary (95–18708 entries depending on model) |
| `rec_batch_hint` | 8 | optional per-model default rec batch (see priority order above) |

Alphabet at decode time = `["blank"] + dict + (use_space ? [" "] : [])`,
blank index 0. Normalization: `(x/255 − 0.5)/0.5`, BGR input.

### `cls` sub-object

| Field | Example | Meaning |
|---|---|---|
| `shape` | `[3,80,160]` | input CHW |
| `labels` | `["0_degree","180_degree"]` | topk=1 |
| `mean` / `std` | `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` | ImageNet normalization |

## Environment variables

| Var | Used by | Meaning |
|---|---|---|
| `PPORC_MNN_MODELS` | engine | default `model_dir` |
| `PPORC_MNN_CACHE` | engine | download cache dir |
| `PPORC_MNN_MIRROR` | engine | download base URL |
| `PPOCR_MNN_SESSION_MODE` | MNN session | diagnostic knob: `debug` \| `release` \| `backend_auto` \| `mem_collect` (unset = MNN defaults; no effect on output, kept for A/B experiments) |

## Python tooling

`tools/run_reference.py` drives the CLI over the reference matrix;
`tools/score.py` scores predictions (image-joined CER, PASS ≤ 0.05);
`tools/bench.py` sweeps model × backend × threads. Full flags and workflow:
[`tools/README.md`](../tools/README.md).
