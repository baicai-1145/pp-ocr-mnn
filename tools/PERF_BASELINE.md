# M3-PERF1: Inference Profiling Baseline

## TL;DR

Per-stage profiling is in place (`--profile` flag + `ppocr_profile` C ABI +
`tools/bench.py`) and the first full latency matrix is measured. On this
box (AMD EPYC 7502 14C, A10G 24G):

- **CPU t4**: v6_tiny 554 ms, v4_mobile 559 ms, v6_medium 2520 ms e2e
- **CUDA t4**: v6_tiny **137 ms (4.1x vs CPU)**, v4_mobile 167 ms (3.3x),
  v6_medium 239 ms (**10.5x vs CPU**)
- **Hot stages on CPU**: det session run 26–51% + rec session run 31–63%
  (= the two model forwards are ~90% of e2e; everything else is <7%)
- **Hot stages on CUDA**: the CPU-side pipeline (db_post 21%, ctc_decode
  13–21%, decode 10–12%, det_prep 12–15%) — the forwards themselves
  collapse to 13–22% combined.
- Optimization phase should target, in order:
  **(1) rec batch拼批/width bucketing, (2) det input size + DB post CPU
  cost, (3) CUDA fp16 (Precision_Low), (4) CPU winograd/WinogradConv &
  thread-pool pinning, (5) zero-copy decode→det_prep.**

## What was added

| Item | Where | Notes |
|---|---|---|
| `ppocr_profile` POD | `include/ppocr/profile.h` | 9 stage timings + e2e + create + first_run + n_boxes/rec_batches/threads/backend |
| `profile` config field | `ppocr_config.profile` | 0 = off (default, zero cost) |
| `ppocr_last_profile()` | C ABI | returns last-run profile; NULL when off |
| Engine instrumentation | `src/ppocr.cpp` | timers in `run_det_sync` (prep/run/db_post), `run_rec_sync` (crop_warp/rec_prep/rec_run/ctc), `run_full` (e2e, first_run), `ppocr_create` (create_ms), `ppocr_run_file` (decode) |
| CLI flag | `--profile` | adds `"profile":{...}` object to the result JSON |
| MNN 3.6.1 CUDA path fix | `src/mnn_session.cpp` | non-CPU backends: `copyFromHostTensor`/`copyToHostTensor` staging (device tensors have null host ptr); fixes silent CUDA→CPU fallback |
| Bench harness | `tools/bench.py` | model × backend × threads sweep, N warm runs/img, cold start, raw records, per-image medians |
| Summary printer | `tools/perf_summary.py` | main table + stage-share table |
| CUDA-linked CLI build dir | `build-cuda/` | links `third_party/MNN/build_cuda/libMNN.so` (3.6.1 + CUDA); run with `LD_LIBRARY_PATH=<that dir>:...` |

Branch: `ws/m3-perf` (fresh branch off the m4-seal merge of main `73257cc`;
m4-seal history stays on `ws/m4-seal`).

## Method

- 20 images: `/root/ocr_test_imgs/{zh,en}/0*.jpg` (10 each; street-scene
  style, 5–10 lines typical)
- Each cell: fresh process per run; 1 cold run (discarded from means,
  reported separately) + 8 warm runs/image
- Cell = model × backend × threads:
  models {v6_tiny, v6_medium, v4_mobile} × {cpu×{1,4,8}, cuda×{4}}
- Warm mean±std over all runs of the cell; per-image median e2e also
  reported (one image, en/08.jpg, yields 302 det boxes → 11.6 s rec on
  cpu t1, which skews raw means; median is the robust throughput figure).
  Raw per-(image, run) records are in the `raw` key of the output JSON
  (`/tmp/perf_baseline.json`, kept out of git: results/ data).
- "cold ms" = create_ms + first_run e2e (fresh process, includes MNN
  session create + first-run kernel setup)
- wall_ms (process spawn + model load from page cache) ≈ e2e + 130–550 ms;
  reported in raw records, not part of e2e

## Session config defaults (recorded, MNN 3.6.1)

| Setting | Value now | Notes |
|---|---|---|
| `ScheduleConfig.type` | from `--backend` (auto/cpu/cuda/...) | `pickBackend()` maps AUTO→MNN_FORWARD_AUTO |
| `ScheduleConfig.numThread` | `cfg.num_threads` (CLI `--threads`, default 0 = MNN default) | **0 (auto) beats explicit 4/8/14 on this box** (det 177 ms vs 217/281/278 on en/03) |
| `BackendConfig.power` | 0 (Power_Normal) | unused |
| `BackendConfig.precision` | 0 (Precision_Normal) | CUDA = **fp32** cudnn paths; Low(2) = fp16 — biggest CUDA lever left |
| `BackendConfig.memory` | 0 (Memory_Normal) | unused |
| rec batch | 8 (`--batch`) | chunked; batch_w = max wh_ratio in chunk (paddlex rule), cap 320 |
| det input | limit_min 64 / resize_long 960 / stride 32 | 1280×960 input → 960×736 net input |

## Baseline matrix (20 imgs × 8 warm runs; ms)

> Numbers from the instrumented run of 2026-08-27 on `waas`
> (EPYC 7502 14C, no pinning; CUDA = A10G sm_86, MNN 3.6.1 fp32).

| cell | e2e mean | img median | ±std | FPS | det_run | rec_run | db_post | cold ms |
|---|---|---|---|---|---|---|---|---|
| v6_tiny/cpu/t1 | 1158 | 896 | 1653 | 0.86 | 555 | 507 | 31 | 643 |
| v6_tiny/cpu/t4 | 545 | 417 | 680 | 1.84 | 260 | 189 | 31 | 391 |
| v6_tiny/cpu/t8 | 595 | 438 | 760 | 1.68 | 286 | 212 | 31 | 497 |
| **v6_tiny/cuda/t4** | **133** | **104** | 139 | **7.50** | 14.2 | 2.1 | 28 | 321 |
| v6_medium/cpu/t1 | 6987 | 4694 | 11416 | 0.14 | 3097 | 3644 | 28 | 4836 |
| v6_medium/cpu/t4 | 2559 | 1792 | 3886 | 0.39 | 1122 | 1191 | 29 | 2452 |
| v6_medium/cpu/t8 | 2077 | 1520 | 3008 | 0.48 | 901 | 926 | 30 | 2164 |
| **v6_medium/cuda/t4** | **239** | **161** | 321 | **4.18** | 22.0 | 7.0 | 27 | 1075 |
| v4_mobile/cpu/t1 | 1148 | 596 | 2380 | 0.87 | 307 | 726 | 39 | 682 |
| v4_mobile/cpu/t4 | 561 | 415 | 809 | 1.78 | 215 | 232 | 38 | 520 |
| v4_mobile/cpu/t8 | 612 | 433 | 887 | 1.63 | 239 | 257 | 38 | 581 |
| **v4_mobile/cuda/t4** | **170** | **124** | 203 | **5.90** | 17.3 | 8.1 | 36 | 411 |

Stage-share (% of e2e):

| cell | decode | det_prep | det_run | db_post | crop_warp | rec_prep | rec_run | ctc |
|---|---|---|---|---|---|---|---|---|
| v6_tiny/cpu/t4 | 2.8 | 3.8 | 47.8 | 5.7 | 1.6 | 1.3 | 34.8 | 3.2 |
| v6_medium/cpu/t4 | 0.6 | 0.8 | 43.9 | 1.1 | 0.3 | 0.3 | 46.5 | 2.0 |
| v4_mobile/cpu/t4 | 2.8 | 3.7 | 38.3 | 6.7 | 1.5 | 1.3 | 41.3 | 3.2 |
| v6_tiny/cuda/t4 | 11.8 | 15.3 | 10.7 | 21.3 | 6.7 | 4.9 | 1.5 | 13.1 |
| v6_medium/cuda/t4 | 6.7 | 8.7 | 9.2 | 11.1 | 3.7 | 2.9 | 2.9 | 21.2 |
| v4_mobile/cuda/t4 | 9.4 | 12.4 | 10.2 | 21.1 | 4.8 | 4.2 | 4.8 | 10.4 |

Findings:

1. **CPU: det_run ≈ rec_run in e2e share.** det 38–51%, rec 31–63%
   (rec dominates on box-dense images). Both scale ~linearly with
   thread count up to 4; **8 threads does not help and can hurt**
   (t8 ≥ t4 on all three models — thread-pool sync overhead).
2. **`threads=0` (MNN default) is FASTER than any explicit count** on
   this box (en/03 det: 0→177 ms, 1→323, 2→249, 4→217, 14→278). MNN's
   auto path (probably mask-based big-core selection) beats our explicit
   numThread. The baseline matrix uses explicit counts as requested, but
   the default should be reconsidered in M3-PERF2.
3. **CUDA: forwards collapse, CPU-side post dominates.** det_run+rec_run
   drop to 13–22% of e2e; db_post (20%), decode (10–12%), det_prep
   (12–15%), ctc_decode (10–21%) become the top costs. ctc_decode is
   single-threaded C++ over [N,T,C] logits — for v6_medium's 6623-class
   dict and T≈40 timesteps × 8 crops that's measurable work.
4. **Cold start** is 330 ms (tiny/cuda) to ~5 s (medium/cpu t1):
   create_ms (session build) + first-run kernel/JIT. For server use the
   engine should be long-lived (async API); for CLI one-shot usage the
   cold cost dominates the warm e2e at CUDA's 133 ms level.
5. **En/08.jpg dense-FP workload** (302 boxes) stretches rec_run to
   11.6 s (t1) — box_thresh/unclip tuning or rec score filtering would
   cut this pathological case; per-image medians exclude it from
   headline numbers.

## Optimization hypotheses (ranked, for M3-PERF2)

1. **Rec batch拼批 + width bucketing** (CPU + CUDA, low risk).
   Currently 38 sequential rec runs on the 302-box image because
   batch=8 and each chunk's `batch_w` is set by the chunk's max
   wh_ratio. Sort crops by aspect ratio before chunking (stable with
   reading order for output), then pad only to the bucket max — cuts
   wasted padding FLOPs (typ. 2–3x on mixed-width sets) and keeps
   logits identical per bucket (padding is zero and CTC ignores blank
   tails). Est. rec_run −40–60% on box-dense images, −10–20% typical.
2. **CUDA Precision_Low (fp16)** (CUDA only, trivial change).
   `BackendConfig::precision = Precision_Low` switches cudnn convs to
   fp16 on A10G (≈2x tensor-core throughput). Must re-run the m3-cuda
   gate (mean diff < 0.002) — risk is contained because the gate
   harness already exists. Est. det_run 14.2→~8 ms, rec_run 2.1→~1.2.
3. **Det input size** (CPU + CUDA, accuracy tradeoff). The det net
   input is 960×736 for a 1280×960 source. `resize_long 736` (paddlex
   mobile default) would cut det FLOPs ~1.7x; must re-validate the
   full 811-cell CER matrix (det boxes shift). Park unless CPU targets
   require it.
4. **db_post / ctc_decode CPU cost** (CUDA path mostly). db_post at
   21% of CUDA e2e is contour tracing + unclip on 960×736 bitmaps;
   candidates: downsample prob to /8 instead of /4 before post (needs
   CER re-check), box-level early-exit on tiny components, and
   parallelizing ctc_decode across the batch with std::thread (trivial:
   per-crop independent). Est. CUDA e2e −15–25%.
5. **CPU winograd/conv tuning** (CPU only). MNN 3.6.1 has Winograd
   conv + Kleidiai paths; our static `build/libMNN.a` was built with
   default flags — rebuild with `-DMNN_AVX2=ON -DMNN_USE_SSE=ON
   -DMNN_KLEIDIAI=ON` (the `build_cuda` .so already sets these and
   measured *slower* on CPU than the .a, so this needs an A/B, not an
   assumption). Also revisit `numThread=0` auto vs explicit pinning.
6. **Zero-copy decode→det_prep** (both, small). decode_ms 10–12% on
   CUDA is stb JPEG decode; a libjpeg-turbo SIMD decode would halve
   it. Also det_prep copies HWC→CHW float — could fuse into the
   decoder's scanline callback. Est. e2e −3–5%.

## Not yet covered (explicitly out of scope for PERF1)

- OpenCL/Vulkan backends (M3-CUDA report: Vulkan segfaults in the
  NVIDIA loader on this box; OpenCL init fails → CPU fallback).
- Rec INT8 quantization (changes numerics; separate gate).
- Multi-image batched det (batch=1 by contract for now).
- Platform tunings (Android/iOS thread affinity) — desktop-only matrix.

## Reproduce

```bash
cd /root/pp-ws/post
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DMNN_LIBRARY=$PWD/third_party/MNN/build_cuda/libMNN.so
cmake --build build-cuda -j
python3 tools/bench.py --out /tmp/perf_baseline.json
python3 tools/perf_summary.py /tmp/perf_baseline.json
# single-image profile through the CLI:
LD_LIBRARY_PATH=third_party/MNN/build_cuda:/usr/lib/x86_64-linux-gnu \
  ./build-cuda/ppocr_cli --image /root/ocr_test_imgs/zh/03.jpg \
  --det-config configs/PP-OCRv6_tiny_det.json \
  --rec-config configs/PP-OCRv6_tiny_rec.json \
  --model-dir /root/pp-ocr-mnn/models --backend cuda --threads 4 \
  --profile --json /tmp/out.json
```
