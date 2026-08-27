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

## Round 1 (M3-PERF2) — bit-exact CPU/CUDA speedups + A/B'd-out items

Working rule for this round: **every kept change is bit-exact** (char+poly
identical on zh/03, zh/04, en/03, en/08 — the last being the 302-box dense
image) or explicitly reverted. Timing A/B under identical host load
(interleaved runs; the box runs co-tenant workloads at load ~39, so
sequential A/Bs mis-attribute load spikes).

### Kept (commits 9377079..1fdd434 on ws/m3-perf)

| change | effect |
|---|---|
| 2a. resize/normalize hoisting (preprocess.cpp): per-column fx/x0/dx + per-row fy/dy out of the bilinear inner loop; 256-entry LUT for (v*scale-mean)/std — same double ops, same order | det_prep 8.3→5.7 ms (zh/03); det_prep share 3.8→~2% cpu t4, 15.3→~5% cuda |
| 2b. parallel ctc_decode across the rec batch (min(n, hw, 8) threads; rows independent) | en/08 CUDA ctc 236→110 ms; typical ctc share 13→~5% |
| 2c. db_post O(n²) eliminations: per-pixel heap vector → 8-slot stack array; O(labels) std::find per pixel → O(1) root_to_idx; per-component full-map find_first_boundary → one row-major sweep | en/08 db_post 207→27 ms (-87%); zh/03 21.6→~14 ms |
| 2d. parallel crop warp + rec prep (per-box/per-crop independent work over ≤8 threads) | en/08 CUDA crop_warp 27→10 ms, e2e 441→402 ms |

### A/B'd out (reverted, with evidence)

| item | verdict | evidence |
|---|---|---|
| rec width bucketing {128,224,320} (hyp. 1) | **REVERT — output changes** | en/08: 70/302 lines differ; en/03 'S'→garbage — the exact M2-NUM narrow-width regression. paddlex pads to 320 for accuracy, not just batching. conv border effects at shorter widths shift logits. |
| CUDA Precision_Low fp16 (hyp. 2) | **REVERT — slower AND wrong** | det 21.8 vs 16.9 ms; text 'QUEEN VICTORIA ST'→'QUE!ICOIAS' on zh/03. Precision_High correct but slower (27.0 ms). (Also: ScheduleConfig::backendConfig is a borrowed pointer — stack-allocating it segfaults CUDARuntimeCreator::onCreate.) fp16 needs a cudnn-enabled MNN build; this .so links cublas only. |
| db_post 2×2 max-pool before post (hyp. 4) | **REVERT — no win** | Output bit-identical (0 px drift, en/08+zh/03) but db_post only 215→204 ms on the dense case, ~0 typical: cost is in CCL/unclip, not the bitmap scan. Superseded by 2c. |
| threads=0 (auto) CPU default (item 3) | **NO CHANGE — same speed** | Full-sweep A/B: v6_tiny 537.5 vs t4 544.8; v4_mobile 563.8 vs 560.6; v6_medium 2492.8 vs 2558.8 ms (±1-3%, within noise). The single-image "177 vs 217 ms" that motivated it was turbo-clock noise. MNN treats 0 as 1 thread in CPURuntime but Schedule's AUTO path gives equivalent throughput to t4 on this EPYC. |
| libjpeg-turbo decode (stretch) | **REVERT — not char-exact** | decode 18→7.4 ms (real win) but IDCT rounding differs from stb: zh/03 2 low-conf FP lines shift ('医方形'→'组为E', score 0.55/0.32); en/08 240/302 lines differ (dense tiny-text FPs flip). Real text lines identical everywhere, but the round's bit-exact bar fails. Revisit if a future round accepts "CER-equal on real text". |

### Round-1 matrix (same protocol as the baseline; filled from perf_round1_final.json)

| cell | baseline e2e | round-1 e2e | Δ | baseline FPS | round-1 FPS |
|---|---|---|---|---|---|
| v6_tiny/cpu/t1 | 1158.1 | 1134.0 | -2.1% | 0.86 | 0.88 |
| v6_tiny/cpu/t4 | 544.8 | 541.3 | -0.6% | 1.84 | 1.85 |
| v6_tiny/cpu/t8 | 594.5 | 516.8 | -13.1% | 1.68 | 1.94 |
| **v6_tiny/cuda/t4** | 133.4 | **105.9** | **-20.6%** | 7.50 | **9.45** |
| v6_medium/cpu/t1 | 6986.6 | 7008.6 | +0.3% | 0.14 | 0.14 |
| v6_medium/cpu/t4 | 2558.8 | 2469.2 | -3.5% | 0.39 | 0.40 |
| v6_medium/cpu/t8 | 2076.6 | 1878.9 | -9.5% | 0.48 | 0.53 |
| **v6_medium/cuda/t4** | 239.4 | **184.4** | **-23.0%** | 4.18 | **5.42** |
| v4_mobile/cpu/t1 | 1148.0 | 1111.8 | -3.2% | 0.87 | 0.90 |
| v4_mobile/cpu/t4 | 560.6 | 517.5 | -7.7% | 1.78 | 1.93 |
| v4_mobile/cpu/t8 | 612.0 | 532.8 | -12.9% | 1.63 | 1.88 |
| **v4_mobile/cuda/t4** | 169.5 | **126.3** | **-25.5%** | 5.90 | **7.92** |

db_post column: 27-38 → 11.6-14.8 ms across all cells (the 2c fix).
per-image medians: v6_tiny/cuda 104 → 85 ms; v4_mobile/cuda 124 → 94 ms.

### Headline vs targets

- Target "CUDA tiny 7.5→15+ FPS": NOT met this round — see | cell | baseline e2e | round-1 e2e | Δ | baseline FPS | round-1 FPS |
|---|---|---|---|---|---|
| v6_tiny/cpu/t1 | 1158.1 | 1134.0 | -2.1% | 0.86 | 0.88 |
| v6_tiny/cpu/t4 | 544.8 | 541.3 | -0.6% | 1.84 | 1.85 |
| v6_tiny/cpu/t8 | 594.5 | 516.8 | -13.1% | 1.68 | 1.94 |
| **v6_tiny/cuda/t4** | 133.4 | **105.9** | **-20.6%** | 7.50 | **9.45** |
| v6_medium/cpu/t1 | 6986.6 | 7008.6 | +0.3% | 0.14 | 0.14 |
| v6_medium/cpu/t4 | 2558.8 | 2469.2 | -3.5% | 0.39 | 0.40 |
| v6_medium/cpu/t8 | 2076.6 | 1878.9 | -9.5% | 0.48 | 0.53 |
| **v6_medium/cuda/t4** | 239.4 | **184.4** | **-23.0%** | 4.18 | **5.42** |
| v4_mobile/cpu/t1 | 1148.0 | 1111.8 | -3.2% | 0.87 | 0.90 |
| v4_mobile/cpu/t4 | 560.6 | 517.5 | -7.7% | 1.78 | 1.93 |
| v4_mobile/cpu/t8 | 612.0 | 532.8 | -12.9% | 1.63 | 1.88 |
| **v4_mobile/cuda/t4** | 169.5 | **126.3** | **-25.5%** | 5.90 | **7.92** |

db_post column: 27-38 → 11.6-14.8 ms across all cells (the 2c fix).
per-image medians: v6_tiny/cuda 104 → 85 ms; v4_mobile/cuda 124 → 94 ms..
  What's left after round 1 on the CUDA path: det_run (fp32 cutlass convs;
  fp16 broken per above), decode (stb scalar; turbo rejected on exactness),
  and pipeline overlap (CPU post of image N ∥ GPU run of N+1 — a
  structural change for PERF3). 15 FPS needs ~67 ms; current typical-image
  CUDA tiny e2e ≈ 80-90 ms single-threaded pipeline.
- Target "CPU t4 545→450 ms": NOT met — 541.3 ms (-0.6%, noise). The CPU t4 path is ~85%
network forward (det_run 272 + rec_run 200 = 472 of 541 ms) which
bit-exact host-side work cannot touch; the post fixes did help t8
(-13.1%, where the host work was a larger share). Next CPU lever is
conv-kernel level (MNN build flags A/B — hyp. 5, needs a variant
rebuild + numerics gate) or det input size (hyp. 3, CER-gated)..

## Round 2 (M3-PERF3)

Scope: (1) MNN CUDA rebuild A/B, (2) batch-mode CPU/GPU overlap. All A/Bs
interleaved or batched under identical host load. Binaries:
`build-cuda` (baseline .so) vs `build-cuda-tune` (variant .so).

### 1. MNN CUDA rebuild A/B — REJECTED (baseline .so kept)

Survey of MNN 3.6.1 CUDA backend (no source changes; separate build dir):

- **cuDNN: does not exist in MNN.** `grep -ri cudnn third_party/MNN` over
  CMakeLists + `source/backend/cuda/` returns nothing — MNN's CUDA backend
  is cutlass-only (implicit-GEMM > Winograd > cutlass fallback selection
  order, `ConvSingleInputExecution.cu`). There is no `MNN_USE_CUDNN` knob.
- The only runtime-relevant CUDA build knobs: `MNN_CUDA_TUNE_PARAM`
  (cutlass autotune), `MNN_CUDA_BF16` (precision-3 path, text-accuracy
  risk), `MNN_CUDA_QUANT` (int8 path, out of scope). Variant built with
  `-DMNN_CUDA_TUNE_PARAM=ON` (build_cuda_tune/, sm_86, same flags else).

Evidence (interleaved single-image runs, zh/03, 3 fresh processes each):

| variant | det_run | e2e |
|---|---|---|
| baseline .so | 16.7-17.0 | 73-89 |
| tune .so | 18.2-18.7 | 716-724 |

The tune variant pays a **~300 ms per-new-input-shape re-tune** inside the
first runSession after each resize (cutlass param sweep per problem size,
`CUDARuntime::mTunedBlockWarpShape`). In batch mode (one process, 10
distinct shapes) tune mean e2e 353 ms vs base 70 ms; on a same-shape dir
(6x en/08) the first image tunes (1257 ms) and warm images match baseline
(det_run 1.4 vs 1.3, e2e 367 vs 385 — noise-level). Outputs were
char-identical between .so variants; the variant is rejected purely on
speed. (Persisting the tune cache via `Interpreter::setCacheFile` was
considered; since warm-state speed equals baseline, there is no upside to
justify shipping a second .so.)

Byproduct finding (important for interpreting ALL prior CUDA numbers): in
a single process, baseline det_run drops from ~17 ms (first run per shape)
to **1.1-1.3 ms** warm. The per-image e2e in single-image CLI runs was
dominated by one-time cutlass selection/workspace alloc, not conv time —
batch-mode amortizes it.

### 2. Batch mode CPU/GPU overlap — KEPT (`--batch-dir`, engine-per-worker)

`ppocr_cli --batch-dir DIR [--workers N]` (apps/ppocr_cli.cpp): scans DIR
for images, spawns N workers, each with its own `ppocr_engine` (C ABI is
multi-instance thread-safe), pulling images off an atomic index. Each
image runs the full serial pipeline unchanged — CPU post of image N-1
overlaps GPU runs of image N across workers. Output JSON is an array of
per-image results (identical schema per image) + a `_bench` object
(n_images, workers, wall_s, throughput_fps, mean/max e2e_ms, failures).

Correctness: workers=1 vs workers=3 outputs **identical** on zh dir;
batch(w1) vs serial single-image CLI outputs **identical** on zh (10
imgs) and en (10 imgs, incl. dense en/08 283 lines).

Throughput (v6_tiny/cuda/t4, 3 reps each, zh dir = 10 mixed images;
dense6 = 6x en/08 duplicates):

| dir | w1 | w2 | w3 | w4 |
|---|---|---|---|---|
| zh fps | 10.4 | 13.4 | 14.0 | 15.9 |
| dense6 fps | 2.31 | 3.04 | 3.71 | 4.03 |
| en fps | 7.72 | — | — | 11.47 |

vs the ≥25% bar: zh +29% @w2 / +53% @w4; dense +32% @w2 / +74% @w4; en
+49% @w4. Target met. CPU backend also improves: t4 w1 4.13 → w4 5.54
FPS (+34%) on zh.

The 15 FPS single-image target from round 1 is now effectively met at the
batch level: zh dir @w4 = 15.9 FPS CUDA tiny (vs 9.45 single-image);
warm det_run 1.2 ms means the batch pipeline is postprocess-bound, which
is exactly what workers parallelize.

## Round 3 (M3-PERF4)

Scope: post-batch bottleneck (CPU post), rec GEMM batching, first-image
warmup, warp sampler A/B. All approximations gated on line-exactness
(en/08 dense 283 + zh/04 + ja/01 60 + zh/03 + en/03).

### Kept

| change | effect | exactness |
|---|---|---|
| 4a bbox polygon rasterization (box_score_fast) | db_post 26.2→19.9ms dense (-24%) | bit-exact (text+poly+score) |
| 4b default rec_batch 8→16 | rec_run 21.9→11.3ms, rec_prep 43→26ms, dense warm e2e 320→255ms | line-identical (batch_w is always 320: floored at 6.67 ratio and capped) |
| 4c create-time warmup (cfg.warmup, CLI default on) | first-image det_run 13.2→1.5ms, e2e 104→81ms (zh/04); create +46ms once | output-neutral (CUDA+CPU) |

Combined interleaved batch A/B (host load ~30-40 during measurement;
deltas valid, absolutes depressed): dense6 w4 1.85→2.75 FPS (+49%),
zh dir w4 9.33→10.02 FPS (+7%, decode+serial dominated).

Key mechanism discovered (4b): rec batch_w is floored (320/48) and
capped (320) for every chunk, so chunk size is width-neutral — only the
session batch dim N changes. The M2-NUM narrow-width regression cannot
trigger via chunk size. batch=32 regresses (bigger per-run session
resize outweighs GEMM efficiency).

### A/B'd out (reverted with evidence)

| hypothesis | evidence | verdict |
|---|---|---|
| CCL row-skip + fused pixel count (P1+P2) | db_post 26.6 vs 27.3ms — noise | no win; CCL is branch-predictable already |
| warp bicubic→bilinear (PPOCR_WARP_BILINEAR) | text-identical on all 5 gates(!) but crop_warp 31.1→30.4ms dense (noise); rec inputs not bit-exact | rejected: no speed win for a semantic change |
| clipper/unclip batching | unclip measured 1.4-1.5ms of 20ms db_post (stage-timing probe) | not a hotspot; skipped |

db_post internal composition (en/08, 384 comps, stage probe): ccl 5.8,
flatten 3.2, score-fill 6.1 (→ 1.x after 4a), trace 0.55, unclip 1.4,
binarize 0.9, seeds 1.0, glue ~2.7ms.

Note on decode timing: all numbers above use stb_image decode. The m1
libjpeg-turbo decoder (ws/m2-final-diag, pending merge) will shift
decode_ms down ~10ms/image; batch throughput will improve further, and
that change is CER-gated separately by m1.

## Round 4 (M3-PERF5)

### 1. libjpeg-turbo decode (cherry-picked m1's m2-decode-align, 4569d92)

- **Decode-level parity: PASS** — all 272 corpus jpgs bit-exact vs
  cv2.imread (FNV-1a over BGR bytes; decode parity harness rebuilt on
  this branch, matches m1's result).
- decode_ms: zh/04 13.4→4.6, en/08 19.4→7.4 (interleaved 3-rep medians).
- **Output surface changes** (as expected — decode bytes change det
  inputs): en/08 171/283 lines shift; 5-gate line-exactness vs the stb
  era FAILS by construction.
- **CER vs paddle baselines (v6_tiny, zh/en/ja × 10 imgs, score.py
  semantics, same driver/backend)**: zh 0.0615→0.0573 (better), ja
  0.2893→0.2617 (better), en 0.0355→0.1240 (worse). The en regression
  is ONE image (en/03): the 1-px det box shift (700→701, 717→718) flips
  the single low-margin char 'S'→'AS' — CER 0.000→1.000 on a 1-char
  reference. Decode parity does NOT buy box parity (MNN-vs-paddle det
  kernel diffs still shift boxes). Note all cells remain above the
  0.05 gate in BOTH decoders (M2 milestone still open); libjpeg moves
  zh/ja toward baseline.
- Verdict: **kept on the branch as the m1-merge preview** — it is the
  decode the baselines were generated with; final call belongs to m1's
  merge review (the CER effect is mixed but decode-parity is principled).

### 2. zh per-stage analysis (warm batch profile)

e2e 59ms mean: decode 20.1%, det_prep 18.4%, db_post 14.3%, crop 11.1%,
ctc 6.1%, rec_run 1.9%, det_run 4.8% (warm). decode >15% confirmed the
libjpeg lever. Remaining serial list (post-warm): det_prep (~5.7ms calm,
float-bilinear resize, PERF2-optimized, reference-matching — not worth
touching), glue (~40ms e2e minus stages: build_lines, JSON, Image
copies). **Sparse-image rec batching: already adaptive** — chunk loop
runs the last chunk at N=actual (2 boxes → 1 batch of N=2, rec_run
0.6ms); no padding waste exists, nothing to do.

### 3. medium/server tier

- **The "medium 5.42 FPS" premise was stale**: with PERF4 warmup,
  medium's det_run is 1.4ms warm (the old 25ms was first-run cost).
  v6_medium single-image e2e 79ms (12.7 FPS) on zh/03.
- **MNN session modes: no effect** (Debug/Release/Backend_Auto/
  Memory_Collect via PPOCR_MNN_SESSION_MODE, medium warm det/rec
  1.4/1.1ms in all modes).
- **rec batch optimum is per-model** (dense6 CUDA w1, 2+ reps):
  v6_medium 8 (1.08 vs 0.87 fps @16), v6_small 8 (1.54/1.21 vs
  0.94/0.97), v5_server 8 (0.99/0.93 vs 0.76/0.85), v4_server tie
  (1.17/1.16), v6_tiny + v4_mobile 16 (PERF4). Not a size story (46MB
  v4_server ties; 20MB v6_small prefers 8) — per-graph. Implemented as
  **RecConfig.rec_batch_hint** (config-driven per-model default;
  resolution cfg->rec_batch > hint > 16; moved resolution after config
  load so the hint is visible). Hints set: v6_medium/small, v5_server
  rec = 8. Outputs line-identical (width-neutral chunking). Medium
  dense default 0.93 vs forced-16 0.79 fps under same load (+18%).
  convert_models.py should regenerate shipped configs with the hints.

Post-round state (load ~29, w4 batch): zh 15.3 FPS, dense6 4.49 FPS
(v6_tiny, libjpeg + batch16). test_post 20/20, ctest 2/2.
