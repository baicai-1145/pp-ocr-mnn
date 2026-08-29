# PERF-NEXT: Hardware-saturation roadmap (2026-08-29)

Measured on A10G + EPYC 7502, v6-tiny, zh test set, warm state,
per-stage profile via ppocr_last_profile() (profile=1).

## Current breakdown — v6-tiny CUDA single-image E2E 126 ms

| Stage | ms | % E2E | Note |
|---|---|---|---|
| GPU det_run | 1.2 | 1% | warm; MNN 3.6.1 + int-overflow fix |
| GPU rec_run | 1.0 | 1% | |
| db_post | 67.8 | 54% | CPU serial; breakdown below |
| crop+warp | 11.1 | 9% | per-box perspective warp |
| det_prep | 6.9 | 5.5% | MNN ImageProcess resize/pad |
| ctc_decode | 8.2 | 6.5% | scalar greedy + dict lookup |
| jpeg_dec | 4.5 | 3.6% | stb_image scalar |
| rec_prep+misc | 4.4 | 3.5% | |

GPU util under 3-process concurrent batch load: 7%, 80 W / 300 W.
GPU network compute is 2% of E2E — the GPU is idle waiting for CPU.

## db_post internal breakdown (67.8 ms, perf-instrumented)

| Sub-step | ms | % of db_post | Cause |
|---|---|---|---|
| hole precompute + trace | 42.1 | 62% | BFS flood-fill over every bg
region; stores ALL pixels of each hole in vector<pair<int,int>> just to
extract its boundary; massive small-heap churn |
| CCL (2-pass union-find) | 20.4 | 30% | per-foreground-pixel
std::find over root list = O(N x n_comp) |
| binarize | 0.9 | 1.3% | |
| minibox x2 / fillmask / clipper | 1.1 | 1.6% | not a bottleneck |

Root cause: the faithful port splits cv2.findContours(RETR_LIST)
semantics into 3 full-image passes + heavy allocation. OpenCV's
Suzuki-Abe does external+hole borders in ONE raster scan, O(N).

## Optimization plan (ranked by ROI)

1. **Suzuki-Abe single-pass border tracking** replacing CCL +
   hole-precompute inside db_post. Target 67.8 -> 8-15 ms.
   Gate: bit-exact vs current implementation on all 60 cells' det
   outputs (existing test harness).
   Expected: E2E 126 -> ~60 ms single-thread.
2. **Pipeline overlap**: GPU runs image N+1 while CPU post-processes
   image N (double buffer + stream). Hides remaining CPU serial under
   GPU time. Expected: 4-worker throughput 12.8 -> 70+ FPS.
3. **ctc_decode vectorize + turbo-jpeg** (performance mode; bit-exact
   gate currently rejects turbo — needs a perf-mode opt-in).
4. Non-goals: cuDNN (does not exist in MNN), cutlass autotune (slower,
   PERF3 A/B), int8 det (out of scope).

Historical: PERF1-3 rounds already banked -25.5% CUDA e2e (post
fixes) and batch-dir overlap (+29-74%); see PERF_BASELINE.md.
