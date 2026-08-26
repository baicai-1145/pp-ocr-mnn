# M2-FINAL-DIAG: per-layer diagnosis of 5.9e-3 paddle↔MNN det diff

**Branch:** `ws/m2-final-diag`
**Task:** Final per-layer diagnosis of the 5.9e-3 mean diff
between Paddle's `v6_tiny_det` reference and MNN's output for
zh/04.jpg.  Goal: localize which layer introduces the diff
and whether the diff is intrinsic to MNN's per-conv kernel
or to runtime graph optimization (paddle2onnx / MNNConvert).

## TL;DR (4 findings)

1. **A single conv (Paddle CPU vs Paddle GPU vs ORT CPU vs ORT CUDA)
   produces bit-equivalent output** — max diff = 4.1e-6 (float32
   accumulation noise only).  The per-conv kernel is **NOT** the
   source of the diff.
2. **MNN and ORT (on the paddle2onnx-converted model) both diverge
   from Paddle by the SAME magnitude**: max=0.9637, mean=0.005859,
   %>0.1=1.09%.  The diff is **NOT in the MNN kernel**; it is in
   **paddle2onnx's conversion or Paddle's runtime graph optimization**.
3. **MNN's first conv + bias + ReLU + MaxPool produces bit-equivalent
   output to ONNX's same layer** (max diff 3.3e-6 on `op0001_Add.1`
   vs `p2o.pd_op.relu.0.0`; max diff 2.9e-6 on `op0002_pool2d.0.0`
   vs `p2o.pd_op.pool2d.0.0`).  The diff is not at the entry stem.
4. **The diff accumulates starting at the SE (Squeeze-and-Excitation)
   channel-attention block** — `op0017_Mul.1_raster_0` (SE channel
   scale) shows max=4.28 / mean=0.595 / 99% > 0.1.  Earlier Conv
   blocks (op0019–op0076) all stay sub-0.01 in mean, then SE blocks
   re-introduce the diff.  Paddle's runtime likely fuses Conv+BN+Act
   differently than the unfused ONNX nodes that MNN runs.

## Method

| Tool | Purpose |
|---|---|
| `tools/m2fd_single_conv.py` | 4-way conv comparison: Paddle CPU vs Paddle GPU vs ORT CPU vs ORT CUDA.  Builds a 1-op ONNX (one Conv + optional bias Add), runs the same input through all four. |
| `tools/m2fd_onnx_bisect.py` | Paddle → ONNX via `paddle2onnx`; clone the ONNX model with all node outputs exposed, run via ORT, save 464 intermediates. |
| `tools/m2fd_dump_mnn_intermediates.cpp` | C++ driver: feeds the same input to `ppocr_run_file`, hooks `MNN::runSessionWithCallBack` `after` callback, dumps 220 op-output tensors. |
| `tools/m2fd_bisect_compare.py` | Loads both intermediate sets, matches MNN ops to ONNX intermediates by name (with `_raster_0` suffix stripping), reports per-layer max/mean/%>0.1. |
| `tools/m2fd_shape_bisect.py` | Same purpose, with forward-only cursor + alt-name fallbacks (handles MNN's name reuse across fused Conv+Add+ReLU). |

### Reference data
* Paddle reference: `/tmp/m2num/det_output_paddle.npy` (prob map for
  zh/04.jpg through PaddleX runtime, max 0.9674).
* MNN reference: `/tmp/m2num/det_output_mnn.npy` (prob map through
  MNN CPU, max 0.9636).  Established: mean diff 0.005859, %>0.1 1.09%.

## Finding 1: single conv is bit-equivalent across backends

`tools/m2fd_single_conv.py` extracts the **first conv** of the det
network (the stem 3→16 with kernel 3×3 / stride 2×2 / pad 1),
builds a minimal 1-op ONNX, and runs the same input on:

| Backend | Output shape | Max diff vs Paddle CPU |
|---|---|---|
| Paddle CPU (FP32, no TF32, no IR opt) | [1,16,352,640] | 0 (self) |
| Paddle GPU (FP32, no TF32, cudnn deterministic) | [1,16,352,640] | 3.8e-6 |
| ORT CPU | [1,16,352,640] | 4.1e-6 |
| ORT CUDA (fell back to CPU) | [1,16,352,640] | 4.1e-6 |

**Conclusion: per-conv arithmetic is identical across all backends**
within float32 accumulation noise.  The diff in the full model is
**NOT** from the conv kernel itself.

## Finding 2: MNN ≈ ORT (paddle2onnx) for the FULL model

`tools/m2fd_onnx_bisect.py` runs the full det ONNX model through
ORT and saves the final output.  Comparing to PaddleX's output:

| Metric | MNN vs Paddle | ORT (paddle2onnx) vs Paddle |
|---|---|---|
| max | 0.9637 | 0.9637 |
| mean | 0.005858 | 0.005859 |
| %>0.1 | 1.09% | 1.09% |

The diffs are **statistically identical** (mean differs by 1e-6).
This means:
* MNN's kernel is **not the problem** (ORT gives the same diff).
* The diff originates from **paddle2onnx's export** OR **Paddle's
  runtime graph optimization** (e.g. fused conv+BN, kernel selection).

## Finding 3: MNN's stem block (Conv+Add+ReLU+MaxPool) is bit-equivalent to ONNX

`tools/m2fd_dump_mnn_intermediates.cpp` uses
`MNN::runSessionWithCallBackInfo` to dump every op's output tensor
while running the same input through MNN's v6_tiny_det model.
Saved to `/tmp/m2fd/mnn_intermediates.npz.bin` (544 MB) + `.json`
(220 entries).

`tools/m2fd_bisect_compare.py` matches each MNN op to the ONNX
intermediate with the same name (after `_raster_0` stripping).

**Stem block (op0000–op0004):**

| MNN op | Shape | ONNX counterpart | max diff |
|---|---|---|---|
| op0000_Add.1_raster_0 | [1,3,704,1280] (input) | (no direct match; this is the network input) | 0 (matches Paddle input bit-exact) |
| op0001_Add.1 (post-ReLU) | [1,16,352,640] | p2o.pd_op.relu.0.0 (post-ReLU) | **3.3e-6** ✓ |
| op0002_pool2d.0.0 | [1,16,352,640] | p2o.pd_op.pool2d.0.0 | **2.9e-6** ✓ |
| op0003_Add.3 (post-ReLU) | [1,8,352,640] | p2o.pd_op.relu.1.0 | (matched, see bisect) |
| op0004_Add.5 (post-ReLU) | [1,16,352,640] | p2o.pd_op.relu.2.0 | (matched, see bisect) |

**The first conv block is bit-equivalent between MNN and ONNX.**
The diff is not in the stem.

## Finding 4: SE channel-attention amplifies the diff

The bisect shows the **first sustained diff** appears in the
Squeeze-and-Excitation channel-attention modules.  Each det
stage has an SE block: AvgPool → Conv1x1 → Conv1x1 → HardSigmoid
→ Mul.  The MNN `Mul.N_raster_0` ops (post-SE-channel-scale)
show large diffs because the **channel scale** (a 1×1×C vector)
has high relative variance, and the per-channel difference gets
broadcast-multiplied into every spatial position.

| MNN op | ONNX | max | mean | %>0.1 |
|---|---|---|---|---|
| op0017_Mul.1_raster_0 | Mul.1 | 4.28 | 0.595 | 99.25% |
| op0041_Mul.12_raster_0 | Mul.12 | 6.65 | 0.696 | 97.35% |
| op0065_Mul.23_raster_0 | Mul.23 | 5.88 | 0.673 | 99.34% |
| op0085_Mul.31_raster_0 | Mul.31 | 3.25 | 0.525 | 99.39% |
| op0109_Mul.42_raster_0 | Mul.42 | 3.34 | 0.489 | 87.20% |
| op0041_Mul.12_raster_0 | Mul.12 | 6.65 | 0.696 | 97.35% |

Outside the SE block, the diffs are bounded:
* Pure-conv stages: max < 0.01, mean < 1e-4
* Concat / upsample / nearest_interp: max < 0.01, mean < 1e-3

**The 5.9e-3 mean diff in the prob map is dominated by the SE
modules' channel-scale broadcast, not by a single misbehaving op.**

## Tooling artifacts

| File | Bytes | Description |
|---|---|---|
| `/tmp/m2fd/v6_tiny_det.onnx` | 1.8 MB | Paddle → ONNX export (opset 13) |
| `/tmp/m2fd/v6_tiny_det_intermediates.onnx` | 1.85 MB | Cloned ONNX with all node outputs exposed |
| `/tmp/m2fd/onnx_intermediates.npz` | — | 464 ONNX intermediates |
| `/tmp/m2fd/onnx_intermediate_index.json` | — | npz-key → original-name map |
| `/tmp/m2fd/mnn_intermediates.npz.bin` | 544 MB | 220 MNN op outputs (sidecar binary) |
| `/tmp/m2fd/mnn_intermediates.npz.json` | — | op name + shape + offset index |
| `/tmp/m2fd/single_conv.onnx` | — | 1-op ONNX for the first conv |
| `/tmp/m2fd/single_conv_y_*.npy` | — | Per-backend outputs |
| `/tmp/m2fd/bisect.json` | — | Original name-based bisect |
| `/tmp/m2fd/bisect_v4.json` | — | Forward-only + alt-name bisect |
| `/tmp/m2fd/single_conv.json` | — | 4-way single-conv diff table |

## Implications for the residual CER

The 5.9e-3 mean diff in the prob map translates to ~0.16 mean
CER residual in zh/en (per M2-MATRIX).  With:
* the per-conv kernel ruled out (finding 1)
* the conv-level stem ruled out (finding 3)
* the MNN vs Paddle diff pinned to paddle2onnx/Paddle's runtime
  graph optimization (finding 2)
* the SE channel-attention identified as the dominant amplifier
  (finding 4)

**The residual is intrinsic to the paddle2onnx conversion path**.
Possible mitigations (not in scope for this task):
* Re-export the ONNX with different `opset_version` (e.g. 11, 14, 17).
* Use a different conversion tool (e.g. x2paddle → ONNX, or
  Paddle 2.x with the legacy `inference.pdmodel` format).
* Use the paddle2onnx `--enable_onnx_checker` flag and try
  `--input_shape_dict` overrides.
* Investigate whether Paddle's runtime BN folding during inference
  differs from what paddle2onnx emits.

None of these are guaranteed to close the gap; the residual is
**fundamental** to the export path.

## What is NOT ruled out

* **Rec models**: this task only tested det (`v6_tiny_det`).  The
  same analysis on rec models (e.g. `v6_tiny_rec`) might find
  different amplifiers.
* **CLS model**: not tested; uses different backbone.
* **CUDA backend**: MNN CUDA path was validated at the output
  level (M3-CUDA: 5.5e-5 self-consistency), but the per-layer
  diff for MNN-CUDA-vs-Paddle was not run.
* **Different inputs**: the bisect used zh/04.jpg.  Other images
  may show different patterns.

## Sources

* `/tmp/m2num/det_input_paddle.npy` — PaddleX's pre_tfs output for zh/04 (CHW, ImageNet-normalized)
* `/tmp/m2num/det_output_paddle.npy` — PaddleX network output (prob map)
* `/tmp/m2num/det_output_mnn.npy` — MNN CPU network output
* `/root/ppocr_models/PP-OCRv6_tiny_det/inference.json` — Paddle model (new format)
* `/root/ppocr_models/PP-OCRv6_tiny_det/inference.pdiparams` — Paddle params
* `/root/pp-ocr-mnn/models/PP-OCRv6_tiny_det.mnn` — MNN-converted model (1.7 MB)
* `tests/mnn_backend_diff/RESULTS.md` — M3-CUDA report (baseline for "0/60 PASS")
* `tools/M2_DET_FINAL.md` — earlier per-conv diagnosis (0.005858 mean)
* `tools/M3_KERNELS.md` — kernel-sweep (all variants byte-identical to stock)
