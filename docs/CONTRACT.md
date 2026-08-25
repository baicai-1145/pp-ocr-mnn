# pp-ocr-mnn Internal Contract (v1) — binding for all executors

Authoritative interface freeze. Changing anything here requires decision-maker approval.
Module owners: `m1` = core/pipeline, `post` = postprocess, `tools` = python tooling.

## Repo / workflow

- Main repo: `/root/pp-ocr-mnn` (branch `main`). Decision-maker only merges.
- Executor worktrees: `/root/pp-ws/<name>` on branch `ws/<name>`. Commit there.
  `third_party/MNN` and `models/` in each worktree are symlinks to the main repo's copies.
- Build dirs: `build-<name>/` inside your worktree (gitignored pattern `build*/`). NEVER share.
- Never commit weights, test images, baselines, build outputs, results. `configs/*.json` (model
  configs incl. dicts) and source are committable.

## Public C ABI — `include/ppocr/ppocr.h` (authored by decision-maker, m1 implements)

See the file. Frozen: handle type, status enum, backend enum, `ppocr_config`, `ppocr_line`,
`ppocr_result`, create/run/free signatures, async stub. Implementation lives in `src/ppocr.cpp`
which also owns `pickBackend()` (the ONLY place backend selection happens).

## Module headers (namespace `ppocr`, header-guard style `PPOCR_<MODULE>_H_`)

### `include/ppocr/image.h` (owner m1)
```cpp
struct Image { int w=0, h=0, c=0; std::vector<uint8_t> data; }; // 8-bit BGR, c==3
Image load_image(const std::string& path);   // stb_image; gray/RGBA→BGR; empty data on error
bool  save_image(const std::string& path, const Image&); // stb_image_write (png/jpg by ext)
```

### `include/ppocr/config.h` (owner m1)
```cpp
struct DetResizeConfig { enum class Mode { LimitMin, ResizeLong };
  Mode mode; int limit_side_len=736; int resize_long=960; int stride; int max_side_limit=4000; };
struct DetConfig { float thresh=0.3f, box_thresh=0.6f, unclip_ratio=1.5f;
  int max_candidates=1000; DetResizeConfig resize; };
struct RecConfig { int c=3,h=48,w=320; std::vector<std::string> dict; bool use_space=true; };
struct ClsConfig { int c=3,h=80,w=160; std::vector<float> mean{0.485f,0.456f,0.406f},
  std{0.229f,0.224f,0.225f}; std::vector<std::string> labels; };
struct ModelConfig { std::string name, type /*det|rec|cls*/, file, sha256, url; uint64_t bytes=0;
  DetConfig det; RecConfig rec; ClsConfig cls; };
ModelConfig load_model_config(const std::string& json_path); // throws std::runtime_error
```

### `include/ppocr/postprocess/geometry.h` (owner post)
```cpp
struct PointF { float x=0, y=0; };
// min-area rect via rotating calipers; out points ordered like cv::minAreaRect().points
bool min_area_rect(const PointF* pts, size_t n, PointF out[4]);
// re-order 4 box points to PaddleOCR canonical order (port of order_points_clockwise +
// GetMinAreaRectPoints sorting from deploy/cpp_infer/src/common/processors.cc)
void sort_min_area_rect_points(PointF box[4]);
// inverse-map bicubic warp, border replicate (port of cv::warpPerspective INTER_CUBIC BORDER_REPLICATE)
// src quad maps to dst rect [0,0]-[w-1,h-1]
Image warp_perspective_quad(const Image& src, const PointF quad[4], int dst_w, int dst_h);
```

### `include/ppocr/postprocess/db_post.h` (owner post)
```cpp
struct DetBox { float poly[8]; float score=0; }; // poly = x0,y0,...,x3,y3 in ORIGINAL image coords
// Faithful port of ppocr/data/postprocess/db_postprocess.py (boxes_from_bitmap +
// filter_tag_det_boxes + get_sorted_boxes). Boxes returned in official reading order.
std::vector<DetBox> db_postprocess(const float* prob, int prob_h, int prob_w,
                                   int src_w, int src_h, float ratio_w, float ratio_h,
                                   const DetConfig& cfg);
```
Reference (do NOT reinvent): `/root/PaddleOCR/ppocr/data/postprocess/db_postprocess.py`,
clipper semantics from `/root/PaddleOCR/deploy/cpp_infer/src/common/clipper.hpp` (header-only,
may be vendored to `third_party/clipper/clipper.hpp` — allowed, it is not MNN).

### `include/ppocr/postprocess/ctc_decode.h` (owner post)
```cpp
struct RecOut { std::string text; float score=0; }; // text UTF-8
// char list = ["blank"] + dict + (use_space ? [" "] : []); blank index 0.
// collapse repeats, drop blanks; score = mean prob of emitted chars (official semantics).
RecOut ctc_decode(const float* logits, int timesteps, int num_classes, const RecConfig& cfg);
```
Reference: `/root/PaddleOCR/ppocr/postprocess/rec_postprocess.py` (CTCLabelDecode + BaseRecLabelDecode).

### `include/ppocr/preprocess.h` (owner m1)
```cpp
struct DetInput { int in_w=0, in_h=0; float ratio_w=1, ratio_h=1; std::vector<float> chw; };
// DetResizeForTest port: LimitMin→type0(736/min, round-to-32), ResizeLong→type2(stride 128).
// normalization mean {0.485,0.456,0.406} std {0.229,0.224,0.225} scale 1/255, BGR input.
DetInput prep_det(const Image& bgr, const DetResizeConfig& rc);
// one rec line: resize keep-ratio to h=48, w'=clamp(ceil(48*wh_ratio),1,batch_w); norm (x/255-0.5)/0.5;
// CHW float32, zero-pad to batch_w; valid_w = w'.
std::vector<float> prep_rec_line(const Image& line_bgr, int img_h, int batch_w, int& valid_w);
// cls: resize to cfg.h*w, ImageNet norm, CHW
std::vector<float> prep_cls(const Image& bgr, const ClsConfig& cfg);
```
Reference: `/root/PaddleOCR/ppocr/data/imaug/operators.py` (DetResizeForTest types 0/2),
`/root/PaddleOCR/ppocr/data/imaug/rec_img_aug.py` (`resize_norm_img`).

### `include/ppocr/mnn_session.h` (owner m1)
Thin RAII wrapper: create from .mnn file + backend, `resize_input(name,{...})`, `run()`,
`output(name) -> const float* + shape`. Must support being copied per thread-safety rule (no
global cache). CPU threads config honored.

## Model config JSON (owner tools generates, m1 consumes)

`configs/<ModelName>.json` + aggregate `configs/registry.json` (name → {file, sha256, bytes, url, type}).
```json
{
  "name": "PP-OCRv6_tiny_rec", "type": "rec",
  "file": "PP-OCRv6_tiny_rec.mnn", "sha256": "…", "bytes": 4434868,
  "url": "PP-OCRv6_tiny_rec.mnn",
  "rec": { "shape": [3,48,320], "use_space": true, "dict": ["…", "…"] }
}
```
det variant: `"det": {"thresh":0.2,"box_thresh":0.4,"unclip_ratio":1.4,"max_candidates":3000,
"resize":{"mode":"limit_min","limit_side_len":736,"stride":32,"max_side_limit":4000}}`
(v4/v5 dets: mode "resize_long", long 960, stride 128; seal: long 736).
cls variant: `"cls": {"shape":[3,80,160],"labels":["0_degree","180_degree"],
"mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]}`.

## CLI (owner m1) — contract used by run_reference.py

```
ppocr_cli --image IMG --det-config configs/X_det.json --rec-config configs/Y_rec.json \
          [--cls-config configs/cls.json] [--backend auto|cpu|cuda|opencl|vulkan] \
          [--threads N] [--batch N] [--det-only] [--json OUT] [--time]
```
stdout (or OUT): `{"image":"…","backend":"cpu","lines":[{"poly":[8×int],"text":"…","score":0.97}],
"ms":{"det":12.3,"rec":8.1,"total":21.0}}`  — `poly` ints in original image coords.
`--det-only` → lines have `text:""`, `score`=det box score.
Exit 0 on success, nonzero + stderr message on any failure.

## Python tools (owner tools)

- `tools/convert_models.py` — full pipeline paddle→onnx→mnn for ALL 30 models
  (7 det + 2 seal + 20 rec + cls at `~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori`).
  Reuse existing `models/_onnx/*.onnx` when fresh; cls must be converted new. Emits
  `models/*.mnn` + `configs/*.json` + `configs/registry.json` (sha256 over .mnn file).
  `--mnnconvert PATH` default `third_party/MNN/build/MNNConvert` (via main repo symlink).
- `tools/extract_dict.py` — inference.yml → dict list + det params (library used by above).
- `tools/run_reference.py` — iterate `/root/ppocr_reference/*__*/` cells (skip `strip__`,
  `manifest*`, seal dirs for full-OCR mode): map combo→configs, run ppocr_cli per image,
  write `results/<combo>/<lang>/pred.json` (same schema as baseline minus GT fields).
  `--cells` filter, `--jobs N` parallel processes, `--backend`, `--cli PATH`.
- `tools/score.py` — CER per cell: per image, CER = levenshtein(pred_join, base_join)/len(base_join)
  where join = `"\n".join(rec_texts)`; cell score = mean over images; PASS ≤ 0.05.
  `--strip` mode scores `strip__*` cells vs `/root/ocr_test_imgs/strip_gt.json`;
  seal cells are M4 scope (report N/A). Output `results/report.md` matrix + exit code
  (0 = all PASS, 1 = any FAIL).

## Verification gates (all executors)

- C++: `cmake -S . -B build-X -DCMAKE_BUILD_TYPE=Release && cmake --build build-X -j`
  must be clean; your module's tests (`tests/test_<module>.cpp`, plain asserts, no framework)
  run via `./build-X/tests/test_<module>` green.
- Python tools: `python3 -m py_compile` + a real invocation whose output you paste in the report.
- Report format when done: branch name, files changed, how verified (commands + key output),
  open questions/risks. Do not claim done without pasted evidence.
