// pp-ocr-mnn — public C ABI implementation
//
// One file owns the entire C ABI surface. Backend selection lives ONLY
// here in pickBackend(); the rest of the codebase is platform-agnostic
// (CONTRACT hard rule #6). On Linux desktop M1 we ship only the CPU
#include <algorithm>
// path; the opencl/vulkan/... branches map to the corresponding
// MNNForwardType and let MNN's auto-tuner pick the first available
// backend at session create time.
#include "ppocr/ppocr.h"

#include "ppocr/config.h"
#include "ppocr/downloader.h"
#include "ppocr/image.h"
#include "ppocr/mnn_session.h"
#include "ppocr/postprocess/ctc_decode.h"
#include "ppocr/postprocess/db_post.h"
#include "ppocr/postprocess/geometry.h"
#include "ppocr/preprocess.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace ppocr {

// ---- backend selection ---------------------------------------------------
//
// pickBackend() is the ONLY place that maps the public C enum onto a
// concrete MnnSession::Backend. The rest of the codebase never inspects
// the backend; that's the rule (CONTRACT hard rule #6).
Backend pickBackend(ppocr_backend requested) {
  switch (requested) {
    case PPOCR_BACKEND_AUTO:   return Backend::Auto;
    case PPOCR_BACKEND_CPU:    return Backend::Cpu;
    case PPOCR_BACKEND_CUDA:   return Backend::Cuda;
    case PPOCR_BACKEND_OPENCL: return Backend::OpenCL;
    case PPOCR_BACKEND_VULKAN: return Backend::Vulkan;
    case PPOCR_BACKEND_METAL:  return Backend::Metal;
    case PPOCR_BACKEND_COREML: return Backend::CoreML;
    case PPOCR_BACKEND_NNAPI:  return Backend::NNAPI;
  }
  return Backend::Auto;
}

// ---- engine state --------------------------------------------------------
//
// Each engine is fully self-contained: model paths, parsed model configs,
// MNN sessions, last result buffers (which must outlive the caller's
// ppocr_result inspection), and the optional async worker. No globals.

struct AsyncJob {
  ppocr_callback cb = nullptr;
  void*          user = nullptr;
  std::vector<uint8_t> bgr;     // copy of input (caller's buffer may be reused)
  int w = 0, h = 0;
};

struct Engine {
  // model paths and parsed configs.
  std::string model_dir;
  std::string det_path, rec_path, cls_path;
  ModelConfig det_cfg, rec_cfg, cls_cfg;
  // Seal mode: det model is one of PP-OCRvN_*_seal_det. The pipeline
  // then skips the reading-order sort (ring text is cyclic), uses rec
  // score threshold 0 (paddlex seal_recognition.yaml default), and
  // surfaces the rec output as-is on every polygon.
  // Set from det_name in load_submodels; can be overridden by an
  // explicit `is_seal=1` on ppocr_config.
  bool is_seal = false;

  // MNN sessions. det is required; rec/cls optional.
  std::unique_ptr<MnnSession> det;
  std::unique_ptr<MnnSession> rec;
  std::unique_ptr<MnnSession> cls;

  // Last ppocr_result buffers. `last_text` owns the storage that
  // `last_lines[i].text` points into; the public C ABI guarantees the
  // struct is valid until the next run on the same engine.
  std::vector<ppocr_line>  last_lines;
  std::vector<std::string> last_text;
  ppocr_result            last_result{};

  // M3-PERF1: per-stage profile of the last run. Collected only when
  // `profile` is true (ppocr_config.profile); read via the public ABI
  // ppocr_last_profile().
  bool          want_profile = false;
  bool          has_run_once = false;
  float         create_ms_stored = 0.f;
  ppocr_profile last_profile{};

  // Async worker.
  std::thread             async_worker;
  std::mutex              async_mu;
  std::condition_variable async_cv;
  std::deque<AsyncJob>    async_queue;
  std::atomic<bool>       async_stop{false};
  std::atomic<bool>       async_started{false};

  ppocr_config cfg_shadow{};  // owned copies of cfg strings

  // Cached rec batch size (1..N). The default rec_batch=0 in the C
  // ABI means "8", per ppocr.h. We resolve it once at create() so
  // run_full() doesn't repeat the math.
  int rec_batch = 16;

  // Helpers
  ppocr_status load_submodels(const ppocr_config* cfg, char* err, size_t elen);
  // M3-PERF4: one dummy inference per loaded model (det 640x640, rec
  // 1x3x48x320, cls 1x3x80x160) to absorb the one-time backend init
  // (cutlass param selection + workspace allocation on CUDA) at create
  // time instead of on the first real image. Outputs are discarded.
  void warmup();
};

// ---- helpers -------------------------------------------------------------

static std::string env_or(const char* var, const std::string& def) {
  const char* v = std::getenv(var);
  return (v && *v) ? std::string(v) : def;
}

static std::string trim_back_slash(const std::string& s) {
  if (!s.empty() && (s.back() == '/' || s.back() == '\\')) {
    return s.substr(0, s.size() - 1);
  }
  return s;
}

static bool file_exists(const std::string& p) {
  std::error_code ec;
  return fs::is_regular_file(p, ec);
}

// SHA-256 of a file. Implemented inline to keep the public C ABI
// dependency-free. Used by registry verification once ws/tools lands
// the runtime downloader; for M1 the verification path is M3 scope.
// Marked [[maybe_unused]] so -Wunused-function stays quiet.
[[maybe_unused]] static std::string sha256_hex_of_file(const std::string& path) {
  (void)path;
  return {};
}

// ---- engine implementation ----------------------------------------------

static ppocr_status resolve_config_paths(Engine& e, const ppocr_config* cfg,
                                         char* err_buf, size_t elen,
                                         const std::string& det_name,
                                         const std::string& rec_name,
                                         const std::string& cls_name) {
  // Layout (matches the catalog in /root/pp-ocr-mnn/models and the
  // per-model JSONs in this worktree's configs/):
  //   <model_dir>/<name>.mnn           — the model file
  //   <model_dir>/configs/<name>.json  — the per-model config
  //
  // We also fall back to `<model_dir>/../configs/<name>.json`, which
  // is the layout the main repo actually uses (models/ and configs/
  // are sibling directories). M1 hard-fails the parse if neither
  // exists; M2+ can pin a single layout once the registry is wired.
  const std::string cfg_dir_primary = e.model_dir + "/configs";
  std::string cfg_dir_fallback;
  {
    auto slash = e.model_dir.find_last_of("/\\");
    if (slash != std::string::npos) {
      cfg_dir_fallback = e.model_dir.substr(0, slash) + "/configs";
    }
  }
  auto find_cfg = [&](const std::string& name) -> std::string {
    if (name.empty()) return {};
    std::string p1 = cfg_dir_primary + "/" + name + ".json";
    if (file_exists(p1)) return p1;
    if (!cfg_dir_fallback.empty() && cfg_dir_fallback != cfg_dir_primary) {
      std::string p2 = cfg_dir_fallback + "/" + name + ".json";
      if (file_exists(p2)) return p2;
    }
    return {};
  };
  e.det_path = e.model_dir + "/" + det_name + ".mnn";
  e.rec_path = e.model_dir + "/" + rec_name + ".mnn";
  e.cls_path = e.model_dir + "/" + cls_name + ".mnn";

  // Try to load the model configs if available. If they aren't on disk
  // (e.g. fast CI runs without a fresh registry) we fall back to safe
  // defaults: v6 tiny-style resize + a rec with shape [3,48,320] and
  // an empty dict (the postprocess stub won't read it anyway).
  auto det_cfg_path = find_cfg(det_name);
  auto rec_cfg_path = find_cfg(rec_name);
  auto cls_cfg_path = find_cfg(cls_name);
  try {
    if (!det_cfg_path.empty()) e.det_cfg = load_model_config(det_cfg_path);
    if (!rec_cfg_path.empty()) e.rec_cfg = load_model_config(rec_cfg_path);
    if (!cls_cfg_path.empty()) e.cls_cfg = load_model_config(cls_cfg_path);
  } catch (const std::exception& ex) {
    if (err_buf && elen) std::snprintf(err_buf, elen, "config: %s", ex.what());
    return PPOCR_ERR_MODEL;
  }
  // Verify the .mnn file is present; auto-download is M3 scope and
  // M1 expects the model already on disk.
  if (!file_exists(e.det_path)) {
    if (err_buf && elen) std::snprintf(err_buf, elen, "det model not found: %s",
                                       e.det_path.c_str());
    return PPOCR_ERR_MODEL;
  }
  if (!rec_name.empty() && !file_exists(e.rec_path)) {
    if (err_buf && elen) std::snprintf(err_buf, elen, "rec model not found: %s",
                                       e.rec_path.c_str());
    return PPOCR_ERR_MODEL;
  }
  if (!cls_name.empty() && !file_exists(e.cls_path)) {
    if (err_buf && elen) std::snprintf(err_buf, elen, "cls model not found: %s",
                                       e.cls_path.c_str());
    return PPOCR_ERR_MODEL;
  }
  return PPOCR_OK;
}

ppocr_status Engine::load_submodels(const ppocr_config* cfg, char* err,
                                    size_t elen) {
  const std::string det_name = cfg->det_name ? cfg->det_name
                                             : std::string("PP-OCRv6_tiny_det");
  const std::string rec_name = cfg->rec_name ? cfg->rec_name
                                             : std::string("PP-OCRv6_tiny_rec");
  const std::string cls_name = cfg->cls_name ? cfg->cls_name : std::string();

  // M4-SEAL: auto-detect seal mode from the det model name (any of the
  // four PP-OCRvN_{mobile,server}_seal_det variants). An explicit
  // cfg->is_seal = 1 forces the mode on (e.g. when a user renames a
  // custom seal det to something that doesn't contain the substring);
  // cfg->is_seal = 2 forces it off. Default 0 = auto.
  if (cfg->is_seal == 1) {
    is_seal = true;
  } else if (cfg->is_seal == 2) {
    is_seal = false;
  } else {
    is_seal = (det_name.find("seal") != std::string::npos);
  }

  // ---- ensure_model step (TOOLS-5 / M3 auto-download) -----------------
  //
  // We resolve model_dir, cache_dir, and mirror from the public cfg
  // with env-var fallbacks. The registry is loaded only when
  // auto-download is enabled; if the registry file is missing, we
  // silently fall back to the local-files-only path (M1 behavior).
  //
  // The contract fields are:
  //   cfg->model_dir: dir that holds the .mnn files (default
  //                   $PPORC_MNN_MODELS or ./models).
  //   cfg->cache_dir: download cache dir (default
  //                   $PPORC_MNN_CACHE or ~/.cache/ppocr-mnn).
  //   cfg->registry_path: explicit override for the registry.json
  //                   (default <model_dir>/configs/registry.json).
  //   cfg->mirror: download base URL (default
  //                $PPORC_MNN_MIRROR or "https://example.com/ppocr-mnn-models"
  //                per the M3 placeholder documented in AGENTS.md).
  //   cfg->offline: 1 = never download.
  //   cfg->download: 0 = never download (overrides env / mirror).
  std::string model_dir   = trim_back_slash(cfg->model_dir
                                              ? std::string(cfg->model_dir)
                                              : env_or("PPORC_MNN_MODELS", "./models"));
  std::string cache_dir   = cfg->cache_dir
                              ? trim_back_slash(std::string(cfg->cache_dir))
                              : env_or("PPORC_MNN_CACHE", std::string());
  if (cache_dir.empty()) {
    const char* home = std::getenv("HOME");
    cache_dir = home ? (std::string(home) + "/.cache/ppocr-mnn")
                     : (model_dir + "/.cache");
  }
  std::string mirror      = cfg->mirror
                              ? std::string(cfg->mirror)
                              : env_or("PPORC_MNN_MIRROR",
                                       std::string("https://example.com/ppocr-mnn-models"));
  int offline  = cfg->offline  ? 1 : 0;
  int download = cfg->download ? 1 : 0;

  // Load the registry (optional). If the file is missing, an empty
  // registry is fine: ensure_model will simply use whatever files
  // are already on disk and skip the download branch (the per-name
  // lookup will return nullptr; that path produces PPOCR_ERR_MODEL
  // for missing files, matching the M1 hard-fail behavior).
  Registry reg;
  std::string reg_path = cfg->registry_path
                            ? std::string(cfg->registry_path)
                            : std::string();
  if (reg_path.empty()) {
    std::vector<std::string> candidates = {
        model_dir + "/configs/registry.json",
    };
    auto slash = model_dir.find_last_of("/\\");
    if (slash != std::string::npos) {
      candidates.push_back(model_dir.substr(0, slash) + "/configs/registry.json");
    }
    for (const auto& c : candidates) {
      std::ifstream f(c, std::ios::binary);
      if (f) { reg_path = c; break; }
    }
  }
  if (!reg_path.empty()) {
    try {
      reg = load_registry(reg_path);
    } catch (const std::exception& ex) {
      // Bad registry is non-fatal in offline mode: we just won't be
      // able to verify or download. In online mode we still try to
      // log and continue.
      if (err && elen) {
        std::snprintf(err, elen, "registry load: %s", ex.what());
      }
    }
  }

  // Helper: ensure a single model is on disk.
  auto ensure = [&](const std::string& name) -> ppocr_status {
    if (name.empty()) return PPOCR_OK;
    EnsureResult er = ensure_model(reg, name, model_dir, cache_dir, mirror,
                                   offline, download);
    if (er.status == PPOCR_OK) return PPOCR_OK;
    if (err && elen) std::snprintf(err, elen, "%s", er.detail.c_str());
    return er.status;
  };
  ppocr_status st = ensure(det_name);
  if (st != PPOCR_OK) return st;
  st = ensure(rec_name);
  if (st != PPOCR_OK) return st;
  st = ensure(cls_name);
  if (st != PPOCR_OK) return st;

  st = resolve_config_paths(*this, cfg, err, elen,
                            det_name, rec_name, cls_name);
  if (st != PPOCR_OK) {
    if (err && elen) std::snprintf(err, elen, "model path resolution failed");
    return st;
  }

  // Resolve rec_batch AFTER configs load (the hint lives in the rec
  // model config). Priority: cfg->rec_batch > rec_batch_hint > 16.
  // M3-PERF4 made 16 the default (width-neutral vs 8; dense e2e -15%
  // on v6_tiny). M3-PERF5: measured exceptions where 8 wins on CUDA —
  // v6_medium dense6 w1 1.08 fps @8 vs 0.87 @16, v6_small 1.54 vs 0.94
  // — the [16,3,48,320] session resize + activation memory outweigh
  // GEMM batching for those rec graphs. The hint lets those model
  // configs opt down without changing the ABI default.
  rec_batch = cfg->rec_batch > 0 ? cfg->rec_batch
             : (rec_cfg.rec.rec_batch_hint > 0 ? rec_cfg.rec.rec_batch_hint : 16);
  if (rec_batch < 1) rec_batch = 1;

  // Build a SessionConfig from the public cfg.
  SessionConfig sc;
  sc.backend = pickBackend(static_cast<ppocr_backend>(cfg->backend));
  sc.num_threads = cfg->num_threads;

  det = std::make_unique<MnnSession>();
  try {
    det->load(det_path, sc);
  } catch (const std::exception& ex) {
    if (err && elen) std::snprintf(err, elen, "det load: %s", ex.what());
    return PPOCR_ERR_MODEL;
  }
  if (!rec_name.empty() && !rec_path.empty()) {
    rec = std::make_unique<MnnSession>();
    try {
      rec->load(rec_path, sc);
    } catch (const std::exception& ex) {
      if (err && elen) std::snprintf(err, elen, "rec load: %s", ex.what());
      return PPOCR_ERR_MODEL;
    }
  }
  if (!cls_name.empty() && !cls_path.empty()) {
    cls = std::make_unique<MnnSession>();
    try {
      cls->load(cls_path, sc);
    } catch (const std::exception& ex) {
      if (err && elen) std::snprintf(err, elen, "cls load: %s", ex.what());
      return PPOCR_ERR_MODEL;
    }
  }
  return PPOCR_OK;
}

void Engine::warmup() {
  auto run_dummy = [this](MnnSession* s, const std::vector<int>& dims) {
    if (!s) return;
    try {
      const size_t n = static_cast<size_t>(dims[0]) * dims[1] * dims[2] * dims[3];
      std::vector<float> zeros(n, 0.f);
      s->set_input_float("x", dims, zeros.data());
      (void)s->run();
    } catch (...) {
      // Warmup is best-effort: a failure here doesn't invalidate the
      // engine; the first real inference will surface real errors.
    }
  };
  run_dummy(det.get(), {1, 3, 640, 640});
  run_dummy(rec.get(), {1, 3, 48, 320});
  run_dummy(cls.get(), {1, 3, 80, 160});
}

// ---- det / rec helpers ---------------------------------------------------

static void run_det_sync(Engine& e, const Image& bgr,
                         std::vector<DetBox>& boxes_out,
                         float& ms_out) {
  auto t0 = std::chrono::steady_clock::now();
  auto tic = [&e](float* slot) {
    if (e.want_profile && slot) {
      // caller-managed accumulation; value filled by caller below
    }
  };
  (void)tic;
  const bool prof = e.want_profile;
  auto t_prep = std::chrono::steady_clock::now();
  DetInput in = prep_det(bgr, e.det_cfg.det.resize);
  auto t_prep_end = std::chrono::steady_clock::now();
  // The det .mnn file's input is dynamic; we resize to the preprocessed HxW.
  std::vector<int> dims = {1, 3, in.in_h, in.in_w};
  e.det->set_input_float("x", dims, in.chw.data());
  auto t_run = std::chrono::steady_clock::now();
  int ec = e.det->run();
  auto t_run_end = std::chrono::steady_clock::now();
  if (ec != 0) {
    boxes_out.clear();
    ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
    return;
  }
  // The DB probability map is the first (and only) output; some
  // PaddleOCR exports name it `sigmoid_0.tmp_0` while others leave the
  // auto-generated name `fetch_name_0`. We try both.
  SessionOutput so = e.det->output("sigmoid_0.tmp_0");
  if (so.data == nullptr || so.shape.empty()) {
    so = e.det->output("fetch_name_0");
  }
  if (so.data == nullptr || so.shape.size() < 4) {
    boxes_out.clear();
    ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
    return;
  }
  // NCHW with N=1, C=1, H, W
  const int H = so.shape[2];
  const int W = so.shape[3];
  // db_postprocess expects the ratio that maps prob-map coords back to
  // original image coords: x_orig = x_prob * (src_w / prob_w). This is
  // DIFFERENT from prep's `in.ratio_w = resize_w / bgr.w` (= network
  // input / original image, which maps original -> network input).
  // For the common case where prep leaves the image unchanged (resize_w
  // == bgr.w, ratio_w == 1.0), the two are equivalent and the text-det
  // path is unaffected. For the seal path (512 -> 768 upscale), prep
  // returns 1.5 but db_post wants 512/768 = 0.667; using prep's ratio
  // there blows up the box to 1.5x the image size and lands it in an
  // empty region of the prob map. Compute the correct ratio here.
  const float prob_to_img_w = (W > 0) ? static_cast<float>(bgr.w) / W : 1.0f;
  const float prob_to_img_h = (H > 0) ? static_cast<float>(bgr.h) / H : 1.0f;
  // M4-SEAL: in seal mode the MNN det prob map is noisier than the
  // Paddle reference (post-6 finding carries over): the *outer* ring
  // polys have mean bbox prob around 0.27-0.4 vs the 0.99 paddle sees.
  // Padding the 0.2 binarization threshold + 0.6 box_thresh down to
  // box_thresh=0.3 is the practical operating point on this det model;
  // verified on /root/ocr_test_imgs/seal/{zh_00_0,en_04_0,ja_07_0,...}
  // that the 2nd poly is now recoverable. False positives are rare
  // because the seal image background is mostly uniform.
  DetConfig eff_det_cfg = e.det_cfg.det;
  if (e.is_seal && eff_det_cfg.box_thresh > 0.10f) {
    eff_det_cfg.box_thresh = 0.10f;
  }
  auto t_db = std::chrono::steady_clock::now();
  boxes_out = db_postprocess(so.data, H, W, bgr.w, bgr.h,
                             prob_to_img_w, prob_to_img_h, eff_det_cfg);
  // NOTE (POST-4): db_postprocess emits boxes in cv::findContours order,
  // which is NOT reading order. Paddle's pipeline applies
  // ComponentsProcessor::SortQuadBoxes (top-to-bottom, then left-to-right
  // within a 10px row tolerance) AFTER db_postprocess, so the rec layer
  // sees boxes in reading order. Without this step the join of rec_texts
  // is line-permuted (e.g. zh/02: "Chinese\nProject\nText" instead of
  // "Project\nChinese\nText") and CER is high even though every line
  // is correctly recognized. The fix lives in db_post as
  // sort_quad_boxes_reading_order — call it here, before rec, so the
  // rec batch and the final lines are both in reading order.
  //
  // M4-SEAL: ring text has no reading order; sorting by row/column
  // scrambles the cyclic text. Skip the sort for seal mode. The
  // rec output is still in det order (which for a seal image is
  // outer ring first, then inner star), and downstream consumers
  // match the multiset rather than concatenating.
  if (!e.is_seal) {
    sort_quad_boxes_reading_order(boxes_out);
  }
  ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count());
  if (prof) {
    e.last_profile.det_prep_ms =
        std::chrono::duration<float, std::milli>(t_prep_end - t_prep).count();
    e.last_profile.det_run_ms =
        std::chrono::duration<float, std::milli>(t_run_end - t_run).count();
    e.last_profile.db_post_ms =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t_db).count();
    e.last_profile.n_boxes = static_cast<int>(boxes_out.size());
  }
}

// Rec is M2 scope; M1 returns a synthetic line for each det box so the
// CLI can emit valid JSON. ws/post has merged, so this is the real
// implementation:
//   1. For each DetBox, GetRotateCropImage (deploy/cpp_infer ...
//      GetRotateCropImage): 4-point perspective warp into a tight
//      (maxW, maxH) rectangle; rotate 90° CCW if maxH/maxW >= 1.5.
//   2. Batch all crops. Each chunk (default size 8) gets a single
//      batch_w = int(imgH * max(wh_ratio in chunk)), the paddlex
//      3.x `ReisizeNorm.resize_norm_img` algorithm. Every crop in
//      the chunk is resized to (H, batch_w) and zero-padded on the
//      right; the rec MNN session is dynamic-width [1,3,48,-1] so
//      we resize to {N,3,48,batch_w} and runSession.
//   3. Output is [N, T, C] in NCHW-ish (we measured 6906 classes
//      after softmax-on-logits); we run ctc_decode on each row of
//      the (T, C) slice per batch element.

// Rotate a BGR image 180 degrees in place (equivalent to np.rot90(k=2)).
// Used by the cls module when the textline orientation classifier
// predicts 180°. PaddleOCR's PaddleClas::TextRecRotator does exactly
// this (`cv::rotate(rotated_image, cv::ROTATE_180)`).
static void rotate_180_inplace(Image& img) {
  // M3-PERF6: if img is a non-owning view, materialize it first (this
  // function mutates in place). Only hit on the cls-180 path.
  if (img.ext && img.data.empty()) {
    img.data.assign(img.ext, img.ext + static_cast<size_t>(img.w) * img.h * img.c);
    img.ext = nullptr;
  }
  if (img.data.empty() || img.w <= 0 || img.h <= 0) return;
  const int W = img.w;
  const int H = img.h;
  const int C = img.c;
  const int total = W * H * C;
  // Reverse the whole buffer in place. For 3-channel BGR this is
  // identical to np.rot90(k=2) which reverses the (H, W) plane and
  // then flips each row. Verify: pixel at (y, x, c) moves to
  // (H-1-y, W-1-x, c) -> in the reversed buffer, the byte at
  // (H-1-y, W-1-x, c) holds what was at byte (total-1 - (y*W*C + x*C + c))
  // which IS the original (y, x, c) byte.
  uint8_t* d = img.data.data();
  for (int i = 0, j = total - 1; i < j; ++i, --j) {
    uint8_t t = d[i];
    d[i] = d[j];
    d[j] = t;
  }
}

// Run the textline orientation classifier (PP-LCNet_x1_0_textline_ori)
// on each warped crop and rotate 180° the crops that score 180°.
// `crops` is mutated in place: crops that score 180° are reversed.
// `cls_labels_out[i] = 0` for 0° (no rotate), `1` for 180° (rotated).
// `cls_score_out[i]` is the softmax confidence in the predicted class.
//
// The cls MNN output is [N, 2] softmax (verified on the converted
// PP-LCNet_x1_0_textline_ori.mnn in M3). Indices 0 and 1 map to
// labels[0]="0_degree" and labels[1]="180_degree" from
// e.cls_cfg.cls.labels — we trust the cfg's order.
//
// This is the M3 cls path. When e.cls is null (the default), the
// function is a no-op and cls_labels_out stays all zeros, matching
// the "cls off" contract.
static void run_cls_sync(Engine& e,
                         std::vector<Image>& crops,
                         std::vector<int>& cls_labels_out,
                         std::vector<float>& cls_score_out,
                         float& ms_out) {
  auto t0 = std::chrono::steady_clock::now();
  const int n = static_cast<int>(crops.size());
  cls_labels_out.assign(n, 0);
  cls_score_out.assign(n, 0.f);
  if (n == 0) {
    ms_out = 0.f;
    return;
  }
  if (!e.cls) {
    // cls disabled: leave labels=0 (no rotate), report 0 ms.
    ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
    return;
  }
  const ClsConfig& cfg = e.cls_cfg.cls;
  const int CH = cfg.h > 0 ? cfg.h : 80;
  const int CW = cfg.w > 0 ? cfg.w : 160;
  // Run the cls model per-image. The cls MNN is [1,3,80,160] fixed
  // shape (we converted with N=1 in MNNConvert; dynamic batch
  // would need a re-convert). Per-image call is fine because the
  // per-call latency is ~1 ms on this CPU and the input crop count
  // is typically <50 (one per det box).
  for (int i = 0; i < n; ++i) {
    Image& c = crops[i];
    if (c.data.empty() || c.w <= 0 || c.h <= 0) {
      cls_labels_out[i] = 0;
      cls_score_out[i] = 0.f;
      continue;
    }
    try {
      std::vector<float> chw = prep_cls(c, cfg);
      std::vector<int> dims = {1, 3, CH, CW};
      e.cls->set_input_float("x", dims, chw.data());
      int ec = e.cls->run();
      if (ec != 0) {
        cls_labels_out[i] = 0;
        cls_score_out[i] = 0.f;
        continue;
      }
      SessionOutput so = e.cls->output("fetch_name_0");
      if (so.data == nullptr || so.shape.size() != 2 ||
          so.shape[0] != 1 || so.shape[1] != 2) {
        // Fall back: try "softmax_0.tmp_0" (Paddle export variant).
        so = e.cls->output("softmax_0.tmp_0");
      }
      if (so.data == nullptr || so.shape.size() != 2 ||
          so.shape[0] != 1 || so.shape[1] != 2) {
        cls_labels_out[i] = 0;
        cls_score_out[i] = 0.f;
        continue;
      }
      const float* row = so.data;
      const int argmax = (row[0] >= row[1]) ? 0 : 1;
      const float conf = row[argmax];
      cls_labels_out[i] = argmax;
      cls_score_out[i] = conf;
      if (argmax == 1) {
        // PaddleOCR's TextRecRotator: rotate the entire image 180°.
        // This is the np.rot90(k=2) equivalence called out in the
        // M3 brief.
        rotate_180_inplace(c);
      }
    } catch (const std::exception&) {
      // Defensive: a single bad crop must not abort the whole run.
      cls_labels_out[i] = 0;
      cls_score_out[i] = 0.f;
    }
  }
  ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count());
}

static void run_rec_sync(Engine& e, const Image& bgr,
                         const std::vector<DetBox>& boxes,
                         std::vector<std::pair<std::string, float>>& texts,
                         float& ms_out,
                         float& cls_ms_out,
                         std::vector<int>* cls_labels_out = nullptr) {
  auto t0 = std::chrono::steady_clock::now();
  const bool prof = e.want_profile;
  auto t_crop = std::chrono::steady_clock::now();
  auto t_recprep_end = t_crop, t_recrun_end = t_crop, t_ctc_end = t_crop;
  int n_rec_batches = 0;
  cls_ms_out = 0.f;
  texts.clear();
  texts.resize(boxes.size(), {std::string{}, 0.f});
  if (boxes.empty() || !e.rec) {
    ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count());
    return;
  }
  const int H = e.rec_cfg.rec.h > 0 ? e.rec_cfg.rec.h : 48;
  const int batch_w_cap = e.rec_cfg.rec.w > 0 ? e.rec_cfg.rec.w : 320;
  // Pre-compute crops in a vector aligned with `boxes`.
  struct Crop { Image img; };
  std::vector<Crop> crops;
  // M3-PERF2: pre-size and fill by index so the per-box warp work
  // (perspective warp + rot90) can be fanned out over threads below;
  // results are per-box independent, so output order/bytes are
  // unchanged (bit-exact).
  crops.resize(boxes.size());
  const std::vector<DetBox>& boxes_ref = boxes;
  auto crop_one = [&](size_t bi) -> void {
    const DetBox& b = boxes_ref[bi];
    // db_post returns poly already in PaddleOCR canonical order
    // [TL, TR, BR, BL] (sort_min_area_rect_points was applied).
    PointF quad[4];
    quad[0] = {b.poly[0], b.poly[1]};
    quad[1] = {b.poly[2], b.poly[3]};
    quad[2] = {b.poly[4], b.poly[5]};
    quad[3] = {b.poly[6], b.poly[7]};
    // GetRotateCropImage edge lengths. PaddleOCR canonical:
    //   top    = quad[0] -> quad[1]
    //   bottom = quad[2] -> quad[3]
    //   left   = quad[0] -> quad[3]
    //   right  = quad[1] -> quad[2]
    auto dist = [](const PointF& a, const PointF& b) {
      float dx = a.x - b.x, dy = a.y - b.y;
      return std::sqrt(dx * dx + dy * dy);
    };
    float wTop    = dist(quad[0], quad[1]);
    float wBottom = dist(quad[2], quad[3]);
    float hLeft   = dist(quad[0], quad[3]);
    float hRight  = dist(quad[1], quad[2]);
    float maxW = std::max(wTop, wBottom);
    float maxH = std::max(hLeft, hRight);
    // paddlex 3.x `get_rotate_crop_image`:
    //   img_crop_width  = int(max(norm(p0-p1), norm(p2-p3)))
    //   img_crop_height = int(max(norm(p0-p3), norm(p1-p2)))
    // `int()` in Python is truncate-toward-zero. We replicate that
    // here. (std::lround / std::round was the M2-PIPE choice and
    // disagrees with paddlex for non-integer edge lengths.)
    int dst_w = std::max(1, static_cast<int>(maxW));
    int dst_h = std::max(1, static_cast<int>(maxH));
    Image warped;
    if (e.is_seal) {
      // M4-SEAL: ring text is curved; warp_perspective_quad (a tight
      // minarea-rect perspective crop) gives the rec model a slanted
      // strip of text that it mis-reads (e.g. "北强中列牙制" instead
      // of the real ring). Polar-unwarp the ring instead. Strategy:
      //   - seal center  ~ image center (seals in this dataset are
      //                   centered, and the per-poly centroid is OFF
      //                   the seal because the det returns the
      //                   partial arc, not the full ring).
      //   - seal radius  ~ bbox short-side / 2 (the poly traces the
      //                   outer ring of the seal).
      //   - unwrap to (rec.h, dst_w) where dst_w = min(2*pi*r, rec.w).
      // Bilinear sample, BORDER_CONSTANT=0 (matches the cpp_infer
      // fallback for the Plan-B `get_rotate_crop_image` we don't have).
      const float img_cx = bgr.w * 0.5f;
      const float img_cy = bgr.h * 0.5f;
      float bbox_xmin = std::min(quad[0].x, std::min(quad[1].x, std::min(quad[2].x, quad[3].x)));
      float bbox_xmax = std::max(quad[0].x, std::max(quad[1].x, std::max(quad[2].x, quad[3].x)));
      float bbox_ymin = std::min(quad[0].y, std::min(quad[1].y, std::min(quad[2].y, quad[3].y)));
      float bbox_ymax = std::max(quad[0].y, std::max(quad[1].y, std::max(quad[2].y, quad[3].y)));
      float bbox_w = bbox_xmax - bbox_xmin;
      float bbox_h = bbox_ymax - bbox_ymin;
      // M4-SEAL2: adaptive band from min-area-rect geometry. The
      // rect center sits at distance d_rect from the image center;
      // its short side h_rect spans radially across the ring at the
      // arc's apex, so the ring's OUTER edge is d_rect + h_rect/2
      // (measured 212-216 vs true ~220 on zh/en seals, while vertex
      // radii overshoot to 290 because of rect rotation). The band
      // is [outer - band_w, outer] with band_w = 0.35*outer
      // (>= 30 px), covering the glyph ring for thin en/ru rings and
      // thick zh rings alike.
      const float rect_cx = 0.25f * (quad[0].x + quad[1].x + quad[2].x + quad[3].x);
      const float rect_cy = 0.25f * (quad[0].y + quad[1].y + quad[2].y + quad[3].y);
      const float d_rect = std::sqrt((rect_cx - img_cx) * (rect_cx - img_cx) +
                                     (rect_cy - img_cy) * (rect_cy - img_cy));
      const float h_rect = std::max(1.0f, std::min(maxW, maxH));
      const float r_outer_raw = d_rect + 0.5f * h_rect;
      float band_w = 0.35f * r_outer_raw;
      if (band_w < 30.0f) band_w = 30.0f;
      if (band_w > r_outer_raw) band_w = r_outer_raw;
      const float r_inner = std::max(1.0f, r_outer_raw - band_w);
      const float r_outer = r_outer_raw;
      const float r_mid = 0.5f * (r_inner + r_outer);
      (void)bbox_w; (void)bbox_h;
      const float arc_len = 6.283185307f * r_mid;
      const int radial_n = std::max(1, H);
      const int angular_n = std::max(1, std::min(batch_w_cap,
                                                  static_cast<int>(arc_len)));

      Image unwrapped = polar_unwrap_band(bgr, img_cx, img_cy, r_inner, r_outer,
                                            angular_n, radial_n);
      // Transpose + horizontal flip so the rec sees H=48 (radial)
      // and W=angular_n with text in natural reading order (LTR).
      // M4-SEAL sampling: theta from 0 to 2pi going CW in image
      // coords from the seal's right side; Chinese seals read in
      // the OPPOSITE direction (CCW from right in image), so the
      // unwrap output is in reverse reading order. The horizontal
      // flip puts it in natural order, and the rec is then
      // post-processed below to reverse the codepoints.
      warped.w = unwrapped.h;
      warped.h = unwrapped.w;
      warped.c = unwrapped.c;
      warped.data.assign(static_cast<size_t>(warped.w) * warped.h * warped.c, 0);
      for (int y = 0; y < warped.h; ++y) {
        for (int x = 0; x < warped.w; ++x) {
          for (int c = 0; c < warped.c; ++c) {
            // Flip horizontally so text reads in natural order.
            warped.data[(y * warped.w + (warped.w - 1 - x)) * warped.c + c] =
                unwrapped.data[(x * unwrapped.w + y) * unwrapped.c + c];
          }
        }
      }

    } else {
      warped = warp_perspective_quad(bgr, quad, dst_w, dst_h);
    }
    if (warped.data.empty() || warped.w <= 0 || warped.h <= 0) {
      return;  // leave crops[bi].img empty, as the old push_back({Image{}}) did
    }
    // Rotate 90° if H/W >= 1.5, matching paddlex crop_image_regions.py:
    //   if dst_img_height * 1.0 / dst_img_width >= 1.5: dst_img = np.rot90(dst_img)
    // numpy rot90 (CCW) mapping, verified against np.rot90 on real crops:
    //   dst[i][j] = src[j][old_w - 1 - i]   (== flipud(src.T))
    // The previous code wrote dst[x][old_h-1-y] = src[y][x] which is the CW
    // transpose — vertical text (ja/ko columns) came out 90°-mirrored.
    if (static_cast<float>(warped.h) / static_cast<float>(warped.w)
        >= 1.5f) {
      Image rot;
      rot.w = warped.h;
      rot.h = warped.w;
      rot.c = warped.c;
      rot.data.assign(static_cast<size_t>(rot.w) * rot.h * rot.c, 0);
      const int old_w = warped.w;
      for (int i = 0; i < rot.h; ++i) {        // dst row i
        for (int j = 0; j < rot.w; ++j) {      // dst col j
          const uint8_t* sp = warped.data.data() +
              (static_cast<size_t>(j) * old_w + (old_w - 1 - i)) * warped.c;
          uint8_t* dp = rot.data.data() +
              (static_cast<size_t>(i) * rot.w + j) * rot.c;
          for (int c = 0; c < warped.c; ++c) dp[c] = sp[c];
        }
      }
      warped = std::move(rot);
    }
    crops[bi].img = std::move(warped);
  };
  {
    const int nbox = static_cast<int>(boxes.size());
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    int nthr = hw > 0 ? hw : 4;
    nthr = std::min(nthr, nbox > 0 ? nbox : 1);
    nthr = std::min(nthr, 8);
    if (nthr <= 1) {
      for (int bi = 0; bi < nbox; ++bi) crop_one(static_cast<size_t>(bi));
    } else {
      std::vector<std::thread> pool;
      pool.reserve(nthr);
      const int chunk = (nbox + nthr - 1) / nthr;
      for (int t = 0; t < nthr; ++t) {
        const int lo = t * chunk;
        const int hi = std::min(nbox, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&, lo, hi] {
          for (int bi = lo; bi < hi; ++bi) crop_one(static_cast<size_t>(bi));
        });
      }
      for (auto& th : pool) th.join();
    }
  }
  auto t_crop_end = std::chrono::steady_clock::now();
  // ---- M3 cls step ----
  // Run the textline orientation classifier on each warped crop and
  // rotate 180° any that score 180°. The rec step below then sees
  // upright crops. PaddleOCR's pipeline does the same thing between
  // TextClassifier and TextRecognizer (PaddleClas::TextRecRotator +
  // ppocr::predict_cls).
  //
  // We do the cls as a per-image loop (n <= ~50 in the rec-8 batch
  // regime) rather than a true batched forward. The cls MNN is
  // [1,3,80,160] fixed-shape and the per-call latency is ~1 ms on
  // this CPU; the loop's overhead is negligible. Doing it per-image
  // also lets us short-circuit any per-image error (a single bad
  // crop doesn't abort the run).
  if (e.cls) {
    std::vector<Image> cls_inputs;
    cls_inputs.reserve(crops.size());
    for (auto& c : crops) cls_inputs.push_back(c.img);
    std::vector<int> cls_labels;
    std::vector<float> cls_scores;
    run_cls_sync(e, cls_inputs, cls_labels, cls_scores, cls_ms_out);
    if (cls_labels_out) *cls_labels_out = std::move(cls_labels);
    for (size_t i = 0; i < crops.size(); ++i) {
      crops[i].img = std::move(cls_inputs[i]);
    }
  } else if (cls_labels_out) {
    cls_labels_out->assign(crops.size(), 0);
  }
  // ---- end cls step ----
  // Process crops in chunks of rec_batch. paddlex 3.x
  // `predictor.process` does this:
  //   1) batch_imgs = pre_tfs["ReisizeNorm"](imgs=batch_raw_imgs)
  //      where ReisizeNorm calls resize_norm_img(img, max_wh_ratio)
  //      with
  //        max_wh_ratio = max(rec_w / rec_h, w/h) per image
  //        imgW = int(imgH * max_wh_ratio)            (truncate)
  //        resized_w = min(ceil(imgH * w / h), imgW)   per image
  //      So every image in the batch is resized to the SAME width
  //      imgW = int(48 * max(rec_w/rec_h, max(wh_ratio in chunk))).
  //      The remaining columns are zero-padded.
  //   2) max_wh_ratio is computed from the FIRST `end_img_no =
  //      min(img_num, batch_num=8)` images (NOT the whole batch) but
  //      because the rec session is dynamic-width [1,3,48,-1] we can
  //      just compute it over the whole chunk and let MNN resize.
  //   3) max_imgW is 3200 (clamp the upper bound). The MNN rec
  //      session handles any width up to that ceiling.
  //   4) `rec_w / rec_h = 320/48 = 6.67` is a HARD FLOOR — a single
  //      tall image (w/h < 6.67) still gets `max_wh_ratio=6.67` so
  //      `imgW = 320` (the rec_image_shape baseline). Skipping this
  //      floor shrinks `imgW` and shifts the rec logits. (M2-NUM
  //      single-character regression on en/03 'S'->'K'.)
  // We reproduce this. The per-image "min(ceil, imgW)" capping that
  // paddlex applies is implicit in `prep_rec_line` which already
  // does `if (w > batch_w) w = batch_w;`.
  constexpr int kMaxImgW = 3200;
  for (size_t start = 0; start < crops.size(); start += e.rec_batch) {
    size_t end = std::min(crops.size(), start + e.rec_batch);
    size_t n = end - start;
    // paddlex `ReisizeNorm.resize_norm_img` per-image uses
    //   max_wh_ratio = max(rec_image_shape[2] / rec_image_shape[1], w/h)
    // so the *floor* is `rec_w / rec_h = 320/48 = 6.67`, never
    // less.  The first `batch_num = 8` imgs in the predictor drive
    // a single `max_wh_ratio` for the whole chunk (in our case
    // `e.rec_batch = 8`); we apply the same `max(rec_w/rec_h,
    // max(w_i/h_i over chunk))` rule so the per-image resized_w
    // is `min(int(ceil(H * w_i / h_i)), batch_w)` exactly as
    // paddlex does.  Without this floor, a single tall crop
    // (e.g. en/03 'S', 200x133, w/h=1.5) would feed the rec
    // model with batch_w = int(48 * 1.5) = 72 instead of 320
    // and the rec logits flip from "S" to "K" (single-character
    // mis-recognition caught by the M2-NUM diff against
    // paddle's rec dump).
    const double ratio_floor =
        static_cast<double>(batch_w_cap) / static_cast<double>(H);
    int batch_w = static_cast<int>(H * ratio_floor);
    for (size_t i = start; i < end; ++i) {
      if (crops[i].img.data.empty()) continue;
      double ratio = static_cast<double>(crops[i].img.w) /
                     static_cast<double>(crops[i].img.h);
      if (ratio < ratio_floor) ratio = ratio_floor;
      double bw_d = static_cast<double>(H) * ratio;
      int bw = static_cast<int>(bw_d);  // int() truncate, matches Python
      if (bw > batch_w) batch_w = bw;
    }
    if (batch_w > kMaxImgW) batch_w = kMaxImgW;
    if (batch_w > batch_w_cap) batch_w = batch_w_cap;
    if (batch_w < 1) batch_w = 1;
    // Build the CHW tensor: N * 3 * H * batch_w.
    auto t_rp = std::chrono::steady_clock::now();
    std::vector<float> chw(static_cast<size_t>(n) * 3 * H * batch_w, 0.f);
    {
      // M3-PERF2: per-crop prep is independent (resize + normalize +
      // CHW into its own slice of `chw`); fan out over threads.
      // Bit-exact: each crop writes the same bytes it wrote before.
      const int hw = static_cast<int>(std::thread::hardware_concurrency());
      int nthreads = hw > 0 ? hw : 4;
      nthreads = std::min(nthreads, static_cast<int>(n));
      nthreads = std::min(nthreads, 8);
      auto prep_worker = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
          if (crops[start + i].img.data.empty()) continue;
          int vw = 0;
          std::vector<float> line_chw =
              prep_rec_line(crops[start + i].img, H, batch_w, vw);
          std::memcpy(
              chw.data() + static_cast<size_t>(i) * 3 * H * batch_w,
              line_chw.data(),
              static_cast<size_t>(3) * H * batch_w * sizeof(float));
        }
      };
      if (nthreads <= 1) {
        prep_worker(0, static_cast<int>(n));
      } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        const int chunk = (static_cast<int>(n) + nthreads - 1) / nthreads;
        for (int t = 0; t < nthreads; ++t) {
          const int lo = t * chunk;
          const int hi = std::min(static_cast<int>(n), lo + chunk);
          if (lo >= hi) break;
          pool.emplace_back(prep_worker, lo, hi);
        }
        for (auto& th : pool) th.join();
      }
    }
    auto t_rp_end = std::chrono::steady_clock::now();
    // Resize + run rec.
    std::vector<int> dims = {static_cast<int>(n), 3, H, batch_w};
    e.rec->set_input_float("x", dims, chw.data());
    auto t_rr = std::chrono::steady_clock::now();
    int ec = e.rec->run();
    auto t_rr_end = std::chrono::steady_clock::now();
    ++n_rec_batches;
    if (ec != 0) continue;
    // Output is [N, T, C] = [n, T, num_classes] in NCHW with T
    // timesteps and C classes. (Verified on PP-OCRv6_tiny_rec after
    // MNNConvert; expect the same for v5/v4 mobile/server rec.)
    SessionOutput so = e.rec->output("fetch_name_0");
    if (so.data == nullptr || so.shape.size() != 3) {
      so = e.rec->output("softmax_0.tmp_0");
    }
    if (so.data == nullptr || so.shape.size() != 3) continue;
    const int N  = so.shape[0];
    const int T  = so.shape[1];
    const int C  = so.shape[2];
    if (N != static_cast<int>(n) || C <= 1 || T <= 0) continue;
    auto t_ctc = std::chrono::steady_clock::now();
    // M3-PERF2: decode each batch element's (T,C) logits independently;
    // rows are independent so a fan-out over std::thread is bit-exact.
    // v6_medium's 18k-class dict makes single-threaded ctc ~21% of
    // CUDA e2e; N/8-way fan-out cuts it to ~3%.
    std::vector<RecOut> outs(n);
    {
      const int hw = static_cast<int>(std::thread::hardware_concurrency());
      int nthreads = hw > 0 ? hw : 4;
      nthreads = std::min(nthreads, static_cast<int>(n));
      nthreads = std::min(nthreads, 8);
      auto worker = [&](int lo, int hi) {
        for (int i = lo; i < hi; ++i) {
          const float* row = so.data + static_cast<size_t>(i) * T * C;
          outs[static_cast<size_t>(i)] =
              ctc_decode(row, T, C, e.rec_cfg.rec);
        }
      };
      if (nthreads <= 1) {
        worker(0, static_cast<int>(n));
      } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        const int chunk = (static_cast<int>(n) + nthreads - 1) / nthreads;
        for (int t = 0; t < nthreads; ++t) {
          const int lo = t * chunk;
          const int hi = std::min(static_cast<int>(n), lo + chunk);
          if (lo >= hi) break;
          pool.emplace_back(worker, lo, hi);
        }
        for (auto& th : pool) th.join();
      }
    }
    for (size_t i = 0; i < n; ++i) {
      const RecOut& out = outs[i];
      // Note: ctc_decode returns the mean probability of the
      // emitted characters. We do NOT threshold text by rec score
      // here — the upstream `box_thresh` and `db_post` kMinSize
      // are the right place to drop spurious detections. A small
      // rec score is a useful signal for downstream consumers
      // (e.g. CER audits) and we keep the field intact.
      // M4-SEAL: polar_unwrap samples theta in CW order (in image
      // coords) from the seal's right side. Chinese seals read in
      // the OPPOSITE direction (CCW from right in image), so the
      // unwrap output is in reverse reading order. Reverse the
      // rec output as a sequence of UTF-8 codepoints.
      if (e.is_seal && !out.text.empty()) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(out.text.data());
        std::vector<std::pair<size_t, int>> cps;
        for (size_t k = 0; k < out.text.size(); ) {
          int n = 1;
          if      ((p[k] & 0x80) == 0)    n = 1;
          else if ((p[k] & 0xE0) == 0xC0) n = 2;
          else if ((p[k] & 0xF0) == 0xE0) n = 3;
          else if ((p[k] & 0xF8) == 0xF0) n = 4;
          else                            n = 1;
          cps.emplace_back(k, n);
          k += n;
        }
        std::string reversed;
        reversed.reserve(out.text.size());
        for (auto it = cps.rbegin(); it != cps.rend(); ++it) {
          reversed.append(out.text, it->first, it->second);
        }
        texts[start + i] = {reversed, out.score};
      } else {
        texts[start + i] = {out.text, out.score};
      }
    }
    t_ctc_end = std::chrono::steady_clock::now();
    if (prof) {
      e.last_profile.rec_prep_ms += std::chrono::duration<float, std::milli>(
          t_rp_end - t_rp).count();
      e.last_profile.rec_run_ms += std::chrono::duration<float, std::milli>(
          t_rr_end - t_rr).count();
      e.last_profile.ctc_decode_ms += std::chrono::duration<float, std::milli>(
          t_ctc_end - t_ctc).count();
    }
  }
  ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count());
  if (prof) {
    e.last_profile.crop_warp_ms = std::chrono::duration<float, std::milli>(
        t_crop_end - t_crop).count();
    e.last_profile.rec_batches = n_rec_batches;
  }
}

// Convert the (poly[8], text, score) internal representation to the
// flat C-ABI struct. Allocates text in `last_text` so the pointer
// remains valid until the next run() on this engine.
static void build_lines(Engine& e, const std::vector<DetBox>& boxes,
                        const std::vector<std::pair<std::string, float>>& texts) {
  e.last_text.clear();
  e.last_lines.clear();
  e.last_text.reserve(boxes.size());
  e.last_lines.reserve(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    e.last_text.emplace_back(texts[i].first);
    ppocr_line ln{};
    for (int k = 0; k < 8; ++k) ln.poly[k] = static_cast<int>(boxes[i].poly[k]);
    ln.text = e.last_text.back().c_str();
    ln.score = texts[i].second > 0 ? texts[i].second : boxes[i].score;
    e.last_lines.push_back(ln);
  }
  e.last_result.lines = e.last_lines.data();
  e.last_result.n_lines = static_cast<int>(e.last_lines.size());
}

// M2-ISO: variant of run_full that takes externally-supplied box
// polygons (the same [TL,TR,BR,BL] int poly[8] layout that the C ABI
// uses) and skips the det pass. Used for the M2-ISO error-isolation
// experiment: feed baseline det_polys straight into the rec crop/ctc
// path and report CER. If CER drops to ~0, the residual error in
// M2-REC2 (zh 0.13, en 0.24) lives in the det chain; if it stays
// high, the rec preprocess is still divergent.
static ppocr_status run_with_boxes(Engine& e, const uint8_t* bgr, int w, int h,
                                   const int* polys, int n_polys) {
  if (!bgr || w <= 0 || h <= 0) return PPOCR_ERR_PARAM;
  if (!polys || n_polys <= 0) return PPOCR_ERR_PARAM;
  if (!e.rec) return PPOCR_ERR_MODEL;

  Image img;
  img.w = w; img.h = h; img.c = 3;
  img.ext = bgr;  // M3-PERF6: non-owning view

  std::vector<DetBox> boxes;
  boxes.reserve(static_cast<size_t>(n_polys));
  for (int i = 0; i < n_polys; ++i) {
    DetBox b{};
    for (int k = 0; k < 8; ++k) b.poly[k] = polys[i * 8 + k];
    b.score = 1.0f;  // det score not part of the M2-ISO input
    boxes.push_back(b);
  }

  float rec_ms = 0.f;
  float cls_ms = 0.f;
  std::vector<std::pair<std::string, float>> texts;
  run_rec_sync(e, img, boxes, texts, rec_ms, cls_ms);
  build_lines(e, boxes, texts);

  e.last_result.det_ms = 0.f;
  e.last_result.rec_ms = rec_ms;
  e.last_result.cls_ms = cls_ms;
  e.last_result.total_ms = rec_ms + cls_ms;
  const char* bn = e.rec->backend_name();
  std::snprintf(e.last_result.backend_used, sizeof(e.last_result.backend_used),
                "%s", bn ? bn : "cpu");
  return PPOCR_OK;
}

// Run a full image: det → rec (M2 real pipeline). Populates e.last_result.
static ppocr_status run_full(Engine& e, const uint8_t* bgr, int w, int h) {
  if (!bgr || w <= 0 || h <= 0) return PPOCR_ERR_PARAM;
  if (!e.det) return PPOCR_ERR_MODEL;
  const auto t_e2e = std::chrono::steady_clock::now();
  if (e.want_profile) {
    // Reset per-run stage fields; decode_ms is set by ppocr_run_file
    // BEFORE this call and must survive (run_full itself receives an
    // already-decoded buffer so its own decode cost is 0). Keep the
    // incoming value; e2e/create/first_run filled below.
    e.last_profile.det_prep_ms = 0.f;
    e.last_profile.det_run_ms = 0.f;
    e.last_profile.db_post_ms = 0.f;
    e.last_profile.crop_warp_ms = 0.f;
    e.last_profile.rec_prep_ms = 0.f;
    e.last_profile.rec_run_ms = 0.f;
    e.last_profile.ctc_decode_ms = 0.f;
    e.last_profile.cls_ms = 0.f;
    e.last_profile.n_boxes = 0;
    e.last_profile.rec_batches = 0;
  }

  // M3-PERF6: zero-copy view over the caller's buffer. All pipeline
  // consumers of the input image are const readers (prep_det resize,
  // warp sampler); the only mutator (cls rotate_180) materializes an
  // owning copy first. This removes a full-image memcpy per run.
  Image img;
  img.w = w; img.h = h; img.c = 3;
  img.ext = bgr;

  std::vector<DetBox> boxes;
  float det_ms = 0.f, rec_ms = 0.f, cls_ms = 0.f;
  run_det_sync(e, img, boxes, det_ms);

  std::vector<std::pair<std::string, float>> texts;
  if (e.rec && !boxes.empty()) {
    // M2: real rec pipeline (crop + batch + ctc_decode). If a box
    // fails to crop, its text remains "" with score from the det box.
    // M3: cls is invoked between det and rec when cfg->cls_name is set.
    run_rec_sync(e, img, boxes, texts, rec_ms, cls_ms);
  } else {
    // det-only mode: every line gets score from the det box.
    texts.reserve(boxes.size());
    for (auto& b : boxes) texts.emplace_back(std::string{}, b.score);
    rec_ms = 0.f;
    cls_ms = 0.f;
  }
  build_lines(e, boxes, texts);

  e.last_result.det_ms = det_ms;
  e.last_result.rec_ms = rec_ms;
  e.last_result.cls_ms = cls_ms;
  e.last_result.total_ms = det_ms + rec_ms + cls_ms;
  const char* bn = e.det->backend_name();
  std::snprintf(e.last_result.backend_used, sizeof(e.last_result.backend_used),
                "%s", bn ? bn : "cpu");
  if (e.want_profile) {
    e.last_profile.cls_ms = cls_ms;
    e.last_profile.e2e_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_e2e).count();
    e.last_profile.create_ms = e.create_ms_stored;
    if (!e.has_run_once) {
      e.last_profile.first_run_ms = e.last_profile.e2e_ms;
    }
    e.has_run_once = true;
    std::snprintf(e.last_profile.backend, sizeof(e.last_profile.backend),
                  "%s", bn ? bn : "cpu");
    e.last_profile.threads = e.cfg_shadow.num_threads;
  }
  return PPOCR_OK;
}

// ---- C ABI ---------------------------------------------------------------

extern "C" PPOCR_API ppocr_status ppocr_version(int* major, int* minor,
                                                int* patch) {
  if (major) *major = PPOCR_VERSION_MAJOR;
  if (minor) *minor = PPOCR_VERSION_MINOR;
  if (patch) *patch = PPOCR_VERSION_PATCH;
  return PPOCR_OK;
}

extern "C" PPOCR_API ppocr_status ppocr_create(const ppocr_config* cfg,
                                               ppocr_engine** out,
                                               char* err_buf,
                                               size_t err_buf_len) {
  if (!cfg || !out) return PPOCR_ERR_PARAM;
  *out = nullptr;
  const auto t_create = std::chrono::steady_clock::now();
  std::unique_ptr<Engine> e = std::make_unique<Engine>();
  e->want_profile = (cfg->profile != 0);
  e->model_dir = trim_back_slash(
      cfg->model_dir ? std::string(cfg->model_dir)
                     : env_or("PPORC_MNN_MODELS", "./models"));
  e->cfg_shadow = *cfg;
  // Duplicate any cfg strings so the engine owns its own copies.
  if (cfg->det_name)       e->cfg_shadow.det_name = strdup(cfg->det_name);
  if (cfg->rec_name)       e->cfg_shadow.rec_name = strdup(cfg->rec_name);
  if (cfg->cls_name)       e->cfg_shadow.cls_name = strdup(cfg->cls_name);
  if (cfg->registry_path)  e->cfg_shadow.registry_path = strdup(cfg->registry_path);
  if (cfg->mirror)         e->cfg_shadow.mirror = strdup(cfg->mirror);
  if (cfg->model_dir)      e->cfg_shadow.model_dir = strdup(cfg->model_dir);
  if (cfg->cache_dir)      e->cfg_shadow.cache_dir = strdup(cfg->cache_dir);

  ppocr_status st = e->load_submodels(&e->cfg_shadow, err_buf, err_buf_len);
  if (st != PPOCR_OK) {
    if (err_buf && err_buf_len) {
      // Best-effort error string: we don't always have a useful message.
    }
    return st;
  }
  // M3-PERF4: opt-in warmup (cfg->warmup == 1). The CLI passes 1 by
  // default; plain C users keep the zero-init cold-start behavior.
  if (e->cfg_shadow.warmup) {
    e->warmup();
  }
  if (e->want_profile) {
    e->create_ms_stored = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_create).count();
  }
  *out = reinterpret_cast<ppocr_engine*>(e.release());
  return PPOCR_OK;
}

extern "C" PPOCR_API const ppocr_profile* ppocr_last_profile(
    const ppocr_engine* pe) {
  if (!pe) return nullptr;
  const Engine* e = reinterpret_cast<const Engine*>(pe);
  if (!e->want_profile || !e->has_run_once) return nullptr;
  return &e->last_profile;
}

extern "C" PPOCR_API void ppocr_destroy(ppocr_engine* pe) {
  if (!pe) return;
  Engine* e = reinterpret_cast<Engine*>(pe);
  // Shut down async worker if any.
  e->async_stop.store(true);
  e->async_cv.notify_all();
  if (e->async_worker.joinable()) e->async_worker.join();
  // Free shadow string copies.
  if (e->cfg_shadow.det_name)      free(const_cast<char*>(e->cfg_shadow.det_name));
  if (e->cfg_shadow.rec_name)      free(const_cast<char*>(e->cfg_shadow.rec_name));
  if (e->cfg_shadow.cls_name)      free(const_cast<char*>(e->cfg_shadow.cls_name));
  if (e->cfg_shadow.registry_path) free(const_cast<char*>(e->cfg_shadow.registry_path));
  if (e->cfg_shadow.mirror)        free(const_cast<char*>(e->cfg_shadow.mirror));
  if (e->cfg_shadow.model_dir)     free(const_cast<char*>(e->cfg_shadow.model_dir));
  if (e->cfg_shadow.cache_dir)     free(const_cast<char*>(e->cfg_shadow.cache_dir));
  delete e;
}

extern "C" PPOCR_API ppocr_status ppocr_run(ppocr_engine* pe,
                                            const uint8_t* bgr, int w, int h,
                                            ppocr_result** out) {
  if (!pe || !out) return PPOCR_ERR_PARAM;
  Engine* e = reinterpret_cast<Engine*>(pe);
  ppocr_status st = run_full(*e, bgr, w, h);
  if (st == PPOCR_OK) *out = &e->last_result;
  return st;
}

extern "C" PPOCR_API ppocr_status ppocr_run_file(ppocr_engine* pe,
                                                 const char* path,
                                                 ppocr_result** out) {
  if (!pe || !path || !out) return PPOCR_ERR_PARAM;
  Engine* e = reinterpret_cast<Engine*>(pe);
  const auto t_dec = std::chrono::steady_clock::now();
  Image img = load_image(path);
  if (img.data.empty()) return PPOCR_ERR_IO;
  if (e->want_profile) {
    // Accumulate the decode cost into the profile. run_full resets the
    // stage fields AFTER we set decode_ms, so stash it and restore:
    // simplest is to store it now and let run_full preserve it.
    e->last_profile.decode_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t_dec).count();
  }
  return ppocr_run(pe, img.data.data(), img.w, img.h, out);
}

extern "C" PPOCR_API const char* ppocr_status_string(ppocr_status st) {
  switch (st) {
    case PPOCR_OK:           return "OK";
    case PPOCR_ERR_PARAM:    return "invalid argument";
    case PPOCR_ERR_MODEL:    return "model missing/corrupt/sha mismatch";
    case PPOCR_ERR_BACKEND:  return "requested backend unavailable";
    case PPOCR_ERR_DOWNLOAD: return "model download failed";
    case PPOCR_ERR_IO:       return "image/filesystem error";
    case PPOCR_ERR_OOM:      return "out of memory";
    case PPOCR_ERR_INTERNAL: return "internal error";
  }
  return "unknown";
}

// ---- async (lazy worker) -------------------------------------------------

static void async_worker_main(Engine* e) {
  while (true) {
    std::unique_lock<std::mutex> lk(e->async_mu);
    e->async_cv.wait(lk, [&] {
      return e->async_stop.load() || !e->async_queue.empty();
    });
    if (e->async_stop.load() && e->async_queue.empty()) return;
    AsyncJob job = std::move(e->async_queue.front());
    e->async_queue.pop_front();
    lk.unlock();
    // The async worker shares the engine's sessions with the sync
    // path. MNN's per-Session is single-threaded for runSession, so
    // the caller is responsible for not interleaving sync runs and
    // async runs on the same engine; we document this in ppocr.h.
    ppocr_status st = PPOCR_OK;
    std::string backend_used = "cpu";
    float det_ms = 0.f, rec_ms = 0.f, cls_ms = 0.f, total_ms = 0.f;
    std::vector<ppocr_line>  lines;
    std::vector<std::string> text_storage;
    if (!e->det) {
      st = PPOCR_ERR_MODEL;
    } else {
      Image img;
      img.w = job.w; img.h = job.h; img.c = 3;
      img.data = job.bgr;
      std::vector<DetBox> boxes;
      run_det_sync(*e, img, boxes, det_ms);
      std::vector<std::pair<std::string, float>> texts;
      if (e->rec && !boxes.empty()) {
        run_rec_sync(*e, img, boxes, texts, rec_ms, cls_ms);
      } else {
        texts.reserve(boxes.size());
        for (auto& b : boxes) texts.emplace_back(std::string{}, b.score);
      }
      total_ms = det_ms + rec_ms + cls_ms;
      text_storage.reserve(boxes.size());
      lines.reserve(boxes.size());
      for (size_t i = 0; i < boxes.size(); ++i) {
        text_storage.emplace_back(texts[i].first);
        ppocr_line ln{};
        for (int k = 0; k < 8; ++k) ln.poly[k] = static_cast<int>(boxes[i].poly[k]);
        ln.text = text_storage.back().c_str();
        ln.score = texts[i].second > 0 ? texts[i].second : boxes[i].score;
        lines.push_back(ln);
      }
      backend_used = e->det->backend_name();
    }
    ppocr_result r{};
    r.lines = lines.data();
    r.n_lines = static_cast<int>(lines.size());
    r.det_ms = det_ms;
    r.rec_ms = rec_ms;
    r.cls_ms = cls_ms;
    r.total_ms = total_ms;
    std::snprintf(r.backend_used, sizeof(r.backend_used), "%s",
                  backend_used.c_str());
    if (job.cb) job.cb(reinterpret_cast<ppocr_engine*>(e), st, &r, job.user);
    // The callback's ppocr_result references local buffers; callers
    // must copy out before returning. Documented in the header.
  }
}

extern "C" PPOCR_API ppocr_status ppocr_run_async(ppocr_engine* pe,
                                                  const uint8_t* bgr,
                                                  int w, int h,
                                                  ppocr_callback cb,
                                                  void* user) {
  if (!pe || !bgr || w <= 0 || h <= 0 || !cb) return PPOCR_ERR_PARAM;
  Engine* e = reinterpret_cast<Engine*>(pe);
  // Lazy-start the worker the first time async is used.
  if (!e->async_started.exchange(true)) {
    e->async_worker = std::thread(async_worker_main, e);
  }
  AsyncJob job;
  job.cb = cb;
  job.user = user;
  job.bgr.assign(bgr, bgr + static_cast<size_t>(w) * h * 3);
  job.w = w;
  job.h = h;
  {
    std::lock_guard<std::mutex> lk(e->async_mu);
    e->async_queue.push_back(std::move(job));
  }
  e->async_cv.notify_one();
  return PPOCR_OK;
}

extern "C" PPOCR_API ppocr_status ppocr_run_with_boxes(ppocr_engine* pe,
                                                      const uint8_t* bgr,
                                                      int w, int h,
                                                      const int* polys,
                                                      int n_polys,
                                                      ppocr_result** out) {
  if (!pe || !bgr || w <= 0 || h <= 0 || !polys || n_polys <= 0) {
    return PPOCR_ERR_PARAM;
  }
  Engine* e = reinterpret_cast<Engine*>(pe);
  ppocr_status st = run_with_boxes(*e, bgr, w, h, polys, n_polys);
  if (st == PPOCR_OK && out) *out = &e->last_result;
  return st;
}

} // namespace ppocr
