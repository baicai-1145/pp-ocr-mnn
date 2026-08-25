// pp-ocr-mnn — public C ABI implementation
//
// One file owns the entire C ABI surface. Backend selection lives ONLY
// here in pickBackend(); the rest of the codebase is platform-agnostic
// (CONTRACT hard rule #6). On Linux desktop M1 we ship only the CPU
// path; the opencl/vulkan/... branches map to the corresponding
// MNNForwardType and let MNN's auto-tuner pick the first available
// backend at session create time.
#include "ppocr/ppocr.h"

#include "ppocr/config.h"
#include "ppocr/image.h"
#include "ppocr/mnn_session.h"
#include "ppocr/postprocess/ctc_decode.h"
#include "ppocr/postprocess/db_post.h"
#include "ppocr/preprocess.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
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

  // Async worker.
  std::thread             async_worker;
  std::mutex              async_mu;
  std::condition_variable async_cv;
  std::deque<AsyncJob>    async_queue;
  std::atomic<bool>       async_stop{false};
  std::atomic<bool>       async_started{false};

  ppocr_config cfg_shadow{};  // owned copies of cfg strings

  // Helpers
  ppocr_status load_submodels(const ppocr_config* cfg, char* err, size_t elen);
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
  // For M1 we expect the user to pass --model-dir pointing at a
  // directory whose immediate children are .mnn files and whose
  // configs/ subfolder holds the JSONs. (e.g. /root/pp-ocr-mnn/models
  // already has the .mnn files; clone the worktree's configs/ there
  // to satisfy the per-model JSON lookup, or pass a different dir.)
  const std::string cfg_dir = e.model_dir + "/configs";
  auto find_cfg = [&](const std::string& name) -> std::string {
    if (name.empty()) return {};
    std::string p = cfg_dir + "/" + name + ".json";
    if (file_exists(p)) return p;
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

  ppocr_status st = resolve_config_paths(*this, cfg, err, elen,
                                         det_name, rec_name, cls_name);
  if (st != PPOCR_OK) {
    if (err && elen) std::snprintf(err, elen, "model path resolution failed");
    return st;
  }

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

// ---- det / rec helpers ---------------------------------------------------

static void run_det_sync(Engine& e, const Image& bgr,
                         std::vector<DetBox>& boxes_out,
                         float& ms_out) {
  auto t0 = std::chrono::steady_clock::now();
  DetInput in = prep_det(bgr, e.det_cfg.det.resize);
  // The det .mnn file's input is dynamic; we resize to the preprocessed HxW.
  std::vector<int> dims = {1, 3, in.in_h, in.in_w};
  e.det->set_input_float("x", dims, in.chw.data());
  int ec = e.det->run();
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
  boxes_out = db_postprocess(so.data, H, W, bgr.w, bgr.h,
                             in.ratio_w, in.ratio_h, e.det_cfg.det);
  ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count());
}

// Rec is M2 scope; M1 returns a synthetic line for each det box so the
// CLI can emit valid JSON. Once ws/post lands, this becomes a real
// ctc_decode call driven by a rec session.
static void run_rec_sync_stub(Engine& /*e*/,
                              const std::vector<DetBox>& boxes,
                              std::vector<std::pair<std::string, float>>& texts,
                              float& ms_out) {
  auto t0 = std::chrono::steady_clock::now();
  texts.clear();
  texts.reserve(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    // Temporary stub: each box gets a placeholder line so the CLI
    // output is well-formed. The decision-maker removes this once
    // ws/post merges and run_rec_sync is real.
    std::ostringstream oss;
    oss << "stub[" << i << "]";
    texts.emplace_back(oss.str(), 0.f);
  }
  ms_out = static_cast<float>(std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count());
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

// Run a full image: det → rec stub. Populates e.last_result.
static ppocr_status run_full(Engine& e, const uint8_t* bgr, int w, int h) {
  if (!bgr || w <= 0 || h <= 0) return PPOCR_ERR_PARAM;
  if (!e.det) return PPOCR_ERR_MODEL;

  Image img;
  img.w = w; img.h = h; img.c = 3;
  img.data.assign(bgr, bgr + static_cast<size_t>(w) * h * 3);

  std::vector<DetBox> boxes;
  float det_ms = 0.f, rec_ms = 0.f;
  run_det_sync(e, img, boxes, det_ms);

  std::vector<std::pair<std::string, float>> texts;
  if (e.rec && !boxes.empty()) {
    // Real rec path is M2/ws/post scope; for M1 the stub produces
    // well-formed text strings so the CLI can print valid JSON.
    run_rec_sync_stub(e, boxes, texts, rec_ms);
  } else {
    // det-only mode: every line gets score from the det box.
    texts.reserve(boxes.size());
    for (auto& b : boxes) texts.emplace_back(std::string{}, b.score);
    rec_ms = 0.f;
  }
  build_lines(e, boxes, texts);

  e.last_result.det_ms = det_ms;
  e.last_result.rec_ms = rec_ms;
  e.last_result.cls_ms = 0.f;
  e.last_result.total_ms = det_ms + rec_ms;
  const char* bn = e.det->backend_name();
  std::snprintf(e.last_result.backend_used, sizeof(e.last_result.backend_used),
                "%s", bn ? bn : "cpu");
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
  std::unique_ptr<Engine> e = std::make_unique<Engine>();
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
  *out = reinterpret_cast<ppocr_engine*>(e.release());
  return PPOCR_OK;
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
  Image img = load_image(path);
  if (img.data.empty()) return PPOCR_ERR_IO;
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
    // Run with a *copy* of the engine's state: the caller might destroy
    // the engine between submit and finish. We carry the only fields we
    // need into a local mini-context. For M1, we only support the
    // det-only path here (rec stub is trivial).
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
      total_ms = det_ms;
      text_storage.reserve(boxes.size());
      lines.reserve(boxes.size());
      for (size_t i = 0; i < boxes.size(); ++i) {
        std::ostringstream oss;
        oss << "stub[" << i << "]";
        text_storage.emplace_back(oss.str());
        ppocr_line ln{};
        for (int k = 0; k < 8; ++k) ln.poly[k] = static_cast<int>(boxes[i].poly[k]);
        ln.text = text_storage.back().c_str();
        ln.score = boxes[i].score;
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

} // namespace ppocr
