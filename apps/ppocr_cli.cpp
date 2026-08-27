// pp-ocr-mnn — command-line entry point
//
// Implements the CLI contract from CONTRACT.md:
//   ppocr_cli --image IMG --det-config X.json --rec-config Y.json
//              [--cls-config cls.json] [--backend ...] [--threads N]
//              [--batch N] [--det-only] [--json OUT] [--time]
//
// stdout (or OUT) is a single JSON object. The CLI does not own any
// long-lived state: it creates the engine, runs the image once, and
// destroys the engine. Errors are written to stderr with a non-zero
// exit code.
#include "ppocr/ppocr.h"
#include "ppocr/config.h"
#include "ppocr/image.h"
#include "ppocr/profile.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
  std::string image;
  std::string det_config;        // path to det .mnn or config
  std::string rec_config;
  std::string cls_config;
  std::string model_dir = "./models";
  std::string registry_path;     // optional
  std::string backend = "auto";
  std::string boxes_json;        // M2-ISO: skip det, use boxes from this file
  int threads = 0;
  int batch   = 0;
  int det_only = 0;
  int time    = 0;
  int profile = 0;               // M3-PERF1: per-stage timings in JSON
  std::string out_path;          // empty -> stdout
  int max_side = 0;
  int help    = 0;
  // M3-PERF3 batch mode
  std::string batch_dir;         // --batch-dir DIR: process all images in DIR
  int workers = 2;               // --workers N: engine-per-worker (default 2)
  int warmup = 1;                // M3-PERF4: dummy inference at create
  int pin_offset = 0;            // M3-PERF6: cores per worker for --batch-dir
};

void usage() {
  std::fprintf(stderr,
    "ppocr_cli --image IMG --det-config X [--rec-config Y] [--cls-config Z]\n"
    "          [--model-dir DIR] [--backend auto|cpu|cuda|opencl|vulkan]\n"
    "          [--threads N] [--batch N] [--det-only] [--json OUT] [--time]\n"
    "          [--profile]  # M3-PERF1: add per-stage ms to the JSON output\n"
    "          [--batch-dir DIR] [--workers N]  # M3-PERF3: engine-per-worker batch mode\n"
    "          [--pin-offset N]  # M3-PERF6: pin each worker to N CPUs (taskset-style)\n");
}

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); return nullptr; }
      return argv[++i];
    };
    if      (k == "--image")        { auto v = need("--image");        if (!v) return false; a.image = v; }
    else if (k == "--det-config")    { auto v = need("--det-config");    if (!v) return false; a.det_config = v; }
    else if (k == "--rec-config")    { auto v = need("--rec-config");    if (!v) return false; a.rec_config = v; }
    else if (k == "--cls-config")    { auto v = need("--cls-config");    if (!v) return false; a.cls_config = v; }
    else if (k == "--model-dir")     { auto v = need("--model-dir");     if (!v) return false; a.model_dir = v; }
    else if (k == "--registry")      { auto v = need("--registry");      if (!v) return false; a.registry_path = v; }
    else if (k == "--backend")       { auto v = need("--backend");       if (!v) return false; a.backend = v; }
    else if (k == "--threads")       { auto v = need("--threads");       if (!v) return false; a.threads = std::atoi(v); }
    else if (k == "--batch")         { auto v = need("--batch");         if (!v) return false; a.batch = std::atoi(v); }
    else if (k == "--det-only")      { a.det_only = 1; }
    else if (k == "--max-side")      { auto v = need("--max-side");      if (!v) return false; a.max_side = std::atoi(v); }
    else if (k == "--json")          { auto v = need("--json");          if (!v) return false; a.out_path = v; }
    else if (k == "--boxes-json")    { auto v = need("--boxes-json");    if (!v) return false; a.boxes_json = v; }
    else if (k == "--time")          { a.time = 1; }
  else if (k == "--profile")       { a.profile = 1; }
  else if (k == "--batch-dir")     { auto v = need("--batch-dir");     if (!v) return false; a.batch_dir = v; }
  else if (k == "--workers")       { auto v = need("--workers");       if (!v) return false; a.workers = std::atoi(v); }
  else if (k == "--no-warmup")     { a.warmup = 0; }
    else if (k == "--pin-offset")   { auto v = need("--pin-offset");   if (!v) return false; a.pin_offset = std::atoi(v); }
    else if (k == "-h" || k == "--help") { a.help = 1; return true; }
    else {
      std::fprintf(stderr, "unknown flag: %s\n", k.c_str());
      return false;
    }
  }
  return true;
}

ppocr_backend parse_backend(const std::string& s) {
  if (s == "cpu")    return PPOCR_BACKEND_CPU;
  if (s == "cuda")   return PPOCR_BACKEND_CUDA;
  if (s == "opencl") return PPOCR_BACKEND_OPENCL;
  if (s == "vulkan") return PPOCR_BACKEND_VULKAN;
  if (s == "metal")  return PPOCR_BACKEND_METAL;
  if (s == "coreml") return PPOCR_BACKEND_COREML;
  if (s == "nnapi")  return PPOCR_BACKEND_NNAPI;
  return PPOCR_BACKEND_AUTO;
}

// Derive the model name from a config path. The config file lives at
// <model_dir>/configs/<name>.json, so we strip the directory and the
// extension. The public C ABI then loads <model_dir>/<name>.mnn.
std::string config_basename(const std::string& p) {
  auto slash = p.find_last_of("/\\");
  std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
  auto dot = base.find_last_of('.');
  if (dot != std::string::npos) base = base.substr(0, dot);
  return base;
}

void write_result(FILE* f, const char* image, const ppocr_result* r,
                  const ppocr_profile* prof = nullptr) {
  // M3-PERF6: assemble the JSON into one string and fwrite once.
  // Byte-identical to the previous per-token fprintf/fputc version
  // (same formats, same order) but avoids thousands of locked stdio
  // calls on dense outputs (283 lines x per-char fputc).
  std::string buf;
  buf.reserve(256 + static_cast<size_t>(r->n_lines) * 96);
  char num[64];
  buf += "{\"image\":\"";
  buf += image ? image : "";
  buf += "\",\"backend\":\"";
  buf += r->backend_used;
  buf += "\",\"lines\":[";
  for (int i = 0; i < r->n_lines; ++i) {
    const ppocr_line& ln = r->lines[i];
    if (i) buf += ',';
    buf += "{\"poly\":[";
    for (int k = 0; k < 8; ++k) {
      if (k) buf += ',';
      std::snprintf(num, sizeof(num), "%d", ln.poly[k]);
      buf += num;
    }
    buf += "],\"text\":\"";
    // Escape JSON string minimally: " and \.
    for (const char* p = ln.text ? ln.text : ""; *p; ++p) {
      if (*p == '"' || *p == '\\') buf += '\\';
      buf += *p;
    }
    std::snprintf(num, sizeof(num), "\",\"score\":%.4f}", ln.score);
    buf += num;
  }
  buf += "],";
  std::snprintf(num, sizeof(num),
                "\"ms\":{\"det\":%.2f,\"rec\":%.2f,\"cls\":%.2f,\"total\":%.2f}",
                r->det_ms, r->rec_ms, r->cls_ms, r->total_ms);
  buf += num;
  if (prof) {
    // Profile JSON grows past `num`, so format through a bigger scratch.
    char pbuf[1024];
    std::snprintf(pbuf, sizeof(pbuf),
        ",\"profile\":{"
        "\"decode_ms\":%.3f,\"det_prep_ms\":%.3f,\"det_run_ms\":%.3f,"
        "\"db_post_ms\":%.3f,\"crop_warp_ms\":%.3f,\"rec_prep_ms\":%.3f,"
        "\"rec_run_ms\":%.3f,\"ctc_decode_ms\":%.3f,\"cls_ms\":%.3f,"
        "\"e2e_ms\":%.3f,\"create_ms\":%.3f,\"first_run_ms\":%.3f,"
        "\"n_boxes\":%d,\"rec_batches\":%d,\"threads\":%d,"
        "\"backend\":\"%s\"}",
        prof->decode_ms, prof->det_prep_ms, prof->det_run_ms,
        prof->db_post_ms, prof->crop_warp_ms, prof->rec_prep_ms,
        prof->rec_run_ms, prof->ctc_decode_ms, prof->cls_ms,
        prof->e2e_ms, prof->create_ms, prof->first_run_ms,
        prof->n_boxes, prof->rec_batches, prof->threads, prof->backend);
    buf += pbuf;
    buf += '}';   // close the profile object AND the outer result
  } else {
    buf += '}';   // close the outer result (no profile)
  }
  buf += '\n';
  std::fwrite(buf.data(), 1, buf.size(), f);
}

} // namespace

// ---- M3-PERF3: batch-dir mode ------------------------------------------
// Process every image in a directory through an engine-per-worker pool.
// Each worker owns an independent ppocr_engine (the C ABI is
// multi-instance thread-safe), pulls the next image off an atomic
// index, and runs the FULL serial pipeline per image — outputs are
// therefore identical to the serial CLI, but the CPU-side stages of
// image N-1 overlap the GPU runs of image N across workers.
// Results are written as a JSON array to --json (or stdout) plus a
// "_bench" object with per-image e2e ms and aggregate throughput.
namespace batchmode {

struct Job {
  std::string path;
  std::string name;
  std::string json;   // serialized result
  float e2e_ms = 0.f;
  int rc = 0;
};

bool has_image_ext(const std::string& n) {
  auto dot = n.find_last_of('.');
  if (dot == std::string::npos) return false;
  std::string ext = n.substr(dot);
  for (auto& ch : ext) ch = static_cast<char>(std::tolower(ch));
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

int run_batch(const Args& a) {
  namespace fs = std::filesystem;
  std::vector<std::string> paths;
  try {
    for (const auto& e : fs::directory_iterator(a.batch_dir)) {
      if (!e.is_regular_file()) continue;
      if (!has_image_ext(e.path().filename().string())) continue;
      paths.push_back(e.path().string());
    }
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "batch-dir: %s\n", ex.what());
    return 2;
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    std::fprintf(stderr, "batch-dir: no images in %s\n", a.batch_dir.c_str());
    return 2;
  }

  const std::string det_name = config_basename(a.det_config);
  const std::string rec_name = (a.det_only || a.rec_config.empty())
                                   ? std::string{}
                                   : config_basename(a.rec_config);
  const std::string cls_name = a.cls_config.empty()
                                   ? std::string{}
                                   : config_basename(a.cls_config);

  auto make_engine = [&]() -> ppocr_engine* {
    ppocr_config cfg{};
    cfg.model_dir = a.model_dir.c_str();
    cfg.det_name  = det_name.empty() ? nullptr : det_name.c_str();
    cfg.rec_name  = rec_name.empty() ? nullptr : rec_name.c_str();
    cfg.cls_name  = cls_name.empty() ? nullptr : cls_name.c_str();
    cfg.registry_path = a.registry_path.empty() ? nullptr : a.registry_path.c_str();
    cfg.mirror     = nullptr;
    cfg.backend    = parse_backend(a.backend);
    cfg.num_threads = a.threads;
    cfg.rec_batch  = a.batch;
  cfg.warmup     = a.warmup;
    cfg.max_side   = a.max_side;
    cfg.offline    = 1;
    cfg.download   = 0;
    cfg.profile    = a.profile;
    char err[256] = {0};
    ppocr_engine* eng = nullptr;
    ppocr_status st = ppocr_create(&cfg, &eng, err, sizeof(err));
    if (st != PPOCR_OK) {
      std::fprintf(stderr, "worker ppocr_create failed: %s (%s)\n",
                   ppocr_status_string(st), err);
      return nullptr;
    }
    return eng;
  };

  const int nw = std::max(1, a.workers);
  std::vector<Job> jobs(paths.size());
  for (size_t i = 0; i < paths.size(); ++i) {
    auto slash = paths[i].find_last_of("/\\");
    jobs[i].path = paths[i];
    jobs[i].name = (slash == std::string::npos) ? paths[i] : paths[i].substr(slash + 1);
  }
  std::atomic<size_t> next{0};
  std::atomic<int> failures{0};
  auto worker_main = [&](int wid) {
    // M3-PERF6: --pin-offset N — pin this worker to N consecutive CPUs
    // starting at (wid*N) % ncpu (taskset semantics, no exec). Keeps
    // batch workers on disjoint core sets. 0 = off (default).
    if (a.pin_offset > 0) {
      const int ncpu = static_cast<int>(std::thread::hardware_concurrency());
      if (ncpu > 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        const int base = (wid * a.pin_offset) % ncpu;
        for (int i = 0; i < a.pin_offset; ++i) {
          CPU_SET((base + i) % ncpu, &set);
        }
        if (sched_setaffinity(0, sizeof(set), &set) != 0) {
          // non-fatal: best-effort pinning
        }
      }
    }
    ppocr_engine* eng = make_engine();
    if (!eng) { failures.fetch_add(1); return; }
    // Per-worker string scratch: write_result needs FILE*; use open_memstream.
    while (true) {
      const size_t i = next.fetch_add(1);
      if (i >= jobs.size()) break;
      auto t0 = std::chrono::steady_clock::now();
      ppocr_result* result = nullptr;
      ppocr_status st = ppocr_run_file(eng, jobs[i].path.c_str(), &result);
      auto t1 = std::chrono::steady_clock::now();
      if (st != PPOCR_OK) {
        jobs[i].rc = 3;
        failures.fetch_add(1);
        continue;
      }
      char* buf = nullptr; size_t buflen = 0;
      FILE* ms = open_memstream(&buf, &buflen);
      if (ms) {
        write_result(ms, jobs[i].path.c_str(), result,
                     a.profile ? ppocr_last_profile(eng) : nullptr);
        std::fclose(ms);
        jobs[i].json = std::string(buf, buflen);
        std::free(buf);
      }
      jobs[i].e2e_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
      jobs[i].rc = 0;
    }
    ppocr_destroy(eng);
  };

  const auto t_start = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int w = 0; w < nw; ++w) pool.emplace_back(worker_main, w);
  for (auto& th : pool) th.join();
  const auto t_end = std::chrono::steady_clock::now();
  const double wall_s = std::chrono::duration<double>(t_end - t_start).count();

  FILE* out = stdout;
  std::unique_ptr<FILE, int(*)(FILE*)> owned{nullptr, [](FILE*){return 0;}};
  if (!a.out_path.empty()) {
    FILE* fp = std::fopen(a.out_path.c_str(), "wb");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", a.out_path.c_str()); return 4; }
    owned.reset(fp); out = fp;
  }
  std::fprintf(out, "[");
  for (size_t i = 0; i < jobs.size(); ++i) {
    if (i) std::fputc(',', out);
    if (jobs[i].rc == 0 && !jobs[i].json.empty()) {
      // strip the trailing newline write_result appends
      std::string j = jobs[i].json;
      while (!j.empty() && (j.back() == '\n' || j.back() == '\r')) j.pop_back();
      std::fwrite(j.data(), 1, j.size(), out);
    } else {
      std::fprintf(out, "{\"image\":\"%s\",\"error\":true}", jobs[i].path.c_str());
    }
  }
  double sum_e2e = 0; double max_e2e = 0;
  for (auto& j : jobs) { sum_e2e += j.e2e_ms; if (j.e2e_ms > max_e2e) max_e2e = j.e2e_ms; }
  std::fprintf(out, ",{\"_bench\":{\"n_images\":%zu,\"workers\":%d,"
      "\"wall_s\":%.3f,\"throughput_fps\":%.2f,\"mean_e2e_ms\":%.1f,"
      "\"max_e2e_ms\":%.1f,\"failures\":%d}}]\n",
      jobs.size(), nw, wall_s,
      static_cast<double>(jobs.size()) / wall_s,
      sum_e2e / static_cast<double>(jobs.size() ? jobs.size() : 1),
      max_e2e, failures.load());
  if (owned) std::fclose(owned.release());
  return failures.load() ? 5 : 0;
}

}  // namespace batchmode

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a) || a.help) { usage(); return a.help ? 0 : 1; }
  if (!a.batch_dir.empty()) {
    if (a.det_config.empty()) {
      std::fprintf(stderr, "error: --batch-dir requires --det-config\n");
      return 1;
    }
    return batchmode::run_batch(a);
  }
  if (a.image.empty() || a.det_config.empty()) {
    std::fprintf(stderr, "error: --image and --det-config are required\n");
    usage();
    return 1;
  }

  // Pre-compute basenames into owning std::strings so the c_str() views
  // stay valid for the duration of ppocr_create (which copies them).
  const std::string det_name = config_basename(a.det_config);
  const std::string rec_name = (a.det_only || a.rec_config.empty())
                                   ? std::string{}
                                   : config_basename(a.rec_config);
  const std::string cls_name = a.cls_config.empty()
                                   ? std::string{}
                                   : config_basename(a.cls_config);

  ppocr_config cfg{};
  cfg.model_dir = a.model_dir.c_str();
  cfg.det_name  = det_name.empty() ? nullptr : det_name.c_str();
  cfg.rec_name  = rec_name.empty() ? nullptr : rec_name.c_str();
  cfg.cls_name  = cls_name.empty() ? nullptr : cls_name.c_str();
  cfg.registry_path = a.registry_path.empty() ? nullptr : a.registry_path.c_str();
  cfg.mirror     = nullptr;
  cfg.backend    = parse_backend(a.backend);
  cfg.num_threads = a.threads;
  cfg.rec_batch  = a.batch;
  cfg.warmup     = a.warmup;
  cfg.max_side   = a.max_side;
  cfg.offline    = 1; // M1: never download; if file is missing we want a hard error
  cfg.download   = 0;
  cfg.profile    = a.profile;

  char err[256] = {0};
  ppocr_engine* engine = nullptr;
  ppocr_status st = ppocr_create(&cfg, &engine, err, sizeof(err));
  if (st != PPOCR_OK) {
    std::fprintf(stderr, "ppocr_create failed: %s (%s)\n",
                 ppocr_status_string(st), err);
    return 2;
  }

  ppocr_result* result = nullptr;
  if (!a.boxes_json.empty()) {
    // M2-ISO: load the image, parse boxes from JSON, call
    // ppocr_run_with_boxes. The JSON is a flat array of 8 ints per
    // box (TL,TR,BR,BL), in the same order the C ABI uses. We do a
    // tiny hand-rolled parser; the boxes file is produced by the
    // M2-ISO script (extract from /root/ppocr_reference/...json
    // field "det_polys" -> [N][8] flat array).
    if (a.rec_config.empty()) {
      std::fprintf(stderr, "error: --boxes-json requires --rec-config\n");
      ppocr_destroy(engine);
      return 2;
    }
    std::ifstream bj(a.boxes_json);
    if (!bj) {
      std::fprintf(stderr, "cannot open %s\n", a.boxes_json.c_str());
      ppocr_destroy(engine);
      return 4;
    }
    std::string buf((std::istreambuf_iterator<char>(bj)),
                    std::istreambuf_iterator<char>());
    std::vector<int> polys;
    int cur = 0; int sign = 1; bool in_num = false;
    for (char c : buf) {
      if (c == '-') { sign = -1; continue; }
      if (c >= '0' && c <= '9') {
        cur = cur * 10 + (c - '0'); in_num = true; continue;
      }
      if (in_num) {
        polys.push_back(sign * cur);
        cur = 0; sign = 1; in_num = false;
      }
    }
    if (in_num) polys.push_back(sign * cur);
    if (polys.size() % 8 != 0) {
      std::fprintf(stderr, "boxes-json: %d ints, not a multiple of 8\n",
                   (int)polys.size());
      ppocr_destroy(engine);
      return 5;
    }
    int n_polys = static_cast<int>(polys.size() / 8);
    std::fprintf(stderr, "boxes-json: %d polys loaded\n", n_polys);
    // Load the image through the public image helper, then call
    // ppocr_run_with_boxes. The BGR buffer is owned by the Image
    // struct; ppocr_run_with_boxes only reads it.
    ppocr::Image pim = ppocr::load_image(a.image);
    if (pim.data.empty() || pim.w <= 0 || pim.h <= 0) {
      std::fprintf(stderr, "load_image failed for %s\n", a.image.c_str());
      ppocr_destroy(engine);
      return 4;
    }
    st = ppocr_run_with_boxes(engine, pim.data.data(), pim.w, pim.h,
                              polys.data(), n_polys, &result);
  } else {
    st = ppocr_run_file(engine, a.image.c_str(), &result);
  }
  if (st != PPOCR_OK) {
    std::fprintf(stderr, "ppocr_run%s failed: %s\n",
                 a.boxes_json.empty() ? "_file" : "_with_boxes",
                 ppocr_status_string(st));
    ppocr_destroy(engine);
    return 3;
  }

  FILE* out = stdout;
  std::unique_ptr<FILE, int(*)(FILE*)> owned{nullptr, [](FILE*){return 0;}};
  if (!a.out_path.empty()) {
    FILE* fp = std::fopen(a.out_path.c_str(), "wb");
    if (!fp) {
      std::fprintf(stderr, "cannot open %s for write\n", a.out_path.c_str());
      ppocr_destroy(engine);
      return 4;
    }
    owned.reset(fp);
    out = fp;
  }
  write_result(out, a.image.c_str(), result,
               a.profile ? ppocr_last_profile(engine) : nullptr);
  if (a.time) {
    std::fprintf(stderr, "det=%.2fms rec=%.2fms total=%.2fms n_lines=%d\n",
                 result->det_ms, result->rec_ms, result->total_ms,
                 result->n_lines);
  }
  if (owned) std::fclose(owned.release());
  ppocr_destroy(engine);
  return 0;
}
