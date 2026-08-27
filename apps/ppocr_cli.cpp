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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

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
};

void usage() {
  std::fprintf(stderr,
    "ppocr_cli --image IMG --det-config X [--rec-config Y] [--cls-config Z]\n"
    "          [--model-dir DIR] [--backend auto|cpu|cuda|opencl|vulkan]\n"
    "          [--threads N] [--batch N] [--det-only] [--json OUT] [--time]\n"
    "          [--profile]  # M3-PERF1: add per-stage ms to the JSON output\n");
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
  std::fprintf(f, "{\"image\":\"%s\",\"backend\":\"%s\",",
               image ? image : "", r->backend_used);
  std::fprintf(f, "\"lines\":[");
  for (int i = 0; i < r->n_lines; ++i) {
    const ppocr_line& ln = r->lines[i];
    if (i) std::fputc(',', f);
    std::fprintf(f, "{\"poly\":[");
    for (int k = 0; k < 8; ++k) {
      if (k) std::fputc(',', f);
      std::fprintf(f, "%d", ln.poly[k]);
    }
    std::fprintf(f, "],\"text\":\"");
    // Escape JSON string minimally: " and \.
    for (const char* p = ln.text ? ln.text : ""; *p; ++p) {
      if (*p == '"' || *p == '\\') std::fputc('\\', f);
      std::fputc(*p, f);
    }
    std::fprintf(f, "\",\"score\":%.4f}", ln.score);
  }
  std::fprintf(f, "],");
  std::fprintf(f, "\"ms\":{\"det\":%.2f,\"rec\":%.2f,\"cls\":%.2f,\"total\":%.2f}",
               r->det_ms, r->rec_ms, r->cls_ms, r->total_ms);
  if (prof) {
    std::fprintf(f,
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
    std::fputc('}', f);   // close the profile object AND the outer result
  } else {
    std::fputc('}', f);   // close the outer result (no profile)
  }
  std::fputc('\n', f);
}

} // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a) || a.help) { usage(); return a.help ? 0 : 1; }
  if (a.image.empty() || a.det_config.empty()) {
    std::fprintf(stderr, "error: --image and --det-config are required\n");
    usage();
    return 1;
  }

  // PERF5-urgent: in --boxes-json mode det is never used; skip its
  // ensure (sha256 of up-to-95MB) and session load entirely.
  const bool rec_only_mode = !a.boxes_json.empty();
  const std::string det_name = rec_only_mode
                                   ? std::string{}
                                   : config_basename(a.det_config);
  // --boxes-json requires only rec; keep arg validation above unchanged.
  const std::string rec_name = (!a.rec_config.empty())
                                   ? config_basename(a.rec_config)
                                   : std::string{};
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
