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
  int threads = 0;
  int batch   = 0;
  int det_only = 0;
  int time    = 0;
  std::string out_path;          // empty -> stdout
  int max_side = 0;
  int help    = 0;
};

void usage() {
  std::fprintf(stderr,
    "ppocr_cli --image IMG --det-config X [--rec-config Y] [--cls-config Z]\n"
    "          [--model-dir DIR] [--backend auto|cpu|cuda|opencl|vulkan]\n"
    "          [--threads N] [--batch N] [--det-only] [--json OUT] [--time]\n");
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
    else if (k == "--time")          { a.time = 1; }
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

void write_result(FILE* f, const char* image, const ppocr_result* r) {
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
  std::fprintf(f, "\"ms\":{\"det\":%.2f,\"rec\":%.2f,\"cls\":%.2f,\"total\":%.2f}}",
               r->det_ms, r->rec_ms, r->cls_ms, r->total_ms);
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
  cfg.max_side   = a.max_side;
  cfg.offline    = 1; // M1: never download; if file is missing we want a hard error
  cfg.download   = 0;

  char err[256] = {0};
  ppocr_engine* engine = nullptr;
  ppocr_status st = ppocr_create(&cfg, &engine, err, sizeof(err));
  if (st != PPOCR_OK) {
    std::fprintf(stderr, "ppocr_create failed: %s (%s)\n",
                 ppocr_status_string(st), err);
    return 2;
  }

  ppocr_result* result = nullptr;
  st = ppocr_run_file(engine, a.image.c_str(), &result);
  if (st != PPOCR_OK) {
    std::fprintf(stderr, "ppocr_run_file failed: %s\n",
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
  write_result(out, a.image.c_str(), result);
  if (a.time) {
    std::fprintf(stderr, "det=%.2fms rec=%.2fms total=%.2fms n_lines=%d\n",
                 result->det_ms, result->rec_ms, result->total_ms,
                 result->n_lines);
  }
  if (owned) std::fclose(owned.release());
  ppocr_destroy(engine);
  return 0;
}
