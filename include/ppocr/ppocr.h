// pp-ocr-mnn — public C ABI (v1, frozen)
// Multi-instance thread-safe. Synchronous by default; async optional.
// License: same as the project. No platform ifdefs here — see platform/ for wrappers.
#ifndef PPOCR_PPOCR_H_
#define PPOCR_PPOCR_H_

#include <stddef.h>
#include <stdint.h>

#include "ppocr/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define PPOCR_API __declspec(dllexport)
#else
#  define PPOCR_API __attribute__((visibility("default")))
#endif

// ---- version -------------------------------------------------------------
#define PPOCR_VERSION_MAJOR 0
#define PPOCR_VERSION_MINOR 1
#define PPOCR_VERSION_PATCH 0

// ---- status ---------------------------------------------------------------
typedef enum {
  PPOCR_OK = 0,
  PPOCR_ERR_PARAM = 1,        // invalid argument
  PPOCR_ERR_MODEL = 2,        // model file missing / corrupt / sha mismatch
  PPOCR_ERR_BACKEND = 3,      // requested backend unavailable
  PPOCR_ERR_DOWNLOAD = 4,     // model fetch failed
  PPOCR_ERR_IO = 5,           // image / filesystem error
  PPOCR_ERR_OOM = 6,
  PPOCR_ERR_INTERNAL = 7,
} ppocr_status;

// ---- backend --------------------------------------------------------------
typedef enum {
  PPOCR_BACKEND_AUTO = 0,     // pickBackend() decides per platform
  PPOCR_BACKEND_CPU = 1,
  PPOCR_BACKEND_CUDA = 2,
  PPOCR_BACKEND_OPENCL = 3,
  PPOCR_BACKEND_VULKAN = 4,
  PPOCR_BACKEND_METAL = 5,
  PPOCR_BACKEND_COREML = 6,
  PPOCR_BACKEND_NNAPI = 7,
} ppocr_backend;

// ---- configuration ---------------------------------------------------------
// All strings are UTF-8, copied at create time. Zero-init gives defaults.
typedef struct ppocr_config {
  const char* model_dir;      // directory containing .mnn + configs/ (default: env PPORC_MNN_MODELS or ./models)
  const char* cache_dir;      // download cache dir (default: env PPORC_MNN_CACHE or ~/.cache/ppocr-mnn)
  const char* det_name;       // det model name in registry (default "PP-OCRv6_tiny_det")
  const char* rec_name;       // rec model name (default "PP-OCRv6_tiny_rec"; NULL disables rec → det-only)
  const char* cls_name;       // cls model name (default NULL = off)
  const char* registry_path;  // registry.json override (default <model_dir>/configs/registry.json)
  const char* mirror;         // download base URL (default env PPORC_MNN_MIRROR)
  ppocr_backend backend;
  int num_threads;            // CPU threads (0 = auto)
  int rec_batch;              // rec batch size (0 = 16, M3-PERF4 sweet spot)
  int max_side;               // det max side limit (0 = from model config)
  int offline;                // 1 = never download, fail if missing
  int download;               // 0 = disable auto-download (default 1)
  int is_seal;                // M4-SEAL: 1 = seal recognition pipeline
                              // (skips reading-order sort; uses rec
                              // score threshold 0). Auto-detected from
                              // det_name containing "seal" when 0.
  int profile;                // M3-PERF1: 1 = collect per-stage timings,
                              // readable via ppocr_last_profile(). 0 = off
                              // (default; zero instrumentation cost).
  int warmup;                 // M3-PERF4: 1 = at ppocr_create, run one
                              // dummy det/rec/cls inference so the first
                              // real image skips the one-time backend init
                              // (~15ms cutlass select + workspace alloc on
                              // CUDA). 0 (default, matches {} zero-init) =
                              // skip; the CLI passes 1 unless --no-warmup.
} ppocr_config;

// ---- results ---------------------------------------------------------------
// Poly = 4 corner points in original-image pixel coords: x0,y0,...,x3,y3.
typedef struct ppocr_line {
  int poly[8];
  const char* text;           // UTF-8, "" when det-only
  float score;                // rec confidence (det box score when det-only)
} ppocr_line;

typedef struct ppocr_result {
  ppocr_line* lines;          // array of n_lines; owned by engine, valid until next run
  int n_lines;
  float det_ms, rec_ms, cls_ms, total_ms;
  char backend_used[16];      // e.g. "cpu", "cuda"
} ppocr_result;

typedef struct ppocr_engine ppocr_engine;

// M3-PERF1: return a pointer to the per-stage profile of the most
// recent run on this engine. Storage is owned by the engine and valid
// until the next run/destroy. Returns NULL when config.profile was 0
// at create time or the engine has not run yet.
PPOCR_API const ppocr_profile* ppocr_last_profile(const ppocr_engine* e);

// ---- lifecycle --------------------------------------------------------------
PPOCR_API ppocr_status ppocr_version(int* major, int* minor, int* patch);

// Creates engine: resolves models (download if enabled & missing), loads det/rec/cls.
// *out set on PPOCR_OK. Failure reason in optional buf.
PPOCR_API ppocr_status ppocr_create(const ppocr_config* cfg, ppocr_engine** out,
                                    char* err_buf, size_t err_buf_len);

PPOCR_API void ppocr_destroy(ppocr_engine* e);

// ---- synchronous inference ---------------------------------------------------
// image: BGR interleaved uint8, w*h*3 bytes. Thread-safe across threads per engine.
// Result internal buffers reused per engine; copy out before next call on same engine.
PPOCR_API ppocr_status ppocr_run(ppocr_engine* e, const uint8_t* bgr, int w, int h,
                                 ppocr_result** out);

// Same, from a file path (jpg/png/bmp via stb).
PPOCR_API ppocr_status ppocr_run_file(ppocr_engine* e, const char* path, ppocr_result** out);

// ---- optional async ----------------------------------------------------------
typedef void (*ppocr_callback)(ppocr_engine* e, ppocr_status st, ppocr_result* res, void* user);

PPOCR_API ppocr_status ppocr_run_async(ppocr_engine* e, const uint8_t* bgr, int w, int h,
                                       ppocr_callback cb, void* user);

// ---- M2-ISO: skip det, run rec on caller-supplied polygons ------------------
// polys: array of 8 int32 per box (TL,TR,BR,BL in image pixel coords), in
// caller-chosen order (e.g. reading order from a reference baseline).
// n_polys: number of boxes. rec_name must be set in the engine config.
// Experimental: added for the M2-ISO error-isolation audit; may move
// behind a feature flag in M3.
PPOCR_API ppocr_status ppocr_run_with_boxes(ppocr_engine* e, const uint8_t* bgr,
                                            int w, int h,
                                            const int* polys, int n_polys,
                                            ppocr_result** out);

// ---- utilities ----------------------------------------------------------------
PPOCR_API const char* ppocr_status_string(ppocr_status st);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // PPOCR_PPOCR_H_
