// pp-ocr-mnn — model auto-downloader (owner: ws/tools, TOOLS-5 / M3)
//
// Wires the runtime to the registry.json produced by tools/convert_models.py
// (30 model entries, each with name/file/sha256/url). The contract is:
//
//   1. Look in <model_dir>/<file>. If present and its sha256 matches the
//      registry's expected value, use it. Mismatches are treated as
//      corruption and re-fetched from the cache or mirror.
//
//   2. Otherwise look in <cache_dir>/<file>. If present and its sha256
//      matches, copy (or hard-link) it into <model_dir>/<file>.
//
//   3. Otherwise download from <mirror>/<url> to <cache_dir>/<file>.part,
//      verify its sha256, and atomically rename to <cache_dir>/<file>.
//      Then materialize it in <model_dir>.
//
// `offline=1` and `download=0` both short-circuit the network path with
// a hard error if the file is missing locally. Empty `mirror` does the
// same when a download is required.
//
// HTTP transport: libcurl when available (detected at CMake configure
// time). Without libcurl the downloader compiles to a stub that returns
// PPOCR_ERR_BACKEND with a clear message; offline / model_dir-local
// usage still works.
//
// Multi-instance thread safety: a per-process map of
// `<url-or-cache-key> -> std::mutex` serializes concurrent downloads of
// the same file. Two threads racing on `ensure_model("X.mnn", ...)` will
// take the same lock; threads requesting different files run in parallel.
// Cache writes use the atomic-rename pattern (write to .part, fsync,
// rename) so a crashed downloader leaves a harmless partial file rather
// than a corrupted final file.
#ifndef PPOCR_DOWNLOADER_H_
#define PPOCR_DOWNLOADER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "ppocr/ppocr.h"  // for ppocr_status

namespace ppocr {

// One row in the registry. Mirrors ModelConfig's provenance fields;
// defined as a separate type so the downloader can be tested without
// pulling in the full model config schema.
struct RegistryEntry {
  std::string name;     // registry key
  std::string type;     // "det" | "rec" | "cls"
  std::string file;     // relative filename inside model_dir
  std::string sha256;   // expected hex digest of `file`
  std::string url;      // download URL relative to mirror base
  uint64_t    bytes = 0;
};

// Read `path` as a registry.json and return the parsed entries.
// The expected format is a top-level object whose keys are model
// names; each value is a RegistryEntry.
//
// Throws std::runtime_error on missing file, malformed JSON, or any
// entry with a missing `file` or `sha256` (sha is the contract for
// accepting a downloaded file; the downloader can still operate if
// sha is empty, but it will skip the verification step).
//
// Implementation lives in config.cpp alongside load_model_config.
struct Registry {
  std::vector<RegistryEntry> entries;
};
Registry load_registry(const std::string& path);

// Find an entry by name. Returns nullptr if not present.
const RegistryEntry* find_entry(const Registry& r, const std::string& name);

// Result of ensure_model. On PPOCR_OK the local file is at
// <model_dir>/<file> with the right sha.
struct EnsureResult {
  ppocr_status status = PPOCR_OK;
  std::string  local_path;     // absolute or relative path used
  std::string  detail;         // error message or info line
  bool         from_cache = false;
  bool         from_mirror = false;
  bool         already_ok = false;  // true when local file already matched
};

// Resolve <model_dir>/<file> for `name`. If the local file is missing
// or its sha doesn't match the registry entry, and the configuration
// allows it, download from <mirror>/<url> into <cache_dir> and copy
// (or hard-link) into <model_dir>.
//
// Parameters:
//   reg           parsed registry
//   name          model name (registry key). Empty string is a no-op
//                 that returns PPOCR_OK with empty local_path.
//   model_dir     directory that holds the .mnn files. Will be created
//                 (mkdir -p) if missing. The final file lives at
//                 <model_dir>/<entry.file>.
//   cache_dir     directory for the persistent download cache. May be
//                 the same as model_dir. Will be created if missing.
//   mirror        base URL for downloads. "" disables downloads even
//                 when `download` is true.
//   offline       1 = never download. Mirrors and the cache are still
//                 consulted; only network fetches are disabled.
//   download      0 = never download. Otherwise downloads are allowed
//                 when needed.
//
// On any error, EnsureResult.status is a non-OK ppocr_status
// (PPOCR_ERR_MODEL if the file is missing or corrupt and we can't
// fetch; PPOCR_ERR_DOWNLOAD for network or sha mismatch; PPOCR_ERR_BACKEND
// when curl is unavailable and a download would be required).
//
// Thread-safe across multiple threads / multiple engines; per-file
// serialization is internal.
EnsureResult ensure_model(const Registry& reg, const std::string& name,
                          const std::string& model_dir,
                          const std::string& cache_dir,
                          const std::string& mirror,
                          int offline, int download);

// True when libcurl was available at build time. When false, the
// downloader refuses to attempt any network call.
bool downloader_has_curl();

// Return the in-process cache of "currently being downloaded" entries;
// for tests and diagnostics.
struct DownloadStats {
  size_t active_locks = 0;
  size_t total_locks_held = 0;
};
DownloadStats downloader_stats();

} // namespace ppocr

#endif // PPOCR_DOWNLOADER_H_
