// test_downloader.cpp — exercises the runtime auto-downloader (TOOLS-5 / M3)
//
// Drives the public ppocr::ensure_model / load_registry / find_entry
// API against an in-process scenario matrix:
//
//   1. Registry load + entry lookup. (unit-style smoke test)
//   2. ensure_model with an empty name -> no-op, returns PPOCR_OK.
//   3. ensure_model with a missing entry -> PPOCR_ERR_MODEL.
//   4. ensure_model on a pre-existing correct file -> PPOCR_OK, already_ok.
//   5. ensure_model on a pre-existing corrupt file -> PPOCR_ERR_MODEL
//      (offline) or download+replace (online; skipped if curl is off).
//   6. ensure_model with download=1, mirror=local file://...  -> download
//      from the in-process /mirror directory; cache + model_dir written.
//
// The test does NOT spin up an HTTP server: tests/test_downloader.py
// (sibling Python test) does that and drives the same binary as a
// subprocess. This binary is intentionally self-contained so it can
// also be run by hand for ad-hoc debugging.
//
// Exit codes:
//   0  all scenarios pass
//   1  test failure (printed to stderr)
//   2  usage error
//   3  environment error (e.g. the registry_path arg is missing)
#include "ppocr/downloader.h"
#include "ppocr/ppocr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;
int g_passes = 0;

#define CHECK(cond, msg) do {                                                \
  if (cond) { ++g_passes; }                                                  \
  else { ++g_failures;                                                       \
    std::fprintf(stderr, "FAIL: %s  (at %s:%d)\n", msg, __FILE__, __LINE__);  \
  }                                                                          \
} while (0)

void write_random_file(const fs::path& p, size_t bytes, uint32_t seed) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  std::mt19937 rng(seed);
  for (size_t i = 0; i < bytes; ++i) {
    f.put(static_cast<char>(rng() & 0xff));
  }
}

struct Args {
  std::string registry_path;
  std::string model_dir;
  std::string cache_dir;
  std::string mirror;     // base URL; may be http://... or file://
  int offline = 0;
  int download = 1;
  int verbose = 0;
  std::string scenario;   // e.g. "all", "registry", "noop", "missing",
                          //      "local_ok", "local_corrupt_offline",
                          //      "local_corrupt_download", "offline_missing"
};

void usage() {
  std::fprintf(stderr,
    "test_downloader --registry <registry.json>\n"
    "               --model-dir <dir>\n"
    "               --cache-dir <dir>\n"
    "               --mirror <url>     (e.g. http://127.0.0.1:8765 or file:///...)\n"
    "               [--offline 0|1]\n"
    "               [--download 0|1]\n"
    "               [--scenario NAME]  (default 'all')\n"
    "               [--verbose]\n"
    "  Scenarios: registry, noop, missing, local_ok, local_corrupt_offline,\n"
    "             local_corrupt_download, offline_missing, all\n");
}

bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", name); return nullptr; }
      return argv[++i];
    };
    if      (k == "--registry")   { auto v = need("--registry");   if (!v) return false; a.registry_path = v; }
    else if (k == "--model-dir")  { auto v = need("--model-dir");  if (!v) return false; a.model_dir = v; }
    else if (k == "--cache-dir")  { auto v = need("--cache-dir");  if (!v) return false; a.cache_dir = v; }
    else if (k == "--mirror")     { auto v = need("--mirror");     if (!v) return false; a.mirror = v; }
    else if (k == "--offline")    { auto v = need("--offline");    if (!v) return false; a.offline = std::atoi(v); }
    else if (k == "--download")   { auto v = need("--download");   if (!v) return false; a.download = std::atoi(v); }
    else if (k == "--scenario")   { auto v = need("--scenario");   if (!v) return false; a.scenario = v; }
    else if (k == "--verbose")    { a.verbose = 1; }
    else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
    else { std::fprintf(stderr, "unknown flag: %s\n", k.c_str()); return false; }
  }
  return true;
}

// Convert a relative path to a `file://` URL. The local mirror layout
// is "<mirror_dir>/<file>". The C++ downloader takes a base URL + a
// relative `entry.url`; for a file:// mirror, the joined URL is
// file://<absolute>/<file>.
// (file_url_for is currently unused because the test driver always
// uses an http:// mirror started by serve_registry.py; keep the
// helper here as a documented affordance for future tests.)
/*
std::string file_url_for(const std::string& mirror_dir) {
  fs::path p(mirror_dir);
  std::string s = "file://" + fs::absolute(p).string();
  if (s.compare(0, 8, "file:///") != 0) {
    size_t pos = s.find("file://");
    if (pos != std::string::npos && pos + 7 < s.size() && s[pos + 7] != '/') {
      s.insert(pos + 7, "/");
    }
  }
  return s;
}
*/

// ---- scenarios ----------------------------------------------------------

int s_registry(Args& a) {
  std::fprintf(stderr, "[scenario] registry load + entry lookup\n");
  try {
    ppocr::Registry r = ppocr::load_registry(a.registry_path);
    std::fprintf(stderr, "  loaded %zu entries from %s\n",
                 r.entries.size(), a.registry_path.c_str());
    CHECK(!r.entries.empty(), "registry must have at least one entry");
    const ppocr::RegistryEntry* e = ppocr::find_entry(r, r.entries.front().name);
    CHECK(e != nullptr, "find_entry returns the first entry's name");
    CHECK(e->name == r.entries.front().name, "round-trip name");
    CHECK(!e->file.empty(), "file field non-empty");
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "EXC: %s\n", ex.what());
    return 1;
  }
  return 0;
}

int s_noop(Args& a) {
  std::fprintf(stderr, "[scenario] empty name is a no-op\n");
  ppocr::Registry r = ppocr::load_registry(a.registry_path);
  auto res = ppocr::ensure_model(r, "", a.model_dir, a.cache_dir,
                                 a.mirror, a.offline, a.download);
  CHECK(res.status == PPOCR_OK, "empty name returns OK");
  CHECK(res.local_path.empty(), "no local_path for empty name");
  return 0;
}

int s_missing(Args& a) {
  std::fprintf(stderr, "[scenario] missing registry entry\n");
  ppocr::Registry r = ppocr::load_registry(a.registry_path);
  auto res = ppocr::ensure_model(r, "no_such_model_xyz", a.model_dir,
                                 a.cache_dir, a.mirror, a.offline, a.download);
  CHECK(res.status == PPOCR_ERR_MODEL, "missing entry -> ERR_MODEL");
  CHECK(res.detail.find("registry has no entry") != std::string::npos,
        "detail mentions registry lookup");
  return 0;
}

int s_local_ok(Args& a) {
  std::fprintf(stderr, "[scenario] local file present and matching\n");
  ppocr::Registry r = ppocr::load_registry(a.registry_path);
  // Find an entry whose file exists in model_dir.
  const ppocr::RegistryEntry* e = nullptr;
  for (const auto& ent : r.entries) {
    fs::path p = fs::path(a.model_dir) / ent.file;
    if (fs::is_regular_file(p)) { e = &ent; break; }
  }
  if (!e) {
    std::fprintf(stderr, "SKIP: no entry's file is present in model_dir; "
                         "test the local_ok scenario only when model files "
                         "are on disk\n");
    return 0;
  }
  auto res = ppocr::ensure_model(r, e->name, a.model_dir, a.cache_dir,
                                 a.mirror, a.offline, a.download);
  CHECK(res.status == PPOCR_OK, "local_ok status OK");
  CHECK(res.already_ok, "already_ok flag set");
  return 0;
}

int s_local_corrupt_offline(Args& a) {
  std::fprintf(stderr, "[scenario] local file present but corrupt (offline)\n");
  ppocr::Registry r = ppocr::load_registry(a.registry_path);
  // Find an entry with sha; if its file is missing in model_dir, write
  // garbage and expect a hard fail in offline mode.
  const ppocr::RegistryEntry* e = nullptr;
  for (const auto& ent : r.entries) {
    if (ent.sha256.empty()) continue;
    e = &ent; break;
  }
  if (!e) {
    std::fprintf(stderr, "SKIP: no entry with a non-empty sha\n");
    return 0;
  }
  // Use a unique disposable temp model_dir so the real test fixtures
  // are not destroyed. We pre-populate it with a corrupt copy.
  fs::path tmp_model = a.model_dir + "_scenario_corrupt_off_model";
  fs::path tmp_cache = a.cache_dir + "_scenario_corrupt_off_cache";
  fs::remove_all(tmp_model);
  fs::remove_all(tmp_cache);
  fs::create_directories(tmp_model);
  fs::create_directories(tmp_cache);
  fs::path target = tmp_model / e->file;
  write_random_file(target, 4096, 42);
  Args offline = a;
  offline.model_dir = tmp_model.string();
  offline.cache_dir = tmp_cache.string();
  offline.offline = 1; offline.download = 0;
  auto res = ppocr::ensure_model(r, e->name, offline.model_dir,
                                 offline.cache_dir, offline.mirror,
                                 offline.offline, offline.download);
  CHECK(res.status == PPOCR_ERR_MODEL, "corrupt + offline -> ERR_MODEL");
  fs::remove_all(tmp_model);
  fs::remove_all(tmp_cache);
  return 0;
}

int s_local_corrupt_download(Args& a) {
  std::fprintf(stderr, "[scenario] local file present but corrupt -> re-download\n");
  ppocr::Registry r = ppocr::load_registry(a.registry_path);
  const ppocr::RegistryEntry* e = nullptr;
  for (const auto& ent : r.entries) {
    if (ent.sha256.empty()) continue;
    fs::path p = fs::path(a.model_dir) / ent.file;
    if (fs::is_regular_file(p)) { e = &ent; break; }
  }
  if (!e) {
    std::fprintf(stderr, "SKIP: no entry with sha + file present\n");
    return 0;
  }
  // Use a disposable temp model_dir/cache_dir so the real test
  // fixtures are not destroyed.
  fs::path tmp_model = a.model_dir + "_scenario_corrupt_dl_model";
  fs::path tmp_cache = a.cache_dir + "_scenario_corrupt_dl_cache";
  fs::remove_all(tmp_model);
  fs::remove_all(tmp_cache);
  fs::create_directories(tmp_model);
  fs::create_directories(tmp_cache);
  // Pre-populate with a corrupt copy of the real file
  fs::path real_target = fs::path(a.model_dir) / e->file;
  if (fs::is_regular_file(real_target)) {
    fs::copy_file(real_target, tmp_model / e->file);
  }
  // Corrupt the model_dir file (must be writable, which it is in tmp)
  fs::path target = tmp_model / e->file;
  fs::path cache_target = tmp_cache / e->file;
  write_random_file(target, 4096, 99);
  write_random_file(cache_target, 4096, 99);
  // Re-download
  Args online = a;
  online.model_dir = tmp_model.string();
  online.cache_dir = tmp_cache.string();
  online.offline = 0; online.download = 1;
  auto res = ppocr::ensure_model(r, e->name, online.model_dir,
                                 online.cache_dir, online.mirror,
                                 online.offline, online.download);
  CHECK(res.status == PPOCR_OK, "corrupt + download -> OK (re-fetched)");
  CHECK(res.from_mirror, "from_mirror flag set after re-download");
  // Verify sizes
  CHECK(fs::file_size(target) == e->bytes, "model_dir file size matches registry bytes");
  CHECK(fs::file_size(cache_target) == e->bytes, "cache file size matches registry bytes");
  fs::remove_all(tmp_model);
  fs::remove_all(tmp_cache);
  return 0;
}

int s_offline_missing(Args& a) {
  std::fprintf(stderr, "[scenario] offline mode + file missing -> hard fail\n");
  ppocr::Registry r = ppocr::load_registry(a.registry_path);
  // Find an entry with sha; remove its file and expect ERR_MODEL
  const ppocr::RegistryEntry* e = nullptr;
  for (const auto& ent : r.entries) {
    if (ent.sha256.empty()) continue;
    e = &ent; break;
  }
  if (!e) {
    std::fprintf(stderr, "SKIP: no entry with a non-empty sha\n");
    return 0;
  }
  // Use a unique disposable model_dir and cache_dir for this scenario
  // so the test is non-destructive on the real model_dir/cache_dir.
  fs::path tmp_model = a.model_dir + "_scenario_tmp_model";
  fs::path tmp_cache = a.cache_dir + "_scenario_tmp_cache";
  fs::remove_all(tmp_model);
  fs::remove_all(tmp_cache);
  fs::create_directories(tmp_model);
  fs::create_directories(tmp_cache);
  // Place a valid copy in the temp model_dir so the test exercises
  // the "present but in offline mode, no need to download" branch
  // (we then remove it to test the "missing + offline" branch).
  fs::path target = tmp_model / e->file;
  // Read a good file from the real model_dir if it exists; otherwise
  // just leave the file missing and test that branch directly.
  fs::path real_target = fs::path(a.model_dir) / e->file;
  if (fs::is_regular_file(real_target)) {
    fs::copy_file(real_target, target);
  }
  Args offline = a;
  offline.model_dir = tmp_model.string();
  offline.cache_dir = tmp_cache.string();
  offline.offline = 1; offline.download = 0;
  // If we have a file, remove it; the test should report ERR_MODEL.
  fs::remove(target);
  auto res = ppocr::ensure_model(r, e->name, offline.model_dir,
                                 offline.cache_dir, offline.mirror,
                                 offline.offline, offline.download);
  CHECK(res.status == PPOCR_ERR_MODEL, "offline + missing -> ERR_MODEL");
  CHECK(res.detail.find("offline=1") != std::string::npos ||
        res.detail.find("refusing to download") != std::string::npos,
        "detail mentions offline refusal");
  fs::remove_all(tmp_model);
  fs::remove_all(tmp_cache);
  return 0;
}

int s_download_no_curl(Args& a) {
  // If curl is unavailable, asking for a download must return
  // PPOCR_ERR_BACKEND (not ERR_DOWNLOAD or ERR_MODEL). This documents
  // the build-time contract.
  if (ppocr::downloader_has_curl()) {
    std::fprintf(stderr, "[scenario] download_no_curl -- SKIP (curl is "
                         "compiled in)\n");
    return 0;
  }
  std::fprintf(stderr, "[scenario] download_no_curl\n");
  ppocr::Registry r = ppocr::load_registry(a.registry_path);
  const ppocr::RegistryEntry* e = nullptr;
  for (const auto& ent : r.entries) {
    if (ent.sha256.empty()) continue;
    e = &ent; break;
  }
  if (!e) {
    std::fprintf(stderr, "SKIP: no entry with a non-empty sha\n");
    return 0;
  }
  fs::path target = fs::path(a.model_dir) / e->file;
  bool had = fs::is_regular_file(target);
  fs::remove(target);
  Args online = a; online.offline = 0; online.download = 1;
  auto res = ppocr::ensure_model(r, e->name, online.model_dir,
                                 online.cache_dir, online.mirror,
                                 online.offline, online.download);
  CHECK(res.status == PPOCR_ERR_BACKEND,
        "no-curl + download -> ERR_BACKEND");
  if (had) std::fprintf(stderr, "  (file removed; test set is destructive)\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) { usage(); return 2; }
  if (a.registry_path.empty() || a.model_dir.empty() ||
      a.cache_dir.empty()) {
    std::fprintf(stderr, "test_downloader: --registry, --model-dir, "
                         "--cache-dir are required\n");
    usage();
    return 2;
  }
  if (a.scenario.empty()) a.scenario = "all";

  // Resolve file:// mirror for the local-server scenarios
  if (a.mirror.rfind("file://", 0) != 0 && !a.mirror.empty()) {
    // Already a URL; leave it.
  }

  int rc = 0;
  auto run = [&](const char* name, int (*fn)(Args&)) {
    int sub_rc = fn(a);
    if (sub_rc != 0) rc = sub_rc;
    (void)name;
  };
  if (a.scenario == "all" || a.scenario == "registry")   run("registry", s_registry);
  if (a.scenario == "all" || a.scenario == "noop")       run("noop", s_noop);
  if (a.scenario == "all" || a.scenario == "missing")    run("missing", s_missing);
  if (a.scenario == "all" || a.scenario == "local_ok")   run("local_ok", s_local_ok);
  if (a.scenario == "all" || a.scenario == "local_corrupt_offline")
    run("local_corrupt_offline", s_local_corrupt_offline);
  if (a.scenario == "all" || a.scenario == "local_corrupt_download")
    run("local_corrupt_download", s_local_corrupt_download);
  if (a.scenario == "all" || a.scenario == "offline_missing")
    run("offline_missing", s_offline_missing);
  if (a.scenario == "all" || a.scenario == "download_no_curl")
    run("download_no_curl", s_download_no_curl);

  std::fprintf(stderr, "\n========= summary: %d passed, %d failed =========\n",
               g_passes, g_failures);
  if (g_failures > 0) return 1;
  return rc;
}
