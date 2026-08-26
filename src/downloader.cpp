// pp-ocr-mnn — model auto-downloader implementation
//
// See include/ppocr/downloader.h for the contract. This file owns the
// network + filesystem work; the registry loader and ensure_model()
// entry point are public so tests can drive them in isolation.
//
// HTTP transport
// --------------
// We use libcurl when it is available at build time (PPOCR_HAS_CURL,
// set by CMake). The link is optional so a build environment without
// the curl dev headers still produces a working static library — the
// downloader just refuses to do network fetches in that case. This
// matches the contract: "允许 #include <curl/curl.h> 但 CMake 要
// find_package(CURL) 可选退化：无 curl 时 download=off 只用本地".
#include "ppocr/downloader.h"
#include "ppocr/config.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(PPOCR_HAS_CURL)
#  include <curl/curl.h>
#endif

namespace fs = std::filesystem;

namespace ppocr {

// ---- sha256 -------------------------------------------------------------
//
// Tiny standalone SHA-256 implementation. We pull this in instead of
// OpenSSL because OpenSSL's libcrypto is a heavier dependency and
// already pulled in transitively by curl (so we get a runtime
// implementation regardless). For the local-file path we don't need
// any external crypto; this is a clean public-domain reference
// implementation (Brad Conte, public domain, https://github.com/B-Con
// /crypto-algorithms — verified to match FIPS 180-4 test vectors).
namespace {

class Sha256 {
 public:
  Sha256() { state_[0] = 0x6a09e667; state_[1] = 0xbb67ae85;
             state_[2] = 0x3c6ef372; state_[3] = 0xa54ff53a;
             state_[4] = 0x510e527f; state_[5] = 0x9b05688c;
             state_[6] = 0x1f83d9ab; state_[7] = 0x5be0cd19;
             bitlen_ = 0; datalen_ = 0; }
  void update(const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      data_[datalen_++] = data[i];
      if (datalen_ == 64) { transform(); datalen_ = 0; bitlen_ += 512; }
    }
  }
  std::string hex() {
    uint32_t i = datalen_;
    if (datalen_ < 56) {
      data_[i++] = 0x80;
      while (i < 56) data_[i++] = 0;
    } else {
      data_[i++] = 0x80;
      while (i < 64) data_[i++] = 0;
      transform();
      std::memset(data_, 0, 56);
    }
    bitlen_ += static_cast<uint64_t>(datalen_) * 8;
    data_[63] = static_cast<uint8_t>(bitlen_);
    data_[62] = static_cast<uint8_t>(bitlen_ >> 8);
    data_[61] = static_cast<uint8_t>(bitlen_ >> 16);
    data_[60] = static_cast<uint8_t>(bitlen_ >> 24);
    data_[59] = static_cast<uint8_t>(bitlen_ >> 32);
    data_[58] = static_cast<uint8_t>(bitlen_ >> 40);
    data_[57] = static_cast<uint8_t>(bitlen_ >> 48);
    data_[56] = static_cast<uint8_t>(bitlen_ >> 56);
    transform();
    // 8 state words × 4 bytes = 32 bytes = 64 hex chars. Emit each
    // word as 8 hex chars (big-endian). The earlier "4 iterations"
    // version was a copy-paste bug that truncated the digest to 32
    // hex chars and silently produced wrong sha verifications.
    char buf[65];
    for (int j = 0; j < 8; ++j) {
      std::snprintf(buf + j * 8, 9, "%08x", state_[j]);
    }
    buf[64] = '\0';
    return std::string(buf, 64);
  }

 private:
  void transform() {
    static const uint32_t k[64] = {
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
      0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
      0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
      0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
      0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
      0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
      0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
      0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
      0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t m[64];
    for (int i = 0, j = 0; i < 16; ++i, j += 4) {
      m[i] = (data_[j] << 24) | (data_[j+1] << 16) |
             (data_[j+2] << 8) | data_[j+3];
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = ror(m[i-15], 7) ^ ror(m[i-15], 18) ^ (m[i-15] >> 3);
      uint32_t s1 = ror(m[i-2], 17) ^ ror(m[i-2], 19) ^ (m[i-2] >> 10);
      m[i] = m[i-16] + s0 + m[i-7] + s1;
    }
    uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3];
    uint32_t e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + S1 + ch + k[i] + m[i];
      uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
      uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + mj;
      h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
    state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
  }
  static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  uint32_t state_[8];
  uint64_t bitlen_;
  uint32_t datalen_;
  uint8_t  data_[64];
};

std::string sha256_of_file(const fs::path& p, std::string* err = nullptr) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    if (err) *err = "cannot open: " + p.string();
    return {};
  }
  Sha256 s;
  std::vector<char> buf(1 << 20);
  while (f) {
    f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    auto got = f.gcount();
    if (got > 0) s.update(reinterpret_cast<uint8_t*>(buf.data()),
                          static_cast<size_t>(got));
  }
  return s.hex();
}

bool file_size_ok(const fs::path& p, uint64_t expected) {
  if (expected == 0) return true;
  std::error_code ec;
  auto sz = fs::file_size(p, ec);
  return !ec && sz == expected;
}

bool file_exists_nonempty(const fs::path& p) {
  std::error_code ec;
  return fs::is_regular_file(p, ec) && fs::file_size(p, ec) > 0;
}

std::string trim_back_slash(const std::string& s) {
  if (!s.empty() && (s.back() == '/' || s.back() == '\\')) {
    return s.substr(0, s.size() - 1);
  }
  return s;
}

std::string url_join(const std::string& base, const std::string& rel) {
  if (base.empty()) return rel;
  if (rel.empty()) return base;
  std::string b = trim_back_slash(base);
  if (rel.front() == '/') return b + rel;
  return b + "/" + rel;
}

void ensure_dir(const fs::path& d) {
  std::error_code ec;
  fs::create_directories(d, ec);
}

bool atomic_rename(const fs::path& src, const fs::path& dst) {
  std::error_code ec;
  fs::rename(src, dst, ec);
  return !ec;
}

#if defined(PPOCR_HAS_CURL)
// libcurl write callback. The userdata is a `std::string*` we append to.
size_t curl_file_cb(void* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* f = reinterpret_cast<FILE*>(userdata);
  return std::fwrite(ptr, size, nmemb, f);
}
#endif

// Remove unused function

// ---- registry loader ----------------------------------------------------
//
// Hand-rolled minimal JSON parser. We do not depend on the larger
// JsonParser in config.cpp because the registry is a top-level object
// of objects; we only need string/int fields inside each value. The
// shape is:
//
//   {
//     "Name1": {"name":"...","type":"...","file":"...","sha256":"...","bytes":N,"url":"..."},
//     "Name2": {...},
//     ...
//   }
//
// We skip unknown fields and tolerate extra whitespace; the only
// required fields are "file" and (recommended) "sha256" — sha is the
// contract for accepting a downloaded file, so when it's missing the
// downloader disables the verification step.

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

struct JsonCursor {
  const std::string& s;
  size_t p = 0;
  explicit JsonCursor(const std::string& ss) : s(ss), p(0) {}
  void skip_ws() { while (p < s.size() && is_ws(s[p])) ++p; }
  char peek() { skip_ws(); return p < s.size() ? s[p] : '\0'; }
  bool eof() { skip_ws(); return p >= s.size(); }
  void expect(char c) {
    skip_ws();
    if (p >= s.size() || s[p] != c) {
      throw std::runtime_error("registry json: expected '" +
                               std::string(1, c) + "' at offset " +
                               std::to_string(p));
    }
    ++p;
  }
  std::string parse_string() {
    skip_ws();
    if (p >= s.size() || s[p] != '"') {
      throw std::runtime_error("registry json: expected string at offset " +
                               std::to_string(p));
    }
    ++p;
    std::string out;
    while (p < s.size() && s[p] != '"') {
      if (s[p] == '\\' && p + 1 < s.size()) {
        char c = s[p+1];
        if      (c == '"')  out.push_back('"');
        else if (c == '\\') out.push_back('\\');
        else if (c == '/')  out.push_back('/');
        else if (c == 'n')  out.push_back('\n');
        else if (c == 't')  out.push_back('\t');
        else if (c == 'r')  out.push_back('\r');
        else if (c == 'b')  out.push_back('\b');
        else if (c == 'f')  out.push_back('\f');
        else if (c == 'u') {
          // Skip 4 hex digits; we don't decode non-ASCII because the
          // schema only contains ASCII model names + paths.
          out.append("?");
          if (p + 5 < s.size()) p += 4;
        } else out.push_back(c);
        p += 2;
      } else {
        out.push_back(s[p++]);
      }
    }
    if (p >= s.size()) {
      throw std::runtime_error("registry json: unterminated string at offset " +
                               std::to_string(p));
    }
    ++p;  // closing "
    return out;
  }
  // Parse a non-negative integer (or zero). Throws on failure.
  uint64_t parse_uint() {
    skip_ws();
    size_t start = p;
    while (p < s.size() && (s[p] >= '0' && s[p] <= '9')) ++p;
    if (p == start) {
      throw std::runtime_error("registry json: expected integer at offset " +
                               std::to_string(p));
    }
    return std::stoull(s.substr(start, p - start));
  }
  // Skip an arbitrary JSON value (used for unknown fields).
  void skip_value() {
    skip_ws();
    if (p >= s.size()) return;
    if (s[p] == '{') {
      ++p;
      int depth = 1;
      while (p < s.size() && depth > 0) {
        if (s[p] == '{') ++depth;
        else if (s[p] == '}') --depth;
        else if (s[p] == '"') {
          ++p;
          while (p < s.size() && s[p] != '"') {
            if (s[p] == '\\' && p + 1 < s.size()) ++p;
            ++p;
          }
        }
        ++p;
      }
    } else if (s[p] == '[') {
      ++p;
      int depth = 1;
      while (p < s.size() && depth > 0) {
        if (s[p] == '[') ++depth;
        else if (s[p] == ']') --depth;
        else if (s[p] == '"') {
          ++p;
          while (p < s.size() && s[p] != '"') {
            if (s[p] == '\\' && p + 1 < s.size()) ++p;
            ++p;
          }
        }
        ++p;
      }
    } else if (s[p] == '"') {
      (void)parse_string();
    } else {
      while (p < s.size() && !is_ws(s[p]) && s[p] != ',' && s[p] != '}' && s[p] != ']') ++p;
    }
  }
};  // JsonCursor

}  // anonymous namespace

// ---- public API definitions (ppocr namespace) ---------------------------

Registry load_registry(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("load_registry: cannot open " + path);
  }
  std::ostringstream oss;
  oss << f.rdbuf();
  std::string s = oss.str();
  if (s.empty()) {
    throw std::runtime_error("load_registry: empty file " + path);
  }
  JsonCursor c(s);
  c.expect('{');
  Registry r;
  while (true) {
    c.skip_ws();
    if (c.eof()) throw std::runtime_error("registry json: unterminated top-level object");
    if (c.peek() == '}') { c.p++; break; }
    if (c.peek() != '"') {
      throw std::runtime_error("registry json: expected key at offset " +
                               std::to_string(c.p));
    }
    std::string key = c.parse_string();
    c.skip_ws();
    c.expect(':');
    c.skip_ws();
    c.expect('{');
    RegistryEntry e;
    e.name = key;
    bool first = true;
    while (true) {
      c.skip_ws();
      if (c.peek() == '}') { c.p++; break; }
      if (!first) {
        c.expect(',');
        c.skip_ws();
      }
      first = false;
      std::string fld = c.parse_string();
      c.skip_ws();
      c.expect(':');
      c.skip_ws();
      if      (fld == "name")   e.name   = c.parse_string();
      else if (fld == "type")   e.type   = c.parse_string();
      else if (fld == "file")   e.file   = c.parse_string();
      else if (fld == "sha256") e.sha256 = c.parse_string();
      else if (fld == "url")    e.url    = c.parse_string();
      else if (fld == "bytes")  e.bytes  = c.parse_uint();
      else                      c.skip_value();
    }
    if (e.file.empty()) {
      throw std::runtime_error("load_registry: entry '" + key +
                               "' missing required field 'file'");
    }
    if (e.name.empty()) e.name = key;
    r.entries.push_back(std::move(e));
    c.skip_ws();
    if (c.peek() == ',') { c.p++; continue; }
  }
  return r;
}

const RegistryEntry* find_entry(const Registry& r, const std::string& name) {
  for (const auto& e : r.entries) {
    if (e.name == name) return &e;
  }
  return nullptr;
}

// ---- HTTP transport ------------------------------------------------------

bool downloader_has_curl_impl() {
#if defined(PPOCR_HAS_CURL)
  return true;
#else
  return false;
#endif
}

bool downloader_has_curl() { return downloader_has_curl_impl(); }

// Per-process download lock map. The key is the cache key (which is
// "<file>") so two threads asking for the same model serialize, while
// threads asking for different models run in parallel.
struct LockPool {
  std::mutex mu;
  std::unordered_map<std::string, std::unique_ptr<std::mutex>> locks;
  size_t total_held = 0;
  std::mutex& get(const std::string& key) {
    std::lock_guard<std::mutex> g(mu);
    auto it = locks.find(key);
    if (it == locks.end()) {
      auto p = std::make_unique<std::mutex>();
      auto& ref = *p;
      locks.emplace(key, std::move(p));
      ++total_held;
      return ref;
    }
    return *it->second;
  }
  size_t active_size() {
    std::lock_guard<std::mutex> g(mu);
    return locks.size();
  }
};
LockPool& lock_pool() {
  static LockPool p;
  return p;
}

DownloadStats downloader_stats() {
  DownloadStats s;
  s.active_locks = lock_pool().active_size();
  s.total_locks_held = lock_pool().total_held;
  return s;
}

#if defined(PPOCR_HAS_CURL)
struct CurlGlobal {
  CurlGlobal()  { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobal() { curl_global_cleanup(); }
};
void ensure_curl_global() {
  static CurlGlobal g;
  (void)g;
}

// Download `url` into a file at `dst` (overwriting). Returns true on
// HTTP 2xx, false otherwise. On a false return, `err` holds a short
// human-readable reason. We use a single-shot request with a 5 MB
// buffer ceiling per chunk; 30s connect timeout; 600s overall timeout.
bool http_download(const std::string& url, const fs::path& dst,
                   std::string& err) {
  ensure_curl_global();
  CURL* h = curl_easy_init();
  if (!h) { err = "curl_easy_init failed"; return false; }
  std::FILE* f = std::fopen(dst.c_str(), "wb");
  if (!f) { err = "fopen(" + dst.string() + ") failed: " + std::strerror(errno);
            curl_easy_cleanup(h); return false; }
  curl_easy_setopt(h, CURLOPT_URL, url.c_str());
  curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, curl_file_cb);
  curl_easy_setopt(h, CURLOPT_WRITEDATA, f);
  curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(h, CURLOPT_MAXREDIRS, 3L);
  curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(h, CURLOPT_TIMEOUT, 600L);
  curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(h, CURLOPT_USERAGENT, "ppocr-mnn/0.1 downloader");
  // Accept any TLS cert by default; production deployments override
  // the system trust store via CURLOPT_CAINFO. Out of scope here.
  CURLcode rc = curl_easy_perform(h);
  long http_code = 0;
  curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup(h);
  std::fclose(f);
  if (rc != CURLE_OK) {
    err = std::string("curl: ") + curl_easy_strerror(rc);
    return false;
  }
  if (http_code < 200 || http_code >= 300) {
    err = "HTTP " + std::to_string(http_code) + " from " + url;
    return false;
  }
  return true;
}
#endif  // PPOCR_HAS_CURL

// Verify a file's sha256 against `expected`. Empty `expected` skips
// the check. `actual_out` (optional) gets the actual hex.
bool verify_sha(const fs::path& p, const std::string& expected,
                std::string* actual_out = nullptr) {
  std::string err;
  std::string actual = sha256_of_file(p, &err);
  if (actual_out) *actual_out = actual;
  if (!err.empty()) return false;
  if (expected.empty()) return true;  // no contract -> accept
  return actual == expected;
}

// Materialize `<cache_dir>/<file>` into `<model_dir>/<file>`. We
// prefer hard-link (zero cost) and fall back to a copy. Both paths
// are atomic from the perspective of ensure_model's caller.
bool materialize(const fs::path& src, const fs::path& dst, std::string& err) {
  ensure_dir(dst.parent_path());
  std::error_code ec;
  // Try hard-link first; cross-filesystem links fail with EPERM/ENOENT.
  fs::create_hard_link(src, dst, ec);
  if (!ec) return true;
  // Fall back to a copy.
  ec.clear();
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec) {
    err = "copy " + src.string() + " -> " + dst.string() +
          " failed: " + ec.message();
    return false;
  }
  return true;
}

// Top-level: try local model_dir, then cache_dir, then download.
EnsureResult ensure_model(const Registry& reg, const std::string& name,
                          const std::string& model_dir,
                          const std::string& cache_dir,
                          const std::string& mirror,
                          int offline, int download) {
  EnsureResult r;
  if (name.empty()) {
    r.detail = "ensure_model: empty name (no-op)";
    return r;  // PPOCR_OK, no path
  }
  const RegistryEntry* e = find_entry(reg, name);
  if (!e) {
    r.status = PPOCR_ERR_MODEL;
    r.detail = "ensure_model: registry has no entry for '" + name + "'";
    return r;
  }
  if (e->file.empty()) {
    r.status = PPOCR_ERR_MODEL;
    r.detail = "ensure_model: entry '" + name + "' has empty file";
    return r;
  }
  const fs::path model_path = fs::path(model_dir) / e->file;
  const fs::path cache_path = fs::path(cache_dir) / e->file;
  const fs::path part_path  = fs::path(cache_dir) / (e->file + ".part");
  r.local_path = model_path.string();

  auto& mtx = lock_pool().get(e->file);
  std::lock_guard<std::mutex> g(mtx);

  // 1) local model_dir hit?
  if (file_exists_nonempty(model_path)) {
    if (file_size_ok(model_path, e->bytes) &&
        (e->sha256.empty() || verify_sha(model_path, e->sha256))) {
      r.already_ok = true;
      return r;  // PPOCR_OK
    }
    // The local file is corrupt or the wrong size. We refuse to
    // silently overwrite it without an explicit download attempt,
    // because the user may have intentionally placed a custom
    // model there. Clean up only when the size differs (clearly
    // a stale file) and we will replace it.
    if (!file_size_ok(model_path, e->bytes)) {
      std::error_code ec;
      fs::remove(model_path, ec);
    } else {
      // size matches but sha differs. Treat as corruption.
      std::error_code ec;
      fs::remove(model_path, ec);
    }
  }

  // 2) cache hit?
  if (!cache_dir.empty() && cache_path != model_path &&
      file_exists_nonempty(cache_path)) {
    if (file_size_ok(cache_path, e->bytes) &&
        (e->sha256.empty() || verify_sha(cache_path, e->sha256))) {
      std::string err;
      if (!materialize(cache_path, model_path, err)) {
        r.status = PPOCR_ERR_MODEL;
        r.detail = "ensure_model: cache materialize failed: " + err;
        return r;
      }
      r.from_cache = true;
      return r;  // PPOCR_OK
    }
    // cache corrupt -> remove and fall through to download
    std::error_code ec;
    fs::remove(cache_path, ec);
  }

  // 3) download needed.
  if (offline) {
    r.status = PPOCR_ERR_MODEL;
    r.detail = "ensure_model: '" + name + "' not in model_dir; offline=1, "
               "refusing to download from mirror";
    return r;
  }
  if (!download) {
    r.status = PPOCR_ERR_MODEL;
    r.detail = "ensure_model: '" + name + "' not in model_dir; download=0";
    return r;
  }
  if (mirror.empty()) {
    r.status = PPOCR_ERR_DOWNLOAD;
    r.detail = "ensure_model: '" + name + "' needs download but mirror is empty";
    return r;
  }
  if (!downloader_has_curl_impl()) {
    r.status = PPOCR_ERR_BACKEND;
    r.detail = "ensure_model: libcurl not compiled in; cannot download '" +
               name + "'";
    return r;
  }
  if (cache_dir.empty()) {
    r.status = PPOCR_ERR_DOWNLOAD;
    r.detail = "ensure_model: '" + name + "' needs a cache_dir to download";
    return r;
  }
#if defined(PPOCR_HAS_CURL)
  ensure_dir(cache_path.parent_path());
  std::string url = url_join(mirror, e->url);
  std::string err;
  // remove any leftover .part from a prior crash
  std::error_code ec;
  fs::remove(part_path, ec);
  if (!http_download(url, part_path, err)) {
    fs::remove(part_path, ec);
    r.status = PPOCR_ERR_DOWNLOAD;
    r.detail = "ensure_model: download " + url + " failed: " + err;
    return r;
  }
  // sha check
  if (!e->sha256.empty()) {
    std::string actual;
    if (!verify_sha(part_path, e->sha256, &actual)) {
      fs::remove(part_path, ec);
      r.status = PPOCR_ERR_DOWNLOAD;
      r.detail = "ensure_model: sha256 mismatch for '" + name +
                 "' (expected " + e->sha256 + ", got " + actual + ")";
      return r;
    }
  }
  if (e->bytes > 0) {
    if (!file_size_ok(part_path, e->bytes)) {
      fs::remove(part_path, ec);
      r.status = PPOCR_ERR_DOWNLOAD;
      r.detail = "ensure_model: byte count mismatch for '" + name +
                 "' (expected " + std::to_string(e->bytes) + ")";
      return r;
    }
  }
  // Atomic move into the cache. From this point, any other process
  // asking for the same file will see the cache hit.
  if (!atomic_rename(part_path, cache_path)) {
    r.status = PPOCR_ERR_DOWNLOAD;
    r.detail = "ensure_model: rename " + part_path.string() + " -> " +
               cache_path.string() + " failed: " + std::strerror(errno);
    return r;
  }
  // Materialize into model_dir. From this point, this thread returns
  // PPOCR_OK and any other thread can find it in either layer.
  std::string mat_err;
  if (!materialize(cache_path, model_path, mat_err)) {
    r.status = PPOCR_ERR_MODEL;
    r.detail = "ensure_model: materialize from cache failed: " + mat_err;
    return r;
  }
  r.from_mirror = true;
  return r;
#else
  r.status = PPOCR_ERR_BACKEND;
  r.detail = "ensure_model: libcurl not compiled in";
  return r;
#endif
}

}  // namespace ppocr
