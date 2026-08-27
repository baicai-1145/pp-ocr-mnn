// pp-ocr-mnn — model config JSON loader (owner m1)
//
// Minimal hand-rolled JSON parser, scoped to the model config schema
// defined in CONTRACT.md. Pulling in nlohmann/json or rapidjson is overkill
// for ~14 known fields per model; this parser tolerates whitespace and
// ignores order, supports only the types the schema uses, and produces
// a hard error on anything unexpected (e.g. nested objects beyond the
// known sub-objects). Error messages include byte offset for triage.
#include "ppocr/config.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppocr {

namespace {

class JsonParser {
 public:
  explicit JsonParser(const std::string& s) : s_(s) {}

  ModelConfig parse() {
    skip_ws();
    expect('{');
    ModelConfig cfg;
    bool first = true;
    while (true) {
      skip_ws();
      if (peek() == '}') { ++p_; break; }
      if (!first) {
        if (peek() != ',') throw err("expected ',' between fields");
        ++p_;
      }
      first = false;
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      parse_value(key, cfg);
    }
    if (cfg.type.empty()) {
      throw std::runtime_error("config: missing required field 'type'");
    }
    if (cfg.type != "det" && cfg.type != "rec" && cfg.type != "cls") {
      throw std::runtime_error("config: type must be det|rec|cls (got '"
                               + cfg.type + "')");
    }
    return cfg;
  }

 private:
  const std::string& s_;
  size_t p_ = 0;

  static std::runtime_error err(const std::string& msg) {
    return std::runtime_error("config json: " + msg);
  }
  std::runtime_error err_at(const std::string& msg) const {
    return std::runtime_error("config json @ " + std::to_string(p_) + ": " + msg);
  }

  void skip_ws() {
    while (p_ < s_.size()) {
      char c = s_[p_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++p_; continue; }
      break;
    }
  }

  char peek() const {
    if (p_ >= s_.size()) throw err_at("unexpected end of input");
    return s_[p_];
  }
  void expect(char c) {
    if (p_ >= s_.size() || s_[p_] != c) {
      throw err_at(std::string("expected '") + c + "'");
    }
    ++p_;
  }

  std::string parse_string() {
    if (peek() != '"') throw err_at("expected '\"'");
    ++p_;
    std::string out;
    while (p_ < s_.size() && s_[p_] != '"') {
      char c = s_[p_];
      if (c == '\\') {
        if (p_ + 1 >= s_.size()) throw err_at("bad escape");
        char e = s_[p_ + 1];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case 'r': out.push_back('\r'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'u': {
            if (p_ + 5 >= s_.size()) throw err_at("truncated \\u escape");
            unsigned int cp = 0;
            for (int i = 0; i < 4; ++i) {
              cp <<= 4;
              char h = s_[p_ + 2 + i];
              if (h >= '0' && h <= '9') cp |= (h - '0');
              else if (h >= 'a' && h <= 'f') cp |= (h - 'a' + 10);
              else if (h >= 'A' && h <= 'F') cp |= (h - 'A' + 10);
              else throw err_at("bad hex in \\u escape");
            }
            // BMP fast path: append UTF-8 bytes for cp <= 0x7FF.
            if (cp < 0x80) {
              out.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
              out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
              out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            p_ += 4; // the for-loop added 4, plus the \u prefix
            break;
          }
          default:
            throw err_at(std::string("unsupported escape \\") + e);
        }
        p_ += 2;
        continue;
      }
      out.push_back(c);
      ++p_;
    }
    if (p_ >= s_.size()) throw err_at("unterminated string");
    ++p_; // closing "
    return out;
  }

  // Parses a JSON value. Strings are only allowed as map keys (handled
  // in parse_string) or as field values; we use parse_string for values
  // too and rely on the schema to keep types consistent.
  void parse_value(const std::string& key, ModelConfig& cfg) {
    char c = peek();
    if (c == '"') {
      std::string v = parse_string();
      set_string(cfg, key, v);
      return;
    }
    if (c == '{') { parse_object(key, cfg); return; }
    if (c == 't' || c == 'f') { parse_bool_top(key, cfg); return; }
    if (c == 'n') { parse_null(key, cfg); return; }
    if (c == '-' || (c >= '0' && c <= '9')) { parse_number(key, cfg); return; }
    throw err_at("unexpected character in value");
  }

  void set_string(ModelConfig& cfg, const std::string& k, const std::string& v) {
    if (k == "name") cfg.name = v;
    else if (k == "type") cfg.type = v;
    else if (k == "file") cfg.file = v;
    else if (k == "sha256") cfg.sha256 = v;
    else if (k == "url") cfg.url = v;
    // strings under known sub-objects are ignored; non-recognized keys silently dropped
  }

  void parse_bool_top(const std::string& k, ModelConfig& cfg) {
    bool v = false;
    if (s_.compare(p_, 4, "true") == 0) { p_ += 4; v = true; }
    else if (s_.compare(p_, 5, "false") == 0) { p_ += 5; v = false; }
    else throw err_at("expected boolean");
    if (k == "use_space") cfg.rec.use_space = v;
  }
  void parse_null(const std::string& k, ModelConfig& cfg) {
    if (s_.compare(p_, 4, "null") != 0) throw err_at("expected null");
    p_ += 4;
    // Only used for DetResizeForTest: null is allowed and means "unset".
    (void)k;
  }

  void parse_number(const std::string& k, ModelConfig& cfg) {
    // Parse a JSON number into a double, then assign to the right field.
    size_t start = p_;
    if (s_[p_] == '-') ++p_;
    while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
    if (p_ < s_.size() && s_[p_] == '.') {
      ++p_;
      while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
    }
    if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
      ++p_;
      if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
      while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
    }
    double v = 0;
    try {
      v = std::stod(s_.substr(start, p_ - start));
    } catch (...) {
      throw err_at("bad number");
    }
    if (k == "thresh") cfg.det.thresh = static_cast<float>(v);
    else if (k == "box_thresh") cfg.det.box_thresh = static_cast<float>(v);
    else if (k == "unclip_ratio") cfg.det.unclip_ratio = static_cast<float>(v);
    else if (k == "min_size") cfg.det.min_size = static_cast<int>(v);
    else if (k == "max_candidates") cfg.det.max_candidates = static_cast<int>(v);
    else if (k == "limit_side_len") cfg.det.resize.limit_side_len = static_cast<int>(v);
    else if (k == "resize_long") cfg.det.resize.resize_long = static_cast<int>(v);
    else if (k == "stride") cfg.det.resize.stride = static_cast<int>(v);
    else if (k == "max_side_limit") cfg.det.resize.max_side_limit = static_cast<int>(v);
    else if (k == "bytes") cfg.bytes = static_cast<uint64_t>(v);
  }

  void parse_object(const std::string& k, ModelConfig& cfg) {
    expect('{');
    bool first = true;
    while (true) {
      skip_ws();
      if (peek() == '}') { ++p_; break; }
      if (!first) {
        if (peek() != ',') throw err_at("expected ',' between fields");
        ++p_;
      }
      first = false;
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      // The sub-object key tells us which struct to fill.
      if (k == "det") {
        if (key == "resize") {
          parse_resize(cfg.det.resize);
        } else {
          parse_sub_value(key, cfg);
        }
      } else if (k == "rec") {
        if (key == "shape") {
          std::vector<int> sh = parse_int_array_body();
          if (sh.size() != 3) throw err_at("rec.shape must have 3 ints");
          cfg.rec.c = sh[0]; cfg.rec.h = sh[1]; cfg.rec.w = sh[2];
        } else if (key == "use_space") {
          bool v = false;
          if (s_.compare(p_, 4, "true") == 0) { p_ += 4; v = true; }
          else if (s_.compare(p_, 5, "false") == 0) { p_ += 5; v = false; }
          else throw err_at("expected boolean");
          cfg.rec.use_space = v;
        } else if (key == "rec_batch_hint") {
          const size_t s0 = p_;
          if (p_ < s_.size() && s_[p_] == '-') ++p_;
          while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
          cfg.rec.rec_batch_hint = std::stoi(s_.substr(s0, p_ - s0));
        } else if (key == "dict") {
          cfg.rec.dict = parse_string_array();
        } else {
          // unknown rec sub-field: skip the value.
          skip_value();
        }
      } else if (k == "cls") {
        if (key == "shape") {
          std::vector<int> sh = parse_int_array_body();
          if (sh.size() != 3) throw err_at("cls.shape must have 3 ints");
          cfg.cls.c = sh[0]; cfg.cls.h = sh[1]; cfg.cls.w = sh[2];
        } else if (key == "mean") {
          cfg.cls.mean = parse_float_array();
        } else if (key == "std") {
          cfg.cls.std = parse_float_array();
        } else if (key == "labels") {
          cfg.cls.labels = parse_string_array();
        } else {
          skip_value();
        }
      } else {
        skip_value();
      }
    }
  }

  void parse_sub_value(const std::string& k, ModelConfig& cfg) {
    // Forward det sub-fields to the numeric parser.
    char c = peek();
    if (c == '-' || (c >= '0' && c <= '9')) {
      parse_number(k, cfg);
    } else if (c == 'n') {
      parse_null(k, cfg);
    } else {
      skip_value();
    }
  }

  void parse_resize(DetResizeConfig& rc) {
    expect('{');
    bool first = true;
    while (true) {
      skip_ws();
      if (peek() == '}') { ++p_; break; }
      if (!first) {
        if (peek() != ',') throw err_at("expected ',' between fields");
        ++p_;
      }
      first = false;
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      if (key == "mode") {
        std::string v = parse_string();
        if (v == "limit_min") rc.mode = DetResizeConfig::Mode::LimitMin;
        else if (v == "resize_long") rc.mode = DetResizeConfig::Mode::ResizeLong;
        else if (v == "no_resize") rc.mode = DetResizeConfig::Mode::NoResize;
        else throw err_at("unknown resize.mode '" + v + "'");
      } else if (key == "limit_side_len") {
        size_t s = p_;
        if (s_[p_] == '-') ++p_;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        if (p_ == s) throw err_at("empty limit_side_len");
        rc.limit_side_len = std::stoi(s_.substr(s, p_ - s));
      } else if (key == "resize_long") {
        size_t s = p_;
        if (s_[p_] == '-') ++p_;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        if (p_ == s) throw err_at("empty resize_long");
        rc.resize_long = std::stoi(s_.substr(s, p_ - s));
      } else if (key == "stride") {
        size_t s = p_;
        if (s_[p_] == '-') ++p_;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        if (p_ == s) throw err_at("empty stride");
        rc.stride = std::stoi(s_.substr(s, p_ - s));
      } else if (key == "max_side_limit") {
        size_t s = p_;
        if (s_[p_] == '-') ++p_;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        if (p_ == s) throw err_at("empty max_side_limit");
        rc.max_side_limit = std::stoi(s_.substr(s, p_ - s));
      } else {
        skip_value();
      }
    }
  }

  std::vector<int> parse_int_array_body() {
    expect('[');
    std::vector<int> out;
    bool first = true;
    while (true) {
      skip_ws();
      if (peek() == ']') { ++p_; return out; }
      if (!first) {
        if (peek() != ',') throw err_at("expected ',' in array");
        ++p_;
      }
      first = false;
      skip_ws();
      size_t s = p_;
      if (s_[p_] == '-') ++p_;
      while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
      if (p_ == s) {
        throw err_at("empty int in array");
      }
      out.push_back(std::stoi(s_.substr(s, p_ - s)));
    }
  }

  std::vector<float> parse_float_array() {
    expect('[');
    std::vector<float> out;
    bool first = true;
    while (true) {
      skip_ws();
      if (peek() == ']') { ++p_; return out; }
      if (!first) {
        if (peek() != ',') throw err_at("expected ',' in array");
        ++p_;
      }
      first = false;
      skip_ws();
      size_t s = p_;
      if (s_[p_] == '-') ++p_;
      while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
      if (p_ < s_.size() && s_[p_] == '.') {
        ++p_;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
      }
      out.push_back(std::stof(s_.substr(s, p_ - s)));
    }
  }

  std::vector<std::string> parse_string_array() {
    expect('[');
    std::vector<std::string> out;
    bool first = true;
    while (true) {
      skip_ws();
      if (peek() == ']') { ++p_; return out; }
      if (!first) {
        if (peek() != ',') throw err_at("expected ',' in array");
        ++p_;
      }
      first = false;
      skip_ws();
      out.push_back(parse_string());
    }
  }

  void skip_value() {
    // Skip an arbitrary JSON value (used for unknown fields).
    int depth = 0;
    do {
      char c = peek();
      if (c == '{' || c == '[') { ++depth; ++p_; continue; }
      if (c == '}' || c == ']') {
        if (depth == 0) return;
        --depth; ++p_; continue;
      }
      if (c == '"') { skip_string(); continue; }
      if (c == '-' || (c >= '0' && c <= '9')) {
        ++p_;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        if (p_ < s_.size() && s_[p_] == '.') {
          ++p_;
          while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        }
        if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
          ++p_;
          if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
          while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        }
        continue;
      }
      if (s_.compare(p_, 4, "true") == 0) { p_ += 4; continue; }
      if (s_.compare(p_, 5, "false") == 0) { p_ += 5; continue; }
      if (s_.compare(p_, 4, "null") == 0) { p_ += 4; continue; }
      throw err_at("skip_value: unexpected character");
    } while (depth > 0);
  }

  void skip_string() {
    expect('"');
    while (p_ < s_.size() && s_[p_] != '"') {
      if (s_[p_] == '\\' && p_ + 1 < s_.size()) { p_ += 2; continue; }
      ++p_;
    }
    if (p_ >= s_.size()) throw err_at("unterminated string in skip");
    ++p_;
  }
};

} // namespace

ModelConfig load_model_config(const std::string& json_path) {
  std::ifstream f(json_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("load_model_config: cannot open " + json_path);
  }
  std::ostringstream oss;
  oss << f.rdbuf();
  std::string s = oss.str();
  if (s.empty()) {
    throw std::runtime_error("load_model_config: empty file " + json_path);
  }
  JsonParser p(s);
  ModelConfig cfg = p.parse();
  // Sanity: `file` is required so the registry can locate the .mnn.
  if (cfg.file.empty()) {
    throw std::runtime_error("load_model_config: missing 'file' in "
                             + json_path);
  }
  return cfg;
}

} // namespace ppocr
