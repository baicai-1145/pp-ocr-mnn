// POST-6 backend-precision diff driver.
//
// Runs v6_tiny_det on a fixed input (3, H, W) under each
// MNN ScheduleConfig backend + precision combo, dumps the
// (1, 1, H_out, W_out) prob map to a .npy file per combo, and
// prints a single-line diff table vs the Paddle reference
// (assumed to be at the path in argv[5] when supplied).
//
// Inputs (positional CLI args):
//   argv[1]  = path to .mnn
//   argv[2]  = path to input .npy  (float32, shape (3, H, W))
//   argv[3]  = output prefix; we write <prefix>__cpu_normal.npy etc.
//   argv[4]  = number of threads (0 = MNN default)
//   argv[5]  (optional) = path to Paddle reference output for diff
//
// Backends / precision tested:
//   cpu_normal     MNN_FORWARD_CPU + BackendConfig::Precision_Normal
//   cpu_high       MNN_FORWARD_CPU + BackendConfig::Precision_High
//   cpu_low        MNN_FORWARD_CPU + BackendConfig::Precision_Low
//                  (FP16 arithmetic on ARM; on x86 typically no-op)
//   cpu_low_bf16   MNN_FORWARD_CPU + BackendConfig::Precision_Low_BF16
//                  (BF16 accumulator; only if MNN_SUPPORT_BF16=ON at
//                   MNN build time; otherwise silently falls back to
//                   the default CPUBackend with no actual BF16)
//
// Note: SSE/AVX code paths are selected at COMPILE time of MNN; the
// `precision` config only changes a handful of activation functions
// (sigmoid, exp, etc.). Most of the det network is conv/matmul, which
// is unaffected. So if the MNN/Paddle diff is dominated by conv
// rounding, this driver will not move the needle.
#include "MNN/Interpreter.hpp"
#include "MNN/MNNForwardType.h"
#include "MNN/Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_npy_float32(const char* path, std::vector<float>& out,
                              std::vector<int>& shape) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "fopen %s failed\n", path); return false; }
  // Minimal NPY reader: 6-byte magic + 1-byte version + 2-byte header_len
  char magic[6]; if (std::fread(magic, 1, 6, f) != 6) { fclose(f); return false; }
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    std::fprintf(stderr, "%s: not a NPY file\n", path); fclose(f); return false;
  }
  unsigned char ver[2]; if (std::fread(ver, 1, 2, f) != 2) { fclose(f); return false; }
  unsigned short hlen = 0;
  if (ver[0] == 1) {
    if (std::fread(&hlen, 2, 1, f) != 1) { fclose(f); return false; }
  } else if (ver[0] >= 2) {
    // v2 / v3: 4-byte header length, little-endian
    unsigned int hlen4 = 0;
    if (std::fread(&hlen4, 4, 1, f) != 1) { fclose(f); return false; }
    hlen = (unsigned short)hlen4;
  } else {
    std::fprintf(stderr, "%s: unsupported NPY version %d\n", path, ver[0]);
    fclose(f); return false;
  }
  // header dict is hlen bytes
  std::vector<char> header(hlen + 1, 0);
  if (std::fread(header.data(), 1, hlen, f) != hlen) { fclose(f); return false; }
  std::string h(header.data());
  // Find shape
  auto pos = h.find("'shape':");
  if (pos == std::string::npos) pos = h.find("\"shape\":");
  if (pos == std::string::npos) {
    std::fprintf(stderr, "%s: no 'shape' in header\n", path);
    fclose(f); return false;
  }
  pos = h.find('(', pos);
  if (pos == std::string::npos) {
    std::fprintf(stderr, "%s: no '(' for shape\n", path);
    fclose(f); return false;
  }
  pos++;
  while (pos < h.size() && h[pos] != ')') {
    int v = 0; int sign = 1;
    if (h[pos] == '-') { sign = -1; pos++; }
    bool any = false;
    while (pos < h.size() && h[pos] >= '0' && h[pos] <= '9') {
      v = v * 10 + (h[pos] - '0');
      pos++; any = true;
    }
    if (any) shape.push_back(sign * v);
    while (pos < h.size() && (h[pos] == ',' || h[pos] == ' ')) pos++;
  }
  // Data starts at the current file position (after header dict).
  size_t total = 1;
  for (int s : shape) total *= (size_t)s;
  out.resize(total);
  if (std::fread(out.data(), sizeof(float), total, f) != total) {
    std::fprintf(stderr, "%s: short read (%zu floats, expected %zu)\n",
                 path, total, total);
    fclose(f); return false;
  }
  fclose(f);
  return true;
}

static bool write_npy_float32(const char* path, const float* data,
                               const std::vector<int>& shape) {
  FILE* f = std::fopen(path, "wb");
  if (!f) { std::fprintf(stderr, "fopen %s failed\n", path); return false; }
  // NPY v1 header for 'f4', shape tuple
  std::string shape_str = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) shape_str += ", ";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += ")";
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
  // Pad to multiple of 64 with spaces
  int hlen_raw = 6 + 1 + 2 + (int)dict.size() + 1;  // magic + ver + hdrlen + dict + newline
  int pad = (64 - (hlen_raw % 64)) % 64;
  for (int i = 0; i < pad; ++i) dict += ' ';
  dict += '\n';
  unsigned short hlen = (unsigned short)dict.size();
  std::fwrite("\x93NUMPY", 1, 6, f);
  unsigned char ver[2] = {1, 0};
  std::fwrite(ver, 1, 2, f);
  std::fwrite(&hlen, 2, 1, f);
  std::fwrite(dict.data(), 1, hlen, f);
  size_t total = 1;
  for (int s : shape) total *= (size_t)s;
  std::fwrite(data, sizeof(float), total, f);
  fclose(f);
  return true;
}

struct Combo {
  const char* name;
  MNNForwardType type;
  MNN::BackendConfig::PrecisionMode prec;
};

static int run_combo(MNN::Interpreter* interp, const std::string& output_name,
                     const Combo& c, int num_threads,
                     const std::vector<float>& input,
                     const std::vector<int>& input_shape, const char* out_path) {
  MNN::ScheduleConfig sc;
  sc.type = c.type;
  if (num_threads > 0) sc.numThread = num_threads;
  // Heap-allocate the BackendConfig so the pointer remains valid for
  // the session's lifetime (MNN may cache the pointer).
  auto* bc = new MNN::BackendConfig();
  bc->precision = c.prec;
  sc.backendConfig = bc;

  MNN::Session* session = interp->createSession(sc);
  if (!session) {
    std::fprintf(stderr, "[%s] createSession failed\n", c.name);
    return 1;
  }
  auto input_tensor = interp->getSessionInput(session, nullptr);
  if (!input_tensor) {
    std::fprintf(stderr, "[%s] no input tensor\n", c.name);
    return 2;
  }
  // The input tensor shape may have been inferred from the model. We
  // need to resize it to match our actual input.
  interp->resizeTensor(input_tensor, input_shape);
  interp->resizeSession(session);
  std::memcpy(input_tensor->host<float>(), input.data(),
              input.size() * sizeof(float));

  // Run
  int rc = interp->runSession(session);
  if (rc != 0) {
    std::fprintf(stderr, "[%s] runSession rc=%d\n", c.name, rc);
    return 3;
  }
  auto output = interp->getSessionOutput(session, output_name.c_str());
  if (!output) {
    std::fprintf(stderr, "[%s] no output '%s'\n", c.name, output_name.c_str());
    return 4;
  }
  std::vector<int> out_shape;
  auto oshape = output->shape();
  for (int i = 0; i < oshape.size(); ++i) out_shape.push_back(oshape[i]);
  size_t total = 1;
  for (int s : out_shape) total *= (size_t)s;
  std::vector<float> out(total);
  std::memcpy(out.data(), output->host<float>(), total * sizeof(float));
  if (!write_npy_float32(out_path, out.data(), out_shape)) {
    std::fprintf(stderr, "[%s] write_npy failed\n", c.name);
    return 5;
  }
  std::printf("[%s] wrote %s shape=(", c.name, out_path);
  for (int s : out_shape) std::printf("%d ", s);
  std::printf(")  min=%.4f max=%.4f\n", out.front(), out.back());
  interp->releaseSession(session);
  return 0;
}

// Compute diff statistics between two same-shaped float arrays.
struct DiffStats {
  double max_abs;
  double mean_abs;
  double pct_gt_001;  // fraction of pixels with |diff| > 0.01
  double pct_gt_01;   // fraction of pixels with |diff| > 0.1
  int n;
};

static DiffStats diff_stats(const std::vector<float>& a, const std::vector<float>& b) {
  DiffStats s{0.0, 0.0, 0.0, 0.0, 0};
  if (a.size() != b.size()) return s;
  s.n = (int)a.size();
  double sum = 0.0;
  int gt001 = 0, gt01 = 0;
  float amax = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float d = std::fabs(a[i] - b[i]);
    if (d > amax) amax = d;
    sum += d;
    if (d > 0.01f) gt001++;
    if (d > 0.1f)  gt01++;
  }
  s.max_abs = amax;
  s.mean_abs = sum / (double)s.n;
  s.pct_gt_001 = 100.0 * gt001 / s.n;
  s.pct_gt_01  = 100.0 * gt01  / s.n;
  return s;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s <model.mnn> <input.npy> <out_prefix> <num_threads> [paddle_ref.npy]\n",
                 argv[0]);
    return 99;
  }
  const char* model_path = argv[1];
  const char* input_path = argv[2];
  const char* out_prefix = argv[3];
  int num_threads = std::atoi(argv[4]);
  const char* paddle_ref_path = (argc >= 6) ? argv[5] : nullptr;

  // Load input
  std::vector<float> input;
  std::vector<int> input_shape;
  if (!read_npy_float32(input_path, input, input_shape)) {
    std::fprintf(stderr, "read_npy %s failed\n", input_path);
    return 10;
  }
  // The det model expects NCHW with a batch dim. The Paddle input dump
  // may be saved as plain CHW; prepend a batch dim if needed.
  if (input_shape.size() == 3) {
    input_shape.insert(input_shape.begin(), 1);
  }

  // Create interpreter
  MNN::Interpreter* interp = MNN::Interpreter::createFromFile(model_path);
  if (!interp) {
    std::fprintf(stderr, "createFromFile failed\n");
    return 20;
  }

  // Find output names from a temp session (kept alive for the
  // duration of the test; MNN's tensorMap entries point into the
  // session's storage, so don't delete it).
  MNN::ScheduleConfig default_sc;
  MNN::Session* session0 = interp->createSession(default_sc);
  if (!session0) { std::fprintf(stderr, "temp session failed\n"); return 21; }
  std::string output_name;
  {
    auto outs = interp->getSessionOutputAll(session0);
    if (outs.empty()) {
      std::fprintf(stderr, "no output names\n");
      return 22;
    }
    output_name = outs.begin()->first;
  }
  std::printf("model output name: %s\n", output_name.c_str());

  // Combos to test.
  std::vector<Combo> combos = {
    {"cpu_normal",  MNN_FORWARD_CPU, MNN::BackendConfig::Precision_Normal},
    {"cpu_high",    MNN_FORWARD_CPU, MNN::BackendConfig::Precision_High},
    {"cpu_low",     MNN_FORWARD_CPU, MNN::BackendConfig::Precision_Low},
    {"cpu_low_bf16",MNN_FORWARD_CPU, MNN::BackendConfig::Precision_Low_BF16},
  };

  // Run all combos
  for (auto& c : combos) {
    char out_path[1024];
    std::snprintf(out_path, sizeof(out_path), "%s__%s.npy", out_prefix, c.name);
    int rc = run_combo(interp, output_name, c, num_threads, input, input_shape, out_path);
    if (rc != 0) {
      std::fprintf(stderr, "[%s] rc=%d\n", c.name, rc);
    }
  }

  // Optional: diff each combo vs the Paddle reference.
  if (paddle_ref_path) {
    std::vector<float> ref;
    std::vector<int> ref_shape;
    if (!read_npy_float32(paddle_ref_path, ref, ref_shape)) {
      std::fprintf(stderr, "read_npy %s failed\n", paddle_ref_path);
    } else {
      std::printf("\n=== Diff vs %s (shape (", paddle_ref_path);
      for (int s : ref_shape) std::printf("%d ", s);
      std::printf(")) ===\n");
      std::printf("%-14s  %10s  %10s  %10s  %10s\n",
                  "combo", "max_abs", "mean_abs", "%>0.01", "%>0.1");
      for (auto& c : combos) {
        char out_path[1024];
        std::snprintf(out_path, sizeof(out_path), "%s__%s.npy", out_prefix, c.name);
        std::vector<float> arr;
        std::vector<int> ashape;
        if (!read_npy_float32(out_path, arr, ashape)) {
          std::fprintf(stderr, "  %s: failed to re-read\n", c.name);
          continue;
        }
        auto s = diff_stats(arr, ref);
        std::printf("%-14s  %10.6f  %10.6f  %9.2f%%  %9.2f%%\n",
                    c.name, s.max_abs, s.mean_abs, s.pct_gt_001, s.pct_gt_01);
      }
    }
  }

  delete interp;
  return 0;
}
