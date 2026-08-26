// M2-FINAL-DIAG: dump every op's output tensor using runSessionWithCallBack.
//
// Usage: ./m2fd_dump_mnn <model.mnn> <input.npy> <out.npz> [n_threads]
//
// Reads the input NPY, creates a session, runs the model with the
// per-op callback enabled, and writes each op's output tensor to a
// .npz file. The mapping from entry name to op name is written to
// <out.npz>.json.

#include "MNN/Interpreter.hpp"
#include "MNN/MNNForwardType.h"
#include "MNN/Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static bool read_npy_float32(const char* path, std::vector<float>& out,
                              std::vector<int>& shape) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "fopen %s failed\n", path); return false; }
  char magic[6]; if (std::fread(magic, 1, 6, f) != 6) { fclose(f); return false; }
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    std::fprintf(stderr, "%s: not a NPY file\n", path); fclose(f); return false;
  }
  unsigned char ver[2]; if (std::fread(ver, 1, 2, f) != 2) { fclose(f); return false; }
  unsigned short hlen = 0;
  if (ver[0] == 1) {
    if (std::fread(&hlen, 2, 1, f) != 1) { fclose(f); return false; }
  } else if (ver[0] >= 2) {
    unsigned int hlen4 = 0;
    if (std::fread(&hlen4, 4, 1, f) != 1) { fclose(f); return false; }
    hlen = (unsigned short)hlen4;
  }
  std::vector<char> header(hlen + 1, 0);
  if (std::fread(header.data(), 1, hlen, f) != hlen) { fclose(f); return false; }
  std::string h(header.data());
  auto pos = h.find("'shape':");
  if (pos == std::string::npos) pos = h.find("\"shape\":");
  if (pos == std::string::npos) { fclose(f); return false; }
  pos = h.find('(', pos);
  if (pos == std::string::npos) { fclose(f); return false; }
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
  size_t total = 1;
  for (int s : shape) total *= (size_t)s;
  out.resize(total);
  if (std::fread(out.data(), sizeof(float), total, f) != total) {
    std::fprintf(stderr, "%s: short read\n", path); fclose(f); return false;
  }
  fclose(f);
  return true;
}

struct NpzEntry {
  std::string name;
  std::vector<float> data;
  std::vector<int> shape;
};
static std::vector<NpzEntry> g_entries;

static void append_tensor(const std::string& orig_name, const float* data,
                           const std::vector<int>& shape) {
  size_t total = 1;
  for (int s : shape) total *= (size_t)s;
  NpzEntry e;
  e.name = orig_name;
  e.data.assign(data, data + total);
  e.shape = shape;
  g_entries.push_back(std::move(e));
}

static uint32_t crc32(const void* data, size_t len) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    init = true;
  }
  uint32_t c = 0xFFFFFFFFu;
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

static void write_npy_in_zip(FILE* zip, const std::string& entry_name,
                              const std::vector<float>& data,
                              const std::vector<int>& shape) {
  std::string shape_str = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) shape_str += ", ";
    shape_str += std::to_string(shape[i]);
  }
  shape_str += ")";
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " +
                     shape_str + ", }";
  int hlen_raw = 6 + 1 + 2 + (int)dict.size() + 1;
  int pad = (64 - (hlen_raw % 64)) % 64;
  for (int i = 0; i < pad; ++i) dict += ' ';
  dict += '\n';
  std::string npy = std::string("\x93NUMPY", 6) + std::string(1, 1) + std::string(1, 0) +
                    std::string(1, (dict.size() >> 0) & 0xFF) +
                    std::string(1, (dict.size() >> 8) & 0xFF) +
                    dict;
  size_t total_bytes = data.size() * sizeof(float);

  uint32_t crc_ = crc32(data.data(), total_bytes);
  uint32_t comp_size = total_bytes;
  uint32_t uncomp_size = total_bytes;
  std::fwrite("PK\x03\x04", 1, 4, zip);
  uint16_t v = 20; std::fwrite(&v, 2, 1, zip);
  uint16_t flags = 0; std::fwrite(&flags, 2, 1, zip);
  uint16_t method = 0; std::fwrite(&method, 2, 1, zip);
  uint16_t mtime = 0, mdate = 0; std::fwrite(&mtime, 2, 1, zip); std::fwrite(&mdate, 2, 1, zip);
  std::fwrite(&crc_, 4, 1, zip);
  std::fwrite(&comp_size, 4, 1, zip);
  std::fwrite(&uncomp_size, 4, 1, zip);
  uint16_t name_len = entry_name.size(); std::fwrite(&name_len, 2, 1, zip);
  uint16_t extra_len = 0; std::fwrite(&extra_len, 2, 1, zip);
  std::fwrite(entry_name.data(), 1, name_len, zip);
  std::fwrite(data.data(), 1, total_bytes, zip);
  (void)npy;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model.mnn> <input.npy> <out.npz> [n_threads=4]\n", argv[0]);
    return 99;
  }
  const char* model_path = argv[1];
  const char* input_path = argv[2];
  const char* out_path = argv[3];
  int n_threads = argc > 4 ? std::atoi(argv[4]) : 4;

  std::vector<float> input;
  std::vector<int> input_shape;
  if (!read_npy_float32(input_path, input, input_shape)) {
    std::fprintf(stderr, "read_npy failed\n"); return 10;
  }
  if (input_shape.size() == 3) input_shape.insert(input_shape.begin(), 1);

  MNN::Interpreter* interp = MNN::Interpreter::createFromFile(model_path);
  if (!interp) { std::fprintf(stderr, "createFromFile failed\n"); return 20; }

  MNN::ScheduleConfig sc;
  sc.type = MNN_FORWARD_CPU;
  sc.numThread = n_threads;
  auto* bc = new MNN::BackendConfig();
  bc->precision = MNN::BackendConfig::Precision_Normal;
  sc.backendConfig = bc;

  MNN::Session* session = interp->createSession(sc);
  if (!session) { std::fprintf(stderr, "createSession failed\n"); return 21; }

  auto input_tensor = interp->getSessionInput(session, nullptr);
  if (!input_tensor) { std::fprintf(stderr, "no input\n"); return 22; }
  interp->resizeTensor(input_tensor, input_shape);
  interp->resizeSession(session);
  std::memcpy(input_tensor->host<float>(), input.data(),
              input.size() * sizeof(float));

  // After-callback: dump each op's outputs.
  int op_count = 0;
  auto after = [&](const std::vector<MNN::Tensor*>& tensors, const std::string& opName) -> bool {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "op%04d_%s", op_count, opName.c_str());
    std::string safe = buf;
    for (auto& c : safe) if (c == '/' || c == ' ') c = '_';
    // tensors is a vector of output tensors for this op
    if (tensors.empty()) { op_count++; return true; }
    // For now, just dump the first output (most ops have 1)
    MNN::Tensor* t = tensors[0];
    MNN::Tensor host_tensor(t, t->getDimensionType());
    if (t->getDimensionType() == MNN::Tensor::CAFFE) {
      t->copyToHostTensor(&host_tensor);
    } else {
      std::memcpy(host_tensor.buffer().host, t->host<float>(), t->size());
    }
    std::vector<int> out_shape;
    auto oshape = t->shape();
    for (int s : oshape) out_shape.push_back(s);
    append_tensor(safe, host_tensor.host<float>(), out_shape);
    op_count++;
    return true;
  };
  auto before = [](const std::vector<MNN::Tensor*>& tensors,
                    const std::string& opName) -> bool { return true; };

  MNN::ErrorCode rc = interp->runSessionWithCallBack(session, before, after, false);
  if (rc != MNN::NO_ERROR) { std::fprintf(stderr, "runSession rc=%d\n", rc); return 23; }

  // Also save the input for sanity
  {
    std::vector<int> out_shape;
    auto oshape = input_tensor->shape();
    for (int s : oshape) out_shape.push_back(s);
    append_tensor("op_input", input_tensor->host<float>(), out_shape);
  }

  std::printf("collected %zu entries from %d ops\n", g_entries.size(), op_count);

  // Use Python's zipfile module to write a proper .npz (with central
  // directory). Build a Python script that takes the entries from a
  // sidecar JSON, plus a .npy file per entry.
  //
  // Simpler: dump the entries to a sidecar .bin + sidecar .json
  // describing shapes, and let the Python side assemble the npz.
  std::string bin_path = std::string(out_path) + ".bin";
  std::string json_path = std::string(out_path) + ".json";
  FILE* bf = std::fopen(bin_path.c_str(), "wb");
  if (!bf) { std::fprintf(stderr, "fopen %s failed\n", bin_path.c_str()); return 30; }
  // Header: # entries as uint64, then per-entry: name_len (uint32),
  // name bytes, shape_ndim (uint32), shape[ndim] as int32, then
  // the float32 data blob.
  uint64_t n = g_entries.size();
  std::fwrite(&n, 8, 1, bf);
  // Two-pass: write offsets first.
  std::vector<long> offsets; offsets.reserve(n);
  std::vector<uint32_t> sizes; sizes.reserve(n);
  for (auto& e : g_entries) {
    uint32_t name_len = e.name.size();
    std::fwrite(&name_len, 4, 1, bf);
    std::fwrite(e.name.data(), 1, name_len, bf);
    uint32_t ndim = e.shape.size();
    std::fwrite(&ndim, 4, 1, bf);
    if (ndim) std::fwrite(e.shape.data(), 4, ndim, bf);
    offsets.push_back(std::ftell(bf));
    uint32_t n_floats = e.data.size();
    std::fwrite(e.data.data(), 4, n_floats, bf);
    sizes.push_back(n_floats);
  }
  std::fclose(bf);
  // JSON: list of {name, shape: [..], n_floats: .., offset: .., nbytes: ..}
  std::ofstream jf(json_path);
  jf << "[\n";
  for (size_t i = 0; i < g_entries.size(); ++i) {
    jf << "  {\"name\": \"" << g_entries[i].name << "\", \"shape\": [";
    for (size_t k = 0; k < g_entries[i].shape.size(); ++k) {
      if (k) jf << ", ";
      jf << g_entries[i].shape[k];
    }
    jf << "], \"n_floats\": " << sizes[i] << ", \"offset\": " << offsets[i] << "}";
    if (i + 1 < g_entries.size()) jf << ",";
    jf << "\n";
  }
  jf << "]\n";
  jf.close();
  std::printf("wrote %s + %s (n=%zu)\n", bin_path.c_str(), json_path.c_str(), g_entries.size());
  return 0;
}
