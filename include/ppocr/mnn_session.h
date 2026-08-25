// pp-ocr-mnn — MNN session wrapper (owner m1)
//
// Thin RAII over MNN::Interpreter + a single inference Session. One
// Session = one .mnn file. Multiple MnnSession instances are independent
// (no global state), so the public C ABI can hand out per-engine sessions
// and remain thread-safe.
//
// Why hand-rolled instead of using the express module directly: the express
// header pulls in protobuf-linked headers even on the read path. We only
// need the static Interpreter API for M1, which keeps our .o files slim
// and lets us link the small libMNN.a that ships in third_party/MNN/build
// without dragging in libMNNConvertDeps.
#ifndef PPOCR_MNN_SESSION_H_
#define PPOCR_MNN_SESSION_H_

#include <cstdint>
#include <string>
#include <vector>

namespace MNN { class Interpreter; class Session; class Tensor; }

namespace ppocr {

// Opaque-forwarded backend choice. Mirrors the public C ABI enum; the
// conversion to MNNForwardType lives in src/ppocr.cpp::pickBackend.
enum class Backend { Auto, Cpu, Cuda, OpenCL, Vulkan, Metal, CoreML, NNAPI };

struct SessionConfig {
  Backend backend = Backend::Auto;
  int     num_threads = 0;  // 0 = MNN default
  // power: 0 = high (default), 1 = low (used to avoid throttling on phones)
  int     power = 0;
  // precision: 0 = normal (FP16 on GPU when available), 1 = low (FP16 everywhere)
  int     precision = 0;
  // memory: 0 = normal, 1 = low (used for memory-tight devices)
  int     memory_mode = 0;
};

// Output tensor reference. The float pointer is valid as long as the
// parent MnnSession is alive and `run()` has been called at least once.
struct SessionOutput {
  const float* data = nullptr; // host pointer, length = product of `shape`
  std::vector<int> shape;      // NCHW layout as reported by MNN
};

// Forward declaration of the implementation. The full definition lives
// in src/mnn_session.cpp; the public ABI only needs a pointer-sized
// handle, so we keep MNN's headers out of every translation unit that
// includes mnn_session.h.
struct MnnSessionImpl;

// One MNN inference session bound to a single .mnn file. Not copyable:
// MNN::Interpreter owns the underlying buffer pool, and copying the
// handle would either double-free or alias. Move is fine.
class MnnSession {
 public:
  MnnSession();
  ~MnnSession();

  MnnSession(const MnnSession&) = delete;
  MnnSession& operator=(const MnnSession&) = delete;
  MnnSession(MnnSession&& other) noexcept;
  MnnSession& operator=(MnnSession&& other) noexcept;

  // Load `model_path` (.mnn) and create a session with the given config.
  // Throws std::runtime_error on failure (file missing, parse error,
  // MNN_NO_EXECUTION, etc.).
  void load(const std::string& model_path, const SessionConfig& cfg);

  // Resize a named input. For dynamic-shape models this is the only way
  // to set the input dims. The input is matched by MNN's `name` field,
  // which is preserved through MNNConvert. `dims` is the full NCHW shape.
  void resize_input(const std::string& name, const std::vector<int>& dims);

  // Read the host pointer for an input tensor. Caller fills NCHW float32
  // data of `size` floats (must equal product of dims from resize_input).
  float* input_host(const std::string& name);
  // Same for an output tensor after run().
  const float* output_host(const std::string& name) const;

  // Convenience: resize the named input and copy `data` (length = product
  // of `dims`) into its host buffer in one step. Use when the dims are
  // known up front and the caller already owns a contiguous float buffer.
  void set_input_float(const std::string& name,
                       const std::vector<int>& dims,
                       const float* data);

  // Synchronous inference. Returns MNN::NO_ERROR on success; otherwise
  // an MNN::ErrorCode value. Throws nothing — callers branch on the
  // return value.
  int run();

  // Snapshot an output tensor. The returned SessionOutput references
  // MNN-owned memory; the caller must consume it before the next run().
  SessionOutput output(const std::string& name) const;

  // Returns the resolved MNNForwardType string ("cpu", "opencl", ...)
  // for the active session. Useful for the C ABI to populate
  // result.backend_used.
  const char* backend_name() const;

 private:
  MnnSessionImpl* impl_ = nullptr;
};

} // namespace ppocr
#endif // PPOCR_MNN_SESSION_H_
