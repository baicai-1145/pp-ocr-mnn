// pp-ocr-mnn — MNN session wrapper implementation
//
// Single-threaded around one MNN::Interpreter + one Session. The session
// can be moved (which transfers ownership of the underlying buffer pool),
// but never copied: MNN does not document a copy constructor and aliasing
// the same buffer pool would corrupt subsequent runSession calls.
//
// Why a PIMPL: MNN's public Interpreter.hpp drags in <map> and <memory>
// but is otherwise header-light. The PIMPL keeps the ABI of the wrapper
// small and lets us swap the backend in src/ppocr.cpp::pickBackend without
// changing the MnnSession API.
#include "ppocr/mnn_session.h"

#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppocr {

// Debug helper: PPOCR_DUMP_INPUT=<prefix> writes the raw float32 input
// tensor of every MNN session input to <prefix>.<name>.<n>.<dims>.bin
// right before it is staged (same bytes the session receives). Used to
// build bit-exact minimal repros for backend numerics debugging.
static void dump_input_if_requested(const std::string& name,
                                    const std::vector<int>& dims,
                                    const float* data) {
  const char* prefix = std::getenv("PPOCR_DUMP_INPUT");
  if (!prefix || !data) return;
  static int seq = 0;
  char path[1024];
  std::snprintf(path, sizeof(path), "%s.%s.%d", prefix, name.c_str(), seq++);
  FILE* f = std::fopen(path, "wb");
  if (!f) return;
  int nd = (int)dims.size();
  std::fwrite(&nd, sizeof(int), 1, f);
  std::fwrite(dims.data(), sizeof(int), nd, f);
  size_t n = 1;
  for (int d : dims) n *= (size_t)(d > 0 ? d : 1);
  std::fwrite(data, sizeof(float), n, f);
  std::fclose(f);
}

struct MnnSessionImpl {
  std::unique_ptr<MNN::Interpreter> interp;
  MNN::Session* session = nullptr;          // owned by `interp`
  // Owned BackendConfig for GPU backends. ScheduleConfig::backendConfig is
  // a BORROWED pointer that must outlive the session (writing through the
  // MNN-internal static default segfaults in CUDARuntimeCreator::onCreate,
  // see M2_FINAL_MATRIX.md addendum), so we keep our own copy here.
  MNN::BackendConfig backend_config;
  bool backend_config_set = false;
  MNNForwardType resolved_type = MNN_FORWARD_CPU;
  std::string resolved_name = "cpu";
  // Cached dims per input name, used by run() to verify sizes.
  std::vector<std::vector<int>> input_dims;
  std::vector<std::string>     input_names;
  // Last-resolved input pointers (host floats). These are valid until
  // the next resizeTensor / resizeSession / runSession call.
  std::vector<float*>          input_hosts;
  // PERF: raw MNN::Tensor* per input/output, in the same order as the
  // name vectors above. MNN::Interpreter::getSessionInput/Output take
  // the net-wide mutex (mNet->lock) on EVERY call and re-insert into
  // tensorMap, so calling them in the hot path (set_input_float, run,
  // output) costs a mutex acquire + std::map lookup each time. Caching
  // the resolved pointers once in load() removes that entirely. They are
  // re-resolved after any resizeTensor (buffers may be reallocated).
  std::vector<MNN::Tensor*>    input_tensors;
  std::vector<MNN::Tensor*>    output_tensors;
  // Cached dims per output name.
  std::vector<std::vector<int>> output_dims;
  std::vector<std::string>      output_names;
  std::vector<const float*>     output_hosts;
  // PERF: reusable host-side staging tensors. One per input name, resized
  // only when the requested dims actually change. This removes a
  // MNN::Tensor::create + heap alloc + free from every inference call.
  std::vector<std::unique_ptr<MNN::Tensor>> input_stage;
  // PERF: reusable host-side readback tensor + owned float buffer for
  // output(). MNN::Tensor::createHostTensorFromDevice allocates per call.
  std::unique_ptr<MNN::Tensor> host_out;
  std::vector<int>             host_out_dims;
  bool host_out_valid = false;
  // Owned float snapshot of the last device output. Returned by
  // SessionOutput::data for the caller's lifetime contract (valid until
  // the next output() call on this session).
  std::vector<float>           device_out_cache;
  // Fast size lookup for output(): name -> index into output_* vectors.
  int output_index(const std::string& name) const {
    for (size_t i = 0; i < output_names.size(); ++i) {
      if (output_names[i] == name) return static_cast<int>(i);
    }
    return -1;
  }
};

namespace {

MNNForwardType to_mnn(Backend b) {
  switch (b) {
    case Backend::Auto:   return MNN_FORWARD_AUTO;
    case Backend::Cpu:    return MNN_FORWARD_CPU;
    case Backend::Cuda:   return MNN_FORWARD_CUDA;
    case Backend::OpenCL: return MNN_FORWARD_OPENCL;
    case Backend::Vulkan: return MNN_FORWARD_VULKAN;
    case Backend::Metal:  return MNN_FORWARD_METAL;
    case Backend::CoreML: return MNN_FORWARD_NN;  // CoreML maps to NN slot
    case Backend::NNAPI:  return MNN_FORWARD_NN;
  }
  return MNN_FORWARD_CPU;
}

// MnnSession::backend_name forwards the resolved label. We keep the
// helper in an anonymous namespace but also expose a public shim with
// __attribute__((used)) so the static function survives the
// -Wunused-function pass on its own (it is wired in once M3 turns on
// the gpu backend paths).
__attribute__((used)) static const char* mnn_backend_label(MNNForwardType t) {
  switch (t) {
    case MNN_FORWARD_CPU:           return "cpu";
    case MNN_FORWARD_CPU_EXTENSION: return "cpu";
    case MNN_FORWARD_CUDA:          return "cuda";
    case MNN_FORWARD_OPENCL:        return "opencl";
    case MNN_FORWARD_VULKAN:        return "vulkan";
    case MNN_FORWARD_METAL:         return "metal";
    case MNN_FORWARD_OPENGL:        return "opengl";
    case MNN_FORWARD_NN:            return "nn";
    default:                        return "auto";
  }
}

} // namespace

MnnSession::MnnSession() : impl_(new MnnSessionImpl) {}
MnnSession::~MnnSession() {
  delete impl_;
}

MnnSession::MnnSession(MnnSession&& other) noexcept = default;
MnnSession& MnnSession::operator=(MnnSession&& other) noexcept = default;

void MnnSession::load(const std::string& model_path,
                      const SessionConfig& cfg) {
  impl_->interp.reset(MNN::Interpreter::createFromFile(model_path.c_str()));
  if (!impl_->interp) {
    throw std::runtime_error("MnnSession: createFromFile failed for "
                             + model_path);
  }
  MNN::ScheduleConfig sc;
  sc.type = to_mnn(cfg.backend);
  if (cfg.num_threads > 0) {
    sc.numThread = cfg.num_threads;
  }
  // power / precision / memory are passed through via MNN's options.
  // (MNN 2.9 ignores out-of-range values, so we keep the defaults when
  // the caller didn't request a low-power / fp16-everywhere path.)
  // GPU backends: force Precision_High (full fp32) via an OWNED config.
  // Diagnostic purpose: if server-det still corrupts on CUDA with fp32
  // forced, the cutlass kernel-selection bug is precision-independent;
  // if it heals, MNN's default was silently taking a reduced-precision
  // path. PERF2 numbers (High slower than Normal on this cutlass build)
  // still argue for shipping Normal on mobile-class models.
  if (cfg.backend == Backend::Cuda || cfg.backend == Backend::OpenCL ||
      cfg.backend == Backend::Vulkan || cfg.backend == Backend::Metal) {
    impl_->backend_config = MNN::BackendConfig{};
    // GPU backends: force full fp32 (Precision_High). On Metal this is
    // mandatory: MNN's Metal fp16 path diverges from CPU (MLC 0.10–0.85;
    // see platform/desktop/README.md), while fp32 is box-exact vs CPU.
    impl_->backend_config.precision = MNN::BackendConfig::Precision_High;
    // PPOCR_METAL_PREC=low: diagnostic opt-in to re-test the Metal fp16
    // path against the CPU fp32/fp16 references.
    if (cfg.backend == Backend::Metal) {
      const char* mp = std::getenv("PPOCR_METAL_PREC");
      if (mp && mp[0] == 'l') {
        impl_->backend_config.precision = MNN::BackendConfig::Precision_Low;
      }
    }
    sc.backendConfig = &impl_->backend_config;
    impl_->backend_config_set = true;
  } else if (impl_->backend_config_set) {
    // backend switched away from GPU in a reload: restore borrowed default
    sc.backendConfig = nullptr;
    impl_->backend_config_set = false;
  }
  // PPOCR_CPU_PREC=low: opt-in CPU fp16 (Arm82/KleidiAI kernels; the shipped
  // prebuilt has both compiled in). CER/PERF validation pending - do not
  // enable by default until the matrix passes.
  if (cfg.backend == Backend::Cpu) {
    const char* prec = std::getenv("PPOCR_CPU_PREC");
    if (prec && prec[0] == 'l') {
      impl_->backend_config = MNN::BackendConfig{};
      impl_->backend_config.precision = MNN::BackendConfig::Precision_Low;
      sc.backendConfig = &impl_->backend_config;
      impl_->backend_config_set = true;
    }
  }
  // MNN 3.6.1's METAL Winograd convolution is corrupt for the det head conv
  // (320x216, 64->16): nondeterministic noise boxes on some images
  // (ru/00,02,03,07 -> 14..37 boxes instead of 1-6), reproducible with the
  // pristine prebuilt libMNN.a and the real engine (per-inference input
  // copy). CPU Winograd 3.6.1 is CORRECT when the input tensor is re-copied
  // before every runSession (bit-parity with Winograd off on ru+zh) - the
  // earlier "CPU winograd data race" was an artifact of debug drivers that
  // copied the input only once: MNN's pipeline raster (NCHW->NC4HW4)
  // overwrites the engine input tensor in place, so repeated runSession
  // without re-setting the input drifts. To be safe on every backend we
  // force level 0 by default; PPOCR_MNN_WINOGRAD=1 opts back in.
  //
  // PERF-M4 (task-2) measurement on THIS pinned MNN 3.6.1, macOS arm64,
  // process CPU-time (contention-robust) over 12-25 interleaved reps:
  //   det 1280x704 t8: wino0 445.9 ms  vs  wino3 450.6 ms  -> wino 1.0% SLOWER
  //   det 1280x960 t8: wino0 628.1 ms  vs  wino3 669.1 ms  -> wino 6.5% SLOWER
  //   det 1280x704 t10: wino0 688.6 ms vs  wino3 663.2 ms  -> wino 3.7% faster
  //   rec 320x48   t8: wino3 is BIT-IDENTICAL (0/276240 diff) -> not selected
  // and wino3 CHANGES the det prob-map floats (maxabs 4.2e-4 at 704). The
  // winograd-vs-dense choice is therefore shape/thread dependent and wins
  // nothing on the shapes this engine actually runs, so keeping the default
  // level 0 (dense implicit-GEMM) stays the right call: it is also the
  // numerically safer one (bit-identical to the pre-M4 baseline).
  {
    const char* w = std::getenv("PPOCR_MNN_WINOGRAD");
    if (!w || w[0] != '1') {
      impl_->interp->setSessionHint(
          MNN::Interpreter::HintMode::WINOGRAD_MEMORY_LEVEL, 0);
    }
  }
  // KleidiAI (Arm's CPU kernel library, MNN_KLEIDIAI=ON at build time,
  // runtimeHint.enableKleidiAI defaults to true) replaces the 1x1-conv and
  // dense-conv paths with its own micro-kernels. PERF-M4 task-2 measured it
  // on THIS machine (Apple M4, macOS arm64, pinned MNN 3.6.1, fp32) with a
  // contention-robust interleaved A/B on process CPU-time:
  //   det 1280x704 t8: kai1 767.4 ms vs kai0 501.8 ms  -> kai DISABLED 1.53x faster (-35%)
  //   det 1280x960 t8: kai1 683.8 ms vs kai0 518.6 ms  -> kai DISABLED 1.32x faster (-24%)
  //   rec 320x48   t8: kai1  18.2 ms vs kai0  17.3 ms  -> kai DISABLED 1.06x faster
  // (MNN prints "KleidiAI is running! AccelType is FP32." on the kai1 arm.)
  // The KleidiAI path is not bit-identical to the default kernels (det
  // prob-map maxabs ~1e-5), so it is a numerical-path change exactly like
  // winograd: it ships OFF by default only after a 5-language MLC gate
  // (zh/en/ja/ar/ru) confirms MLC <= 0.05 on every language. Opt back in
  // with PPOCR_MNN_KLEIDIAI=1.
  {
    const char* kai = std::getenv("PPOCR_MNN_KLEIDIAI");
    if (!kai || kai[0] != '1') {
      impl_->interp->setSessionHint(
          MNN::Interpreter::HintMode::CPU_ENABLE_KLEIDIAI, 0);
    }
  }
  impl_->session = impl_->interp->createSession(sc);
  if (!impl_->session) {
    throw std::runtime_error("MnnSession: createSession failed");
  }
  impl_->resolved_type = to_mnn(cfg.backend);
  impl_->resolved_name = mnn_backend_label(impl_->resolved_type);

  // Cache input and output names. The wrapper does not own these strings;
  // MNN keeps the backing storage alive for the lifetime of the session.
  // PERF: also cache the raw MNN::Tensor* so the hot path never re-enters
  // Interpreter::getSessionInput/Output (each takes mNet->lock + a map
  // lookup).
  const auto& in_map = impl_->interp->getSessionInputAll(impl_->session);
  const auto& out_map = impl_->interp->getSessionOutputAll(impl_->session);
  impl_->input_names.clear();
  impl_->input_dims.clear();
  impl_->input_hosts.clear();
  impl_->input_tensors.clear();
  impl_->input_stage.clear();
  for (const auto& kv : in_map) {
    impl_->input_names.push_back(kv.first);
    impl_->input_dims.push_back(kv.second->shape());
    impl_->input_hosts.push_back(kv.second->host<float>());
    impl_->input_tensors.push_back(kv.second);
    impl_->input_stage.emplace_back(nullptr);
  }
  impl_->output_names.clear();
  impl_->output_dims.clear();
  impl_->output_hosts.clear();
  impl_->output_tensors.clear();
  for (const auto& kv : out_map) {
    impl_->output_names.push_back(kv.first);
    impl_->output_dims.push_back(kv.second->shape());
    impl_->output_hosts.push_back(kv.second->host<float>());
    impl_->output_tensors.push_back(kv.second);
  }
  impl_->host_out.reset();
  impl_->host_out_valid = false;
}

// PERF: refresh the cached device tensor pointers + host addresses after a
// resize. resizeTensor/resizeSession can reallocate the CPU buffers, so the
// previously cached pointers must be re-read from the interpreter.
void MnnSession::refresh_tensors_locked() {
  for (size_t i = 0; i < impl_->input_names.size(); ++i) {
    MNN::Tensor* t = impl_->interp->getSessionInput(
        impl_->session, impl_->input_names[i].c_str());
    if (t) {
      impl_->input_tensors[i] = t;
      impl_->input_hosts[i] = t->host<float>();
    }
  }
  for (size_t i = 0; i < impl_->output_names.size(); ++i) {
    MNN::Tensor* t = impl_->interp->getSessionOutput(
        impl_->session, impl_->output_names[i].c_str());
    if (t) {
      impl_->output_tensors[i] = t;
      impl_->output_dims[i] = t->shape();
      impl_->output_hosts[i] = t->host<float>();
    }
  }
}

void MnnSession::resize_input(const std::string& name,
                              const std::vector<int>& dims) {
  if (!impl_->interp) throw std::runtime_error("MnnSession: not loaded");
  int idx = -1;
  for (size_t i = 0; i < impl_->input_names.size(); ++i) {
    if (impl_->input_names[i] == name) { idx = static_cast<int>(i); break; }
  }
  if (idx < 0) throw std::runtime_error("MnnSession: no input named " + name);
  // PERF: skip the whole resize when the shape is unchanged. resizeTensor
  // itself early-outs on identical dims, but we would still pay a
  // resizeSession() (which re-encodes/re-allocates the pipeline) plus the
  // tensor refresh. det/rec/cls all reuse the same shape across runs in the
  // common case, so this removes real per-run work.
  if (impl_->input_dims[idx] == dims) return;
  impl_->interp->resizeTensor(impl_->input_tensors[idx], dims);
  // After resizing an input, the session must be told to settle so that
  // the new buffers are allocated. MNN recommends calling resizeSession
  // exactly once after the last resizeTensor.
  impl_->interp->resizeSession(impl_->session);
  impl_->input_dims[idx] = dims;
  // Resize may have reallocated buffers; re-resolve every cached pointer.
  refresh_tensors_locked();
}

float* MnnSession::input_host(const std::string& name) {
  if (!impl_->interp) return nullptr;
  for (size_t i = 0; i < impl_->input_names.size(); ++i) {
    if (impl_->input_names[i] == name) return impl_->input_hosts[i];
  }
  // MNN 3.6.1: for non-CPU backends the session input tensor lives on
  // the device (host == nullptr). set_input_float detects this and uses
  // the copyFromHostTensor path; input_host returning null is expected
  // there and no longer an error.
  return nullptr;
}

const float* MnnSession::output_host(const std::string& name) const {
  if (!impl_->interp) return nullptr;
  int idx = impl_->output_index(name);
  if (idx < 0) return nullptr;
  return impl_->output_tensors[idx] ? impl_->output_tensors[idx]->host<float>()
                                    : nullptr;
}

void MnnSession::set_input_float(const std::string& name,
                                 const std::vector<int>& dims,
                                 const float* data) {
  if (!data) throw std::runtime_error("MnnSession: null input data");
  MnnSession::resize_input(name, dims);
  size_t n = 1;
  for (int d : dims) n *= static_cast<size_t>(d > 0 ? d : 1);
  // Always stage through a host tensor and copyFromHostTensor.
  // NEVER memcpy into the session tensor's host pointer directly: for the
  // Arm82 (CPU fp16, precision=Low) backend the session tensors carry fp16
  // data, and copyFromHostTensor routes through onCopyBuffer, which
  // quantizes fp32->fp16 (CPU->Arm82) or is a plain copy for float
  // backends. Direct memcpy feeds fp32 bytes to kernels that read fp16.
  int idx = -1;
  for (size_t i = 0; i < impl_->input_names.size(); ++i) {
    if (impl_->input_names[i] == name) { idx = static_cast<int>(i); break; }
  }
  if (idx < 0) throw std::runtime_error("MnnSession: no input named " + name);
  MNN::Tensor* dev = impl_->input_tensors[idx];
  if (!dev) throw std::runtime_error("MnnSession: no input named " + name);
  // PERF: reuse a persistent staging tensor. It only needs re-creating when
  // the shape changes (after resize_input refreshed the dims).
  MNN::Tensor* host_tensor = impl_->input_stage[idx].get();
  if (!host_tensor || host_tensor->shape() != dev->shape()) {
    MNN::Tensor* fresh = MNN::Tensor::create(
        dev->shape(), halide_type_of<float>(), nullptr, MNN::Tensor::CAFFE);
    if (!fresh || !fresh->host<float>()) {
      delete fresh;
      throw std::runtime_error("MnnSession: host staging tensor alloc failed");
    }
    impl_->input_stage[idx].reset(fresh);
    host_tensor = fresh;
  }
  std::memcpy(host_tensor->host<float>(), data, n * sizeof(float));
  dump_input_if_requested(name, dims, data);
  if (!dev->copyFromHostTensor(host_tensor)) {
    throw std::runtime_error("MnnSession: copyFromHostTensor failed");
  }
}

int MnnSession::run() {
  if (!impl_->interp) return -1;
  // PERF: no per-run output-pointer refresh. MNN only reallocates output
  // buffers on resize, and refresh_tensors_locked() re-resolves every
  // cached pointer right after a resize. Re-querying here would take
  // mNet->lock (per output, twice) for nothing.
  MNN::ErrorCode ec = impl_->interp->runSession(impl_->session);
  return static_cast<int>(ec);
}

SessionOutput MnnSession::output(const std::string& name) const {
  SessionOutput so;
  if (!impl_->interp) return so;
  int idx = impl_->output_index(name);
  if (idx < 0) return so;
  MNN::Tensor* dev = impl_->output_tensors[idx];
  if (!dev) return so;
  // Read the shape straight off the cached tensor. This is a cheap
  // vector<int> copy with NO interpreter lock, and it keeps dynamic-shape
  // models correct even though run() no longer re-queries outputs: MNN can
  // update an output's shape at run time without a resize_input.
  so.shape = dev->shape();
  impl_->output_dims[idx] = so.shape;
  // Always snapshot through copyToHostTensor. NEVER read the session
  // output tensor's host pointer directly: for the Arm82 (CPU fp16,
  // precision=Low) backend the session tensors carry fp16 data, and
  // copyToHostTensor routes through onCopyBuffer, which dequantizes
  // fp16->fp32 (Arm82->CPU). A direct read would reinterpret fp16
  // bytes as fp32 (e.g. uniform 0x7e007e00 = fp16 NaN pattern).
  // PERF: reuse the host readback tensor + the owned float buffer. Both
  // only grow when the output shape grows.
  if (!impl_->host_out_valid || impl_->host_out_dims != dev->shape()) {
    impl_->host_out.reset(MNN::Tensor::createHostTensorFromDevice(dev, false));
    impl_->host_out_dims = dev->shape();
    impl_->host_out_valid = true;
  }
  MNN::Tensor* host_out = impl_->host_out.get();
  if (!host_out) return so;
  if (!dev->copyToHostTensor(host_out)) return so;
  const size_t total = host_out->size() / sizeof(float);
  if (impl_->device_out_cache.size() < total) {
    impl_->device_out_cache.resize(total);
  }
  std::memcpy(impl_->device_out_cache.data(), host_out->host<float>(),
              host_out->size());
  if (const char* dp = std::getenv("PPOCR_DUMP_OUTPUT")) {
    static int oseq = 0;
    char pth[1024];
    std::snprintf(pth, sizeof(pth), "%s.%s.%d", dp, name.c_str(), oseq++);
    FILE* f = std::fopen(pth, "wb");
    if (f) {
      int nd = (int)impl_->output_dims[idx].size();
      std::fwrite(&nd, sizeof(int), 1, f);
      std::fwrite(impl_->output_dims[idx].data(), sizeof(int), nd, f);
      std::fwrite(impl_->device_out_cache.data(), sizeof(float), total, f);
      std::fclose(f);
    }
  }
  so.data = impl_->device_out_cache.data();
  return so;
}

const char* MnnSession::backend_name() const {
  return impl_->resolved_name.c_str();
}

} // namespace ppocr
