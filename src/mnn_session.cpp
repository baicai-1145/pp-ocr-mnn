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

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppocr {

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
  // Cached dims per output name.
  std::vector<std::vector<int>> output_dims;
  std::vector<std::string>      output_names;
  std::vector<const float*>     output_hosts;
  // Host snapshot of the last device output (non-CPU backends). Owned.
  float* device_out_cache = nullptr;
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
  delete[] impl_->device_out_cache;
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
      cfg.backend == Backend::Vulkan) {
    impl_->backend_config = MNN::BackendConfig{};
    impl_->backend_config.precision = MNN::BackendConfig::Precision_High;
    sc.backendConfig = &impl_->backend_config;
    impl_->backend_config_set = true;
  } else if (impl_->backend_config_set) {
    // backend switched away from GPU in a reload: restore borrowed default
    sc.backendConfig = nullptr;
    impl_->backend_config_set = false;
  }
  impl_->session = impl_->interp->createSession(sc);
  if (!impl_->session) {
    throw std::runtime_error("MnnSession: createSession failed");
  }
  impl_->resolved_type = to_mnn(cfg.backend);
  impl_->resolved_name = mnn_backend_label(impl_->resolved_type);

  // Cache input and output names. The wrapper does not own these strings;
  // MNN keeps the backing storage alive for the lifetime of the session.
  const auto& in_map = impl_->interp->getSessionInputAll(impl_->session);
  const auto& out_map = impl_->interp->getSessionOutputAll(impl_->session);
  impl_->input_names.clear();
  impl_->input_dims.clear();
  impl_->input_hosts.clear();
  for (const auto& kv : in_map) {
    impl_->input_names.push_back(kv.first);
    impl_->input_dims.push_back(kv.second->shape());
    impl_->input_hosts.push_back(kv.second->host<float>());
  }
  impl_->output_names.clear();
  impl_->output_dims.clear();
  impl_->output_hosts.clear();
  for (const auto& kv : out_map) {
    impl_->output_names.push_back(kv.first);
    impl_->output_dims.push_back(kv.second->shape());
    impl_->output_hosts.push_back(kv.second->host<float>());
  }
}

void MnnSession::resize_input(const std::string& name,
                              const std::vector<int>& dims) {
  if (!impl_->interp) throw std::runtime_error("MnnSession: not loaded");
  MNN::Tensor* t = impl_->interp->getSessionInput(impl_->session,
                                                    name.c_str());
  if (!t) throw std::runtime_error("MnnSession: no input named " + name);
  impl_->interp->resizeTensor(t, dims);
  // After resizing an input, the session must be told to settle so that
  // the new buffers are allocated. MNN recommends calling resizeSession
  // exactly once after the last resizeTensor.
  impl_->interp->resizeSession(impl_->session);
  // Refresh host pointers and dims; resize may have reallocated.
  for (size_t i = 0; i < impl_->input_names.size(); ++i) {
    if (impl_->input_names[i] == name) {
      impl_->input_dims[i] = dims;
      impl_->input_hosts[i] =
          impl_->interp->getSessionInput(impl_->session, name.c_str())
              ->host<float>();
    }
  }
}

float* MnnSession::input_host(const std::string& name) {
  if (!impl_->interp) return nullptr;
  MNN::Tensor* t = impl_->interp->getSessionInput(impl_->session,
                                                    name.c_str());
  // MNN 3.6.1: for non-CPU backends the session input tensor lives on
  // the device (host == nullptr). set_input_float detects this and uses
  // the copyFromHostTensor path; input_host returning null is expected
  // there and no longer an error.
  return t ? t->host<float>() : nullptr;
}

const float* MnnSession::output_host(const std::string& name) const {
  if (!impl_->interp) return nullptr;
  MNN::Tensor* t = impl_->interp->getSessionOutput(impl_->session,
                                                     name.c_str());
  return t ? t->host<float>() : nullptr;
}

void MnnSession::set_input_float(const std::string& name,
                                 const std::vector<int>& dims,
                                 const float* data) {
  if (!data) throw std::runtime_error("MnnSession: null input data");
  MnnSession::resize_input(name, dims);
  float* dst = MnnSession::input_host(name);
  size_t n = 1;
  for (int d : dims) n *= static_cast<size_t>(d > 0 ? d : 1);
  if (dst) {
    // CPU backend (or device backend with host-mapped memory): the
    // fast path — write the host pointer directly.
    std::memcpy(dst, data, n * sizeof(float));
    return;
  }
  // MNN 3.6.1, non-CPU backend: the session input tensor lives on the
  // device (host == nullptr). Stage through a host tensor so the
  // backend's onCopyBuffer (CUDA / Vulkan / OpenCL) does the transfer
  // (same fix as the m3-cuda gate driver).
  MNN::Tensor* dev = impl_->interp->getSessionInput(impl_->session,
                                                      name.c_str());
  if (!dev) throw std::runtime_error("MnnSession: no input named " + name);
  MNN::Tensor* host_tensor = MNN::Tensor::create(
      dev->shape(), halide_type_of<float>(), nullptr, MNN::Tensor::CAFFE);
  if (!host_tensor || !host_tensor->host<float>()) {
    delete host_tensor;
    throw std::runtime_error("MnnSession: host staging tensor alloc failed");
  }
  std::memcpy(host_tensor->host<float>(), data, n * sizeof(float));
  if (!dev->copyFromHostTensor(host_tensor)) {
    delete host_tensor;
    throw std::runtime_error("MnnSession: copyFromHostTensor failed");
  }
  delete host_tensor;
}

int MnnSession::run() {
  if (!impl_->interp) return -1;
  // Refresh output pointers; MNN may reallocate between runs.
  for (size_t i = 0; i < impl_->output_names.size(); ++i) {
    MNN::Tensor* t = impl_->interp->getSessionOutput(
        impl_->session, impl_->output_names[i].c_str());
    if (t) impl_->output_hosts[i] = t->host<float>();
  }
  MNN::ErrorCode ec = impl_->interp->runSession(impl_->session);
  // After runSession, output tensor dims and host pointers are valid.
  for (size_t i = 0; i < impl_->output_names.size(); ++i) {
    MNN::Tensor* t = impl_->interp->getSessionOutput(
        impl_->session, impl_->output_names[i].c_str());
    if (t) {
      impl_->output_dims[i] = t->shape();
      impl_->output_hosts[i] = t->host<float>();
    }
  }
  return static_cast<int>(ec);
}

SessionOutput MnnSession::output(const std::string& name) const {
  SessionOutput so;
  if (!impl_->interp) return so;
  for (size_t i = 0; i < impl_->output_names.size(); ++i) {
    if (impl_->output_names[i] != name) continue;
    so.shape = impl_->output_dims[i];
    so.data = impl_->output_hosts[i];
    if (so.data) return so;
    // MNN 3.6.1, non-CPU backend: the output tensor lives on the
    // device. Snapshot it into a caller-lifetime host buffer (the
    // MnnSession owns it until the next output() call — fine for the
    // single-threaded run-then-decode flow of the engine).
    MNN::Tensor* dev = impl_->interp->getSessionOutput(
        impl_->session, name.c_str());
    if (!dev) return so;
    MNN::Tensor* host_out =
        MNN::Tensor::createHostTensorFromDevice(dev, false);
    if (!host_out) return so;
    if (!dev->copyToHostTensor(host_out)) { delete host_out; return so; }
    // Cache the buffer; freed on the next output() call or destruction.
    delete[] impl_->device_out_cache;
    const size_t total = host_out->size() / sizeof(float);
    impl_->device_out_cache = new float[total];
    std::memcpy(impl_->device_out_cache, host_out->host<float>(),
                host_out->size());
    so.data = impl_->device_out_cache;
    delete host_out;
    return so;
  }
  return so;
}

const char* MnnSession::backend_name() const {
  return impl_->resolved_name.c_str();
}

} // namespace ppocr
