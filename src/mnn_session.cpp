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

// Winograd memory level handed to MNN. Default 0 (Winograd convolution off).
// Measured on this build (MNN 3.6.1, M4, v6-tiny det @1280x960, min-of-18):
//   level 0: 57.6 ms, RSS 169 MB
//   level 3: 63.3 ms, RSS 190 MB   (MNN's own default)
// so level 3 is slower AND costs ~20 MB more workspace here. Level 1 is
// reported broken (wrong boxes on ru). The det graph is dominated by
// global-pooling/Raster traffic, not convolution, so Winograd cannot pay
// for itself; keep it off. PPOCR_MNN_WINOGRAD overrides (0/1/3).
#ifndef MNN_DEFAULT_WINOGRAD_LEVEL
#define MNN_DEFAULT_WINOGRAD_LEVEL 0
#endif
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
  // Cached dims per output name.
  std::vector<std::vector<int>> output_dims;
  std::vector<std::string>      output_names;
  std::vector<const float*>     output_hosts;
  // Host snapshot of the last device output (non-CPU backends). Owned.
  float* device_out_cache = nullptr;
  size_t device_out_cache_cap = 0;    // floats, reused instead of new/delete
  // Staging tensor reused across set_input_float calls (created once per
  // distinct shape). Avoids a 4.9 MB alloc + free per frame.
  MNN::Tensor* stage_tensor = nullptr;
  std::vector<int> stage_dims;
  // Last dims handed to resizeTensor, so we can skip a redundant
  // resizeSession when the caller asks for the same shape again.
  std::vector<std::vector<int>> resized_dims;
  // Cached per-output readback tensors (host side), reused across runs.
  std::vector<MNN::Tensor*> out_host_tensors;
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
  for (auto* t : impl_->out_host_tensors) delete t;
  delete impl_->stage_tensor;
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
    // GPU backends: precision defaults to full fp32 (Precision_High).
    // On Metal, High = fp32 storage + fp32 shaders (~58 ms det on v6-tiny at
    // 1280x960); Normal = fp16 storage + fp32 accumulate is ~35% faster
    // (38 ms) BUT fails the accuracy gate badly (measured, metal backend,
    // 5 langs, vs the fp32 paddle reference): v6-tiny MLC zh 0.099 / en 0.160
    // / ru 0.600 / ja 0.596 / ar 0.448; v4-mobile ru 0.139 / ar 0.267.
    // So High stays the default. PPOCR_METAL_PREC=normal|low is a
    // diagnostic-only opt-in and must never be shipped.
    impl_->backend_config = MNN::BackendConfig{};
    impl_->backend_config.precision = MNN::BackendConfig::Precision_High;
    if (cfg.backend == Backend::Metal) {
      const char* mp = std::getenv("PPOCR_METAL_PREC");
      if (mp && mp[0] == 'n') {
        impl_->backend_config.precision = MNN::BackendConfig::Precision_Normal;
      } else if (mp && mp[0] == 'l') {
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
  // Winograd memory level (see MNN_DEFAULT_WINOGRAD_LEVEL above): default 0.
  // PPOCR_MNN_WINOGRAD=3 / =1 opts into MNN's aggressive levels.
  {
    const char* w = std::getenv("PPOCR_MNN_WINOGRAD");
    int level = MNN_DEFAULT_WINOGRAD_LEVEL;  // 0 = Winograd off
    if (w) {
      if (w[0] == '1') level = 1;
      else if (w[0] == '3') level = 3;
      else if (w[0] == '0') level = 0;
    }
    impl_->interp->setSessionHint(
        MNN::Interpreter::HintMode::WINOGRAD_MEMORY_LEVEL, level);
  }
  // OP_ENCODER_NUMBER_FOR_COMMIT batches N encoded ops per MTLCommandBuffer.
  // MNN's default is 10; on the det graph that yields ~11 command buffers per
  // inference. Measured neutral (10/32/128/100000 all min ~58 ms), because
  // the global poolings force their own syncs regardless, so leave MNN's
  // default alone. PPOCR_MNN_COMMIT_OPS is the A/B knob.
  {
    const char* c = std::getenv("PPOCR_MNN_COMMIT_OPS");
    if (c && c[0]) {
      int ops = std::atoi(c);
      if (ops > 0) {
        impl_->interp->setSessionHint(
            MNN::Interpreter::HintMode::OP_ENCODER_NUMBER_FOR_COMMIT, ops);
      }
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
  // Readback host tensors are shaped per (output, dims); drop any stale
  // ones from a previous load() and re-create lazily.
  for (auto* t : impl_->out_host_tensors) delete t;
  impl_->out_host_tensors.assign(impl_->output_names.size(), nullptr);
  delete impl_->stage_tensor;
  impl_->stage_tensor = nullptr;
  impl_->stage_dims.clear();
  impl_->resized_dims.assign(impl_->input_names.size(), std::vector<int>());
}

void MnnSession::resize_input(const std::string& name,
                              const std::vector<int>& dims) {
  if (!impl_->interp) throw std::runtime_error("MnnSession: not loaded");
  MNN::Tensor* t = impl_->interp->getSessionInput(impl_->session,
                                                    name.c_str());
  if (!t) throw std::runtime_error("MnnSession: no input named " + name);
  // PERF: resizeTensor + resizeSession on every call is redundant churn when
  // the shape repeats (the common case: same image size for det, same
  // batch_w for rec), so remember the last dims per input and skip the
  // settle. Measured effect on the Metal det path: ~0.4-0.8 ms/frame off
  // set_input (min 1.4 ms with the skip vs 2.2 ms without). It also avoids
  // invalidating the backend's cached buffers every frame.
  // PPOCR_NO_RESIZE_CACHE=0 restores the old always-resize behaviour (A/B).
  bool dirty = true;
  for (size_t i = 0; i < impl_->input_names.size(); ++i) {
    if (impl_->input_names[i] != name) continue;
    if (i < impl_->resized_dims.size() && impl_->resized_dims[i] == dims) {
      dirty = false;
    }
    const char* ab = std::getenv("PPOCR_NO_RESIZE_CACHE");
    if (ab && ab[0] == '0') dirty = true;
    if (dirty) {
      impl_->interp->resizeTensor(t, dims);
      // After resizing an input the session must settle once so the new
      // buffers are allocated (MNN's documented contract).
      impl_->interp->resizeSession(impl_->session);
      impl_->resized_dims[i] = dims;
    }
    // Refresh host pointers and dims; resize may have reallocated (and on
    // Metal the session input lives on the device -> host == nullptr).
    impl_->input_dims[i] = dims;
    impl_->input_hosts[i] =
        impl_->interp->getSessionInput(impl_->session, name.c_str())
            ->host<float>();
    break;
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
  size_t n = 1;
  for (int d : dims) n *= static_cast<size_t>(d > 0 ? d : 1);
  // Always stage through a host tensor and copyFromHostTensor.
  // NEVER memcpy into the session tensor's host pointer directly: for the
  // Arm82 (CPU fp16, precision=Low) backend the session tensors carry fp16
  // data, and copyFromHostTensor routes through onCopyBuffer, which
  // quantizes fp32->fp16 (CPU->Arm82) or is a plain copy for float
  // backends. Direct memcpy feeds fp32 bytes to kernels that read fp16.
  MNN::Tensor* dev = impl_->interp->getSessionInput(impl_->session,
                                                      name.c_str());
  if (!dev) throw std::runtime_error("MnnSession: no input named " + name);
  // PERF: reuse one staging tensor instead of create+delete per call
  // (the det input is 1280x960x3 f32 = 4.9 MB; create/delete churned
  // malloc + a full-page memset every frame). Re-create only when the
  // staging buffer is missing, its shape/type no longer matches the device
  // input, or the caller's element count would not fit (MNN may keep a
  // different shape than `dims` when it pads channels).
  const size_t stage_bytes = n * sizeof(float);
  const bool need_new_stage =
      impl_->stage_tensor == nullptr || impl_->stage_dims != dev->shape() ||
      impl_->stage_tensor->getType() != dev->getType() ||
      static_cast<size_t>(impl_->stage_tensor->size()) < stage_bytes;
  if (need_new_stage) {
    delete impl_->stage_tensor;
    impl_->stage_tensor = MNN::Tensor::create(
        dev->shape(), dev->getType(), nullptr, MNN::Tensor::CAFFE);
    impl_->stage_dims = dev->shape();
    if (!impl_->stage_tensor || !impl_->stage_tensor->host<void>() ||
        static_cast<size_t>(impl_->stage_tensor->size()) < stage_bytes) {
      delete impl_->stage_tensor;
      impl_->stage_tensor = nullptr;
      throw std::runtime_error("MnnSession: host staging tensor alloc failed");
    }
  }
  std::memcpy(impl_->stage_tensor->host<float>(), data, stage_bytes);
  dump_input_if_requested(name, dims, data);
  if (!dev->copyFromHostTensor(impl_->stage_tensor)) {
    throw std::runtime_error("MnnSession: copyFromHostTensor failed");
  }
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
  // DEBUG(wino): PPOCR_DUMP_OUTPUT=<prefix> dumps each output after readback.
  struct DumpGuard { ~DumpGuard() {} } _dumpGuardUnused;
  for (size_t i = 0; i < impl_->output_names.size(); ++i) {
    if (impl_->output_names[i] != name) continue;
    so.shape = impl_->output_dims[i];
    // Always snapshot through copyToHostTensor. NEVER read the session
    // output tensor's host pointer directly: for the Arm82 (CPU fp16,
    // precision=Low) backend the session tensors carry fp16 data, and
    // copyToHostTensor routes through onCopyBuffer, which dequantizes
    // fp16->fp32 (Arm82->CPU). A direct read would reinterpret fp16
    // bytes as fp32 (e.g. uniform 0x7e007e00 = fp16 NaN pattern).
    MNN::Tensor* dev = impl_->interp->getSessionOutput(
        impl_->session, name.c_str());
    if (!dev) return so;
    // PERF: reuse both the host readback tensor and the exposed cache
    // buffer instead of createHostTensorFromDevice + new/delete per frame
    // (2.7 MB alloc churn per call on the det prob map).
    const size_t elem = dev->getType().bytes() > 0 ? dev->getType().bytes() : sizeof(float);
    const size_t total = static_cast<size_t>(dev->size()) / elem;
    if (i >= impl_->out_host_tensors.size()) return so;
    MNN::Tensor* host_out = impl_->out_host_tensors[i];
    if (host_out == nullptr || host_out->shape() != dev->shape() ||
        static_cast<size_t>(host_out->size()) < static_cast<size_t>(dev->size())) {
      delete host_out;
      host_out = MNN::Tensor::createHostTensorFromDevice(dev, false);
      impl_->out_host_tensors[i] = host_out;
    }
    if (host_out == nullptr) return so;
    if (!dev->copyToHostTensor(host_out)) return so;
    if (impl_->device_out_cache == nullptr ||
        impl_->device_out_cache_cap < total) {
      delete[] impl_->device_out_cache;
      impl_->device_out_cache = new float[total];
      impl_->device_out_cache_cap = total;
    }
    std::memcpy(impl_->device_out_cache, host_out->host<float>(),
                total * sizeof(float));
    if (const char* dp = std::getenv("PPOCR_DUMP_OUTPUT")) {
      static int oseq = 0;
      char pth[1024];
      std::snprintf(pth, sizeof(pth), "%s.%s.%d", dp, name.c_str(), oseq++);
      FILE* f = std::fopen(pth, "wb");
      if (f) {
        int nd = (int)impl_->output_dims[i].size();
        std::fwrite(&nd, sizeof(int), 1, f);
        std::fwrite(impl_->output_dims[i].data(), sizeof(int), nd, f);
        std::fwrite(impl_->device_out_cache, sizeof(float), total, f);
        std::fclose(f);
      }
    }
    so.data = impl_->device_out_cache;
    return so;
  }
  return so;
}

const char* MnnSession::backend_name() const {
  return impl_->resolved_name.c_str();
}

} // namespace ppocr
