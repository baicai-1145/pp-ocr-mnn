// tools/verify_mnn_probe.cpp — multi-resolution MNN det model probe.
//
// Usage: verify_mnn_probe <model.mnn> <backend> [shape1 shape2 ...]
//   shape = "NxCxHxW" (e.g. "1x3x720x1280")
//   backend = "cpu" | "opencl" | "vulkan"
//
// For each requested shape:
//   - resize input tensor
//   - runSession
//   - read output tensor shape
//   - report run success / failure and output shape
//
// Exit code: 0 if all shapes ran, 1 if any failed.
#include <MNN/Interpreter.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using Shape = std::vector<int>;

static std::vector<Shape> parse_shapes(int argc, char** argv, int start) {
    std::vector<Shape> out;
    for (int i = start; i < argc; ++i) {
        Shape s;
        const char* p = argv[i];
        while (*p) {
            int v = std::atoi(p);
            s.push_back(v);
            const char* q = std::strchr(p, 'x');
            if (!q) break;
            p = q + 1;
        }
        if (s.size() != 4) {
            std::fprintf(stderr, "bad shape: %s (need NxCxHxW)\n", argv[i]);
            std::exit(2);
        }
        out.push_back(s);
    }
    return out;
}

static MNNForwardType to_backend(const std::string& s) {
    if (s == "cpu") return MNN_FORWARD_CPU;
    if (s == "opencl") return MNN_FORWARD_OPENCL;
    if (s == "vulkan") return MNN_FORWARD_VULKAN;
    if (s == "cuda") return MNN_FORWARD_CUDA;
    if (s == "metal") return MNN_FORWARD_METAL;
    if (s == "auto") return MNN_FORWARD_AUTO;
    std::fprintf(stderr, "unknown backend: %s\n", s.c_str());
    std::exit(2);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.mnn> <backend> [shapes...]\n", argv[0]);
        return 2;
    }
    const char* model_path = argv[1];
    const std::string backend_str = argv[2];
    auto shapes = parse_shapes(argc, argv, 3);
    auto btype = to_backend(backend_str);

    auto* interp = MNN::Interpreter::createFromFile(model_path);
    if (!interp) {
        std::fprintf(stderr, "createFromFile failed: %s\n", model_path);
        return 1;
    }
    MNN::ScheduleConfig cfg;
    cfg.type = btype;
    cfg.numThread = 4;
    MNN::Session* sess = interp->createSession(cfg);
    if (!sess) {
        std::fprintf(stderr, "createSession failed\n");
        delete interp;
        return 1;
    }

    // Pick the first input tensor (det models have exactly one: "x").
    auto in_info = interp->getSessionInputAll(sess);
    if (in_info.empty()) {
        std::fprintf(stderr, "no input tensors\n");
        interp->releaseSession(sess);
        delete interp;
        return 1;
    }
    const std::string in_name = in_info.begin()->first;
    auto in_tensor = in_info.begin()->second;
    if (!in_tensor) {
        std::fprintf(stderr, "input tensor null for %s\n", in_name.c_str());
        interp->releaseSession(sess);
        delete interp;
        return 1;
    }
    const Shape orig_dims = in_tensor->shape();
    std::fprintf(stdout, "model: %s\n", model_path);
    std::fprintf(stdout, "  backend: %s\n", backend_str.c_str());
    std::fprintf(stdout, "  input: %s, default shape: [", in_name.c_str());
    for (size_t i = 0; i < orig_dims.size(); ++i) {
        if (i) std::fprintf(stdout, ",");
        std::fprintf(stdout, "%d", orig_dims[i]);
    }
    std::fprintf(stdout, "]\n");
    std::fprintf(stdout, "  num_shapes: %zu\n\n", shapes.size());

    int fail_count = 0;
    for (size_t i = 0; i < shapes.size(); ++i) {
        const Shape& s = shapes[i];
        std::ostringstream sstr;
        for (size_t k = 0; k < s.size(); ++k) {
            if (k) sstr << "x";
            sstr << s[k];
        }
        const std::string sstr_s = sstr.str();
        // resize
        interp->resizeTensor(in_tensor, s);
        interp->resizeSession(sess);
        // Re-fetch input pointer after resize (host buffer may move).
        in_tensor = interp->getSessionInput(sess, in_name.c_str());
        // Write zeros so we don't read uninitialized host memory.
        const int total = s[0] * s[1] * s[2] * s[3];
        if (in_tensor->host<void>()) {
            std::memset(in_tensor->host<void>(), 0, total * sizeof(float));
        }
        // run
        const int rc = interp->runSession(sess);
        if (rc != 0) {
            std::fprintf(stdout, "[FAIL] shape=%s runSession rc=%d\n",
                         sstr_s.c_str(), rc);
            ++fail_count;
            continue;
        }
        // Read output shape
        auto out_info = interp->getSessionOutputAll(sess);
        std::ostringstream out_sstr;
        out_sstr << "[";
        int n_out = 0;
        for (auto& kv : out_info) {
            if (n_out) out_sstr << ",";
            const auto& shape = kv.second->shape();
            for (size_t k = 0; k < shape.size(); ++k) {
                if (k) out_sstr << "x";
                out_sstr << shape[k];
            }
            ++n_out;
        }
        out_sstr << "]";
        std::fprintf(stdout, "[ OK ] shape=%s -> output %s\n",
                     sstr_s.c_str(), out_sstr.str().c_str());
    }

    interp->releaseSession(sess);
    delete interp;

    std::fprintf(stdout, "\nSummary: %zu/%zu shapes passed, %d failed\n",
                 shapes.size() - fail_count, shapes.size(), fail_count);
    return fail_count > 0 ? 1 : 0;
}
