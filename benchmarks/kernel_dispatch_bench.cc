#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "core/kernel/micro_kernel.h"

namespace {
class BenchPod final : public pomai::core::Pod {
public:
    void Handle(pomai::core::Message&& msg) override {
        if (msg.status_ptr) *msg.status_ptr = pomai::Status::Ok();
    }
    pomai::core::PodId Id() const override { return pomai::core::PodId::kIndex; }
    std::string Name() const override { return "BenchPod"; }
    pomai::core::MemoryQuota GetQuota() const override { return {}; }
};
}

int main(int argc, char** argv) {
    const uint32_t n = (argc > 1) ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 200000u;
    pomai::core::MicroKernel kernel;
    auto st = kernel.RegisterPod(std::make_unique<BenchPod>());
    if (!st.ok()) {
        std::cerr << "register failed: " << st.message() << "\n";
        return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < n; ++i) {
        pomai::Status op_st = pomai::Status::Ok();
        auto msg = pomai::core::Message::Create(pomai::core::PodId::kIndex, pomai::core::Op::kFlush);
        msg.status_ptr = &op_st;
        kernel.Enqueue(std::move(msg));
    }
    kernel.ProcessAll();
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    const double mps = sec > 0.0 ? static_cast<double>(n) / sec : 0.0;
    std::cout << "kernel_dispatch_bench n=" << n << " sec=" << sec << " msg_per_sec=" << mps << "\n";
    return 0;
}
