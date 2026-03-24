#include <chrono>
#include <cstdio>
#include <vector>

#include "compute/vulkan/vulkan_device_context.h"
#include "compute/vulkan/vulkan_memory_bridge.h"
#include "compute/vulkan/vulkan_staging_pool.h"

namespace {

std::vector<std::byte> Payload(std::size_t n) {
    std::vector<std::byte> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<std::byte>(static_cast<unsigned>(i % 251));
    }
    return v;
}

}  // namespace

int main() {
    pomai::compute::vulkan::BridgeOptions bopt;
    bopt.prefer_unified_memory = true;
    bopt.zero_copy_min_bytes = 1u << 20;
    bopt.staging_pool_mb = 32;

    pomai::compute::vulkan::VulkanComputeContext ctx;
    auto st = pomai::compute::vulkan::VulkanComputeContext::Create(bopt, &ctx);
    if (!st.ok()) {
        std::fprintf(stderr, "Vulkan init skipped: %s\n", st.message());
        return 0;
    }

    pomai::compute::vulkan::VulkanStagingPool pool;
    st = pomai::compute::vulkan::VulkanStagingPool::Create(&ctx, static_cast<std::uint64_t>(bopt.staging_pool_mb) * 1024ull * 1024ull,
                                                           &pool);
    if (!st.ok()) {
        std::fprintf(stderr, "Staging pool failed: %s\n", st.message());
        return 1;
    }

    const auto data = Payload(256 * 1024);
    constexpr int kIters = 50;
    double ms_copy = 0.0;
    double ms_upload = 0.0;

    for (int i = 0; i < kIters; ++i) {
        pomai::compute::vulkan::HostBuffer hb;
        const auto t0 = std::chrono::steady_clock::now();
        st = pomai::compute::vulkan::VulkanMemoryBridge::CreateBufferCopyMapped(&ctx, data, &hb);
        const auto t1 = std::chrono::steady_clock::now();
        if (!st.ok()) {
            std::fprintf(stderr, "copy mapped failed: %s\n", st.message());
            return 2;
        }
        ms_copy += std::chrono::duration<double, std::milli>(t1 - t0).count();

        pomai::compute::vulkan::HostBuffer dev;
        const auto t2 = std::chrono::steady_clock::now();
        st = pomai::compute::vulkan::VulkanMemoryBridge::UploadToDeviceBuffer(&ctx, &pool, data, &dev);
        const auto t3 = std::chrono::steady_clock::now();
        if (!st.ok()) {
            std::fprintf(stderr, "upload failed: %s\n", st.message());
            return 3;
        }
        ms_upload += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    std::printf("Vulkan transfer bench: copy_mapped_avg_ms=%.4f upload_device_avg_ms=%.4f (iters=%d, bytes=%zu)\n",
                ms_copy / kIters, ms_upload / kIters, kIters, data.size());
    return 0;
}
