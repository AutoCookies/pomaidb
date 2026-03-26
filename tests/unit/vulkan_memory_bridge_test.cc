#include "tests/common/test_main.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include "compute/vulkan/vulkan_device_context.h"
#include "compute/vulkan/vulkan_memory_bridge.h"
#include "compute/vulkan/vulkan_staging_pool.h"
#include "core/memory/gpu_buffer_pin_manager.h"
#include "palloc_compat.h"

namespace {

// GitHub-hosted runners often have no usable Vulkan ICD/GPU; set POMAI_SKIP_VULKAN_TESTS=1 in CI.
bool SkipVulkanGpuTests() {
    const char* s = std::getenv("POMAI_SKIP_VULKAN_TESTS");
    return s != nullptr && s[0] != '\0' && std::strcmp(s, "0") != 0;
}

std::vector<std::byte> MakeBytes(std::size_t n, std::byte seed) {
    std::vector<std::byte> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = static_cast<std::byte>(static_cast<unsigned>(seed) + static_cast<unsigned>(i % 251));
    }
    return v;
}

}  // namespace

POMAI_TEST(Vulkan_MemoryBridge_CopyMapped) {
    if (SkipVulkanGpuTests()) {
        POMAI_EXPECT_TRUE(true);
        return;
    }
    pomai::compute::vulkan::BridgeOptions bopt;
    bopt.prefer_unified_memory = true;
    bopt.zero_copy_min_bytes = 1ull << 30;  // force copy path for small payloads
    bopt.staging_pool_mb = 4;

    pomai::compute::vulkan::VulkanComputeContext ctx;
    auto st = pomai::compute::vulkan::VulkanComputeContext::Create(bopt, &ctx);
    if (!st.ok()) {
        POMAI_EXPECT_TRUE(true);  // headless / CI without Vulkan
        return;
    }

    auto payload = MakeBytes(1024, std::byte{3});
    pomai::compute::vulkan::HostBuffer hb;
    st = pomai::compute::vulkan::VulkanMemoryBridge::CreateBufferCopyMapped(&ctx, payload, &hb);
    POMAI_EXPECT_OK(st);
    POMAI_EXPECT_EQ(hb.path, pomai::compute::vulkan::HostBufferPath::kCopyMappedHost);
    POMAI_EXPECT_TRUE(hb.mapped != nullptr);
    POMAI_EXPECT_EQ(static_cast<std::size_t>(hb.size), payload.size());
}

POMAI_TEST(Vulkan_StagingPool_Bounded) {
    pomai::compute::vulkan::BridgeOptions bopt;
    pomai::compute::vulkan::VulkanComputeContext ctx;
    auto st = pomai::compute::vulkan::VulkanComputeContext::Create(bopt, &ctx);
    if (!st.ok()) {
        POMAI_EXPECT_TRUE(true);
        return;
    }

    pomai::compute::vulkan::VulkanStagingPool pool;
    st = pomai::compute::vulkan::VulkanStagingPool::Create(&ctx, 64 * 1024, &pool);
    POMAI_EXPECT_OK(st);

    pomai::compute::vulkan::VulkanStagingPool::StagingSlice a{};
    st = pool.Acquire(100, 256, &a);
    POMAI_EXPECT_OK(st);
    POMAI_EXPECT_TRUE(a.valid());

    pomai::compute::vulkan::VulkanStagingPool::Telemetry tel = pool.telemetry();
    POMAI_EXPECT_EQ(tel.acquires, 1u);
}

POMAI_TEST(Vulkan_PinRegistry_RoundTrip) {
    auto hb = std::make_shared<pomai::compute::vulkan::HostBuffer>();
    std::uint64_t id = pomai::compute::vulkan::VulkanMemoryBridge::PinHostBuffer(hb);
    POMAI_EXPECT_TRUE(id != 0u);
    pomai::compute::vulkan::VulkanMemoryBridge::UnpinHostBuffer(id);
}

POMAI_TEST(Vulkan_ImportAlignment_AlignedPalloc) {
    if (SkipVulkanGpuTests()) {
        POMAI_EXPECT_TRUE(true);
        return;
    }
    pomai::compute::vulkan::BridgeOptions bopt;
    bopt.prefer_unified_memory = true;
    bopt.zero_copy_min_bytes = 64;

    pomai::compute::vulkan::VulkanComputeContext ctx;
    auto st = pomai::compute::vulkan::VulkanComputeContext::Create(bopt, &ctx);
    if (!st.ok() || !ctx.ext_external_memory_host()) {
        POMAI_EXPECT_TRUE(true);
        return;
    }

    const std::uint64_t align = ctx.min_host_import_alignment();
    const std::size_t bytes = 4096;
    void* p = palloc_malloc_aligned(bytes, static_cast<std::size_t>(align));
    POMAI_EXPECT_TRUE(p != nullptr);
    std::memset(p, 7, bytes);

    std::span<const std::byte> span(static_cast<const std::byte*>(p), bytes);
    pomai::compute::vulkan::HostBuffer hb;
    st = pomai::compute::vulkan::VulkanMemoryBridge::CreateBufferFromHostSpan(&ctx, span, &hb);
    palloc_free(p);

    if (!st.ok()) {
        POMAI_EXPECT_TRUE(true);
        return;
    }
    POMAI_EXPECT_TRUE(hb.path == pomai::compute::vulkan::HostBufferPath::kImportHostPointer ||
                      hb.path == pomai::compute::vulkan::HostBufferPath::kCopyMappedHost);
}
