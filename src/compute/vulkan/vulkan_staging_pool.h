#pragma once

#include <cstddef>
#include <cstdint>

#include "compute/vulkan/vulkan_config.h"
#include "pomai/status.h"

namespace pomai::compute::vulkan {

class VulkanComputeContext;

/// Bounded ring of host-visible staging memory for uploads to device-local buffers (discrete GPU path).
class VulkanStagingPool {
public:
    VulkanStagingPool() = default;
    VulkanStagingPool(const VulkanStagingPool&) = delete;
    VulkanStagingPool& operator=(const VulkanStagingPool&) = delete;
    VulkanStagingPool(VulkanStagingPool&&) = default;
    VulkanStagingPool& operator=(VulkanStagingPool&&) = default;
    ~VulkanStagingPool();

    /// total_bytes: hard cap; must be > 0.
    static Status Create(VulkanComputeContext* ctx, std::uint64_t total_bytes, VulkanStagingPool* out);

    std::uint64_t capacity() const { return capacity_; }

    struct StagingSlice {
        vk::Buffer buffer{};
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        void* mapped = nullptr;
        bool valid() const { return buffer && size > 0 && mapped != nullptr; }
    };

    /// Single-threaded acquire: may call waitIdle on wrap.
    Status Acquire(std::uint64_t size, std::uint64_t alignment, StagingSlice* out);

    struct Telemetry {
        std::uint64_t acquires = 0;
        std::uint64_t acquire_failures = 0;
        std::uint64_t wrap_count = 0;
        std::uint64_t peak_used = 0;
    };

    const Telemetry& telemetry() const { return telemetry_; }

private:
    VulkanComputeContext* ctx_{nullptr};
    vk::UniqueBuffer buffer_{};
    vk::UniqueDeviceMemory memory_{};
    void* mapped_base_{nullptr};
    std::uint64_t capacity_{0};
    std::uint64_t cursor_{0};
    Telemetry telemetry_{};
};

}  // namespace pomai::compute::vulkan
