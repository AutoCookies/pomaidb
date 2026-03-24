#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "compute/vulkan/vulkan_config.h"
#include "pomai/status.h"

namespace pomai::compute::vulkan {

/// Tunables for Vulkan memory bridge (mirrors DBOptions vulkan_* fields).
struct BridgeOptions {
    bool prefer_unified_memory = true;
    uint32_t staging_pool_mb = 16;
    uint64_t zero_copy_min_bytes = 4096;
};

/// Snapshot of device capabilities for planner / heuristics (Phase 2+).
struct VulkanDeviceCapabilities {
    bool integrated_gpu = false;
    bool ext_external_memory_host = false;
    uint64_t min_host_alignment = 1;
    uint32_t memory_heap_count = 0;
    uint32_t memory_type_count = 0;
};

/// Runtime Vulkan state: instance, device, one compute/transfer queue, sync primitives.
class VulkanComputeContext {
public:
    VulkanComputeContext() = default;
    VulkanComputeContext(const VulkanComputeContext&) = delete;
    VulkanComputeContext& operator=(const VulkanComputeContext&) = delete;
    VulkanComputeContext(VulkanComputeContext&&) = default;
    VulkanComputeContext& operator=(VulkanComputeContext&&) = default;
    ~VulkanComputeContext() = default;

    static Status Create(const BridgeOptions& opts, VulkanComputeContext* out);

    vk::Instance instance() const { return instance_.get(); }
    vk::PhysicalDevice physical_device() const { return physical_device_; }
    vk::Device device() const { return device_.get(); }
    vk::Queue queue() const { return queue_; }
    uint32_t queue_family_index() const { return queue_family_index_; }

    const vk::PhysicalDeviceMemoryProperties& memory_properties() const { return memory_properties_; }

    bool ext_external_memory_host() const { return ext_external_memory_host_; }
    uint64_t min_host_import_alignment() const { return min_host_import_alignment_; }

    vk::CommandPool transfer_cmd_pool() const { return transfer_cmd_pool_.get(); }
    vk::Fence transfer_fence() const { return transfer_fence_.get(); }

    const VulkanDeviceCapabilities& capabilities() const { return capabilities_; }
    const BridgeOptions& bridge_options() const { return bridge_opts_; }

private:
    vk::UniqueInstance instance_;
    vk::PhysicalDevice physical_device_{};
    vk::UniqueDevice device_;
    vk::Queue queue_{};
    uint32_t queue_family_index_{0};

    vk::PhysicalDeviceMemoryProperties memory_properties_{};

    bool ext_external_memory_host_{false};
    uint64_t min_host_import_alignment_{1};

    vk::UniqueCommandPool transfer_cmd_pool_;
    vk::UniqueFence transfer_fence_;

    BridgeOptions bridge_opts_;
    VulkanDeviceCapabilities capabilities_{};
};

}  // namespace pomai::compute::vulkan
