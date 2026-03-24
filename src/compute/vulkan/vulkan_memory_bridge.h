#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>
#include <span>

#include "compute/vulkan/vulkan_config.h"
#include "pomai/status.h"

namespace pomai::compute::vulkan {

class VulkanComputeContext;
class VulkanStagingPool;

enum class HostBufferPath : std::uint8_t {
    kImportHostPointer = 0,
    kCopyMappedHost = 1,
    kDeviceLocalFromStaging = 2,
};

/// GPU buffer backed by host-visible or imported memory; single-owner (move-only).
struct HostBuffer {
    vk::Device device{};
    vk::UniqueBuffer buffer{};
    vk::UniqueDeviceMemory mem{};
    void* mapped = nullptr;
    vk::DeviceSize size = 0;
    HostBufferPath path = HostBufferPath::kCopyMappedHost;

    HostBuffer() = default;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    HostBuffer(HostBuffer&&) = default;
    HostBuffer& operator=(HostBuffer&&) = default;
    ~HostBuffer() {
        if (path == HostBufferPath::kCopyMappedHost && device && mem && mapped) {
            device.unmapMemory(mem.get());
            mapped = nullptr;
        }
    }

    void Reset(VulkanComputeContext* ctx);
};

/// Bridges palloc-compatible host pointers into VkBuffer bindings with minimal copies.
class VulkanMemoryBridge {
public:
    /// Create buffer from a host span. Prefers VK_EXT_external_memory_host import when enabled and aligned.
    static Status CreateBufferFromHostSpan(VulkanComputeContext* ctx,
                                           std::span<const std::byte> bytes,
                                           HostBuffer* out);

    /// Copy from host pointer into a host-visible buffer (always works, no import).
    static Status CreateBufferCopyMapped(VulkanComputeContext* ctx,
                                         std::span<const std::byte> bytes,
                                         HostBuffer* out);

    /// Upload to a device-local buffer using staging pool (discrete / non-UMA path).
    static Status UploadToDeviceBuffer(VulkanComputeContext* ctx,
                                       VulkanStagingPool* pool,
                                       std::span<const std::byte> bytes,
                                       HostBuffer* out_device);

    static std::uint64_t PinHostBuffer(std::shared_ptr<HostBuffer> buf);
    static void UnpinHostBuffer(std::uint64_t session_id);
};

}  // namespace pomai::compute::vulkan
