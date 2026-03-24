#include "compute/vulkan/vulkan_memory_bridge.h"

#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "compute/vulkan/vulkan_device_context.h"
#include "compute/vulkan/vulkan_staging_pool.h"
#include "core/memory/gpu_buffer_pin_manager.h"

namespace pomai::compute::vulkan {

namespace {

std::uint32_t FindMemoryType(const vk::PhysicalDeviceMemoryProperties& mp,
                             std::uint32_t type_bits,
                             vk::MemoryPropertyFlags props) {
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0 &&
            (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return std::numeric_limits<std::uint32_t>::max();
}

Status TryImportHostPointer(VulkanComputeContext* ctx,
                            std::span<const std::byte> bytes,
                            HostBuffer* out) {
    vk::BufferCreateInfo bci{};
    bci.size = bytes.size();
    bci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc;
    bci.sharingMode = vk::SharingMode::eExclusive;

    vk::UniqueBuffer buffer;
    try {
        buffer = ctx->device().createBufferUnique(bci);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    const vk::MemoryRequirements req = ctx->device().getBufferMemoryRequirements(buffer.get());

    void* host_ptr = const_cast<void*>(static_cast<const void*>(bytes.data()));

    vk::MemoryHostPointerPropertiesEXT host_props{};
    try {
        // vkGetMemoryHostPointerPropertiesEXT is dispatched on VkDevice (per Vulkan spec).
        host_props = ctx->device().getMemoryHostPointerPropertiesEXT(
            vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, host_ptr);
    } catch (...) {
        return Status::Internal("getMemoryHostPointerPropertiesEXT failed");
    }

    std::uint32_t type_bits = host_props.memoryTypeBits & req.memoryTypeBits;
    if (type_bits == 0) {
        return Status::Internal("no compatible memory type for host pointer import");
    }

    std::uint32_t mem_index = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t i = 0; i < ctx->memory_properties().memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) != 0) {
            mem_index = i;
            break;
        }
    }
    if (mem_index == std::numeric_limits<std::uint32_t>::max()) {
        return Status::Internal("FindMemoryType failed for import");
    }

    vk::MemoryAllocateInfo mai{};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_index;

    vk::ImportMemoryHostPointerInfoEXT import{};
    import.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
    import.pHostPointer = host_ptr;
    mai.pNext = &import;

    vk::UniqueDeviceMemory mem;
    try {
        mem = ctx->device().allocateMemoryUnique(mai);
        ctx->device().bindBufferMemory(buffer.get(), mem.get(), 0);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    out->device = ctx->device();
    out->buffer = std::move(buffer);
    out->mem = std::move(mem);
    out->mapped = host_ptr;
    out->size = bytes.size();
    out->path = HostBufferPath::kImportHostPointer;
    return Status::Ok();
}

}  // namespace

void HostBuffer::Reset(VulkanComputeContext* ctx) {
    (void)ctx;
    if (path == HostBufferPath::kCopyMappedHost && device && mem && mapped) {
        device.unmapMemory(mem.get());
        mapped = nullptr;
    }
    buffer.reset();
    mem.reset();
    device = vk::Device{};
    size = 0;
    path = HostBufferPath::kCopyMappedHost;
}

Status VulkanMemoryBridge::CreateBufferCopyMapped(VulkanComputeContext* ctx,
                                                  std::span<const std::byte> bytes,
                                                  HostBuffer* out) {
    if (!ctx || !out || bytes.empty()) {
        return Status::InvalidArgument("CreateBufferCopyMapped invalid args");
    }

    vk::BufferCreateInfo bci{};
    bci.size = bytes.size();
    bci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc;
    bci.sharingMode = vk::SharingMode::eExclusive;

    vk::UniqueBuffer buffer;
    try {
        buffer = ctx->device().createBufferUnique(bci);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    const vk::MemoryRequirements req = ctx->device().getBufferMemoryRequirements(buffer.get());
    const auto want = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::uint32_t mem_index = FindMemoryType(ctx->memory_properties(), req.memoryTypeBits, want);
    if (mem_index == std::numeric_limits<std::uint32_t>::max()) {
        return Status::Internal("no HOST_VISIBLE|HOST_COHERENT memory for copy-mapped buffer");
    }

    vk::MemoryAllocateInfo mai{};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_index;

    vk::UniqueDeviceMemory mem;
    void* mapped = nullptr;
    try {
        mem = ctx->device().allocateMemoryUnique(mai);
        ctx->device().bindBufferMemory(buffer.get(), mem.get(), 0);
        mapped = ctx->device().mapMemory(mem.get(), 0, req.size);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    std::memcpy(mapped, bytes.data(), bytes.size());

    out->device = ctx->device();
    out->buffer = std::move(buffer);
    out->mem = std::move(mem);
    out->mapped = mapped;
    out->size = bytes.size();
    out->path = HostBufferPath::kCopyMappedHost;
    return Status::Ok();
}

Status VulkanMemoryBridge::CreateBufferFromHostSpan(VulkanComputeContext* ctx,
                                                    std::span<const std::byte> bytes,
                                                    HostBuffer* out) {
    if (!ctx || !out || bytes.empty()) {
        return Status::InvalidArgument("CreateBufferFromHostSpan invalid args");
    }

    if (ctx->ext_external_memory_host() && ctx->bridge_options().prefer_unified_memory &&
        bytes.size() >= ctx->bridge_options().zero_copy_min_bytes) {
        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(bytes.data());
        const std::uint64_t align = ctx->min_host_import_alignment();
        if (align > 0 && (addr % align) == 0) {
            const Status st = TryImportHostPointer(ctx, bytes, out);
            if (st.ok()) {
                return st;
            }
        }
    }
    return CreateBufferCopyMapped(ctx, bytes, out);
}

Status VulkanMemoryBridge::UploadToDeviceBuffer(VulkanComputeContext* ctx,
                                                VulkanStagingPool* pool,
                                                std::span<const std::byte> bytes,
                                                HostBuffer* out_device) {
    if (!ctx || !pool || !out_device || bytes.empty()) {
        return Status::InvalidArgument("UploadToDeviceBuffer invalid args");
    }

    vk::BufferCreateInfo bci{};
    bci.size = bytes.size();
    bci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
    bci.sharingMode = vk::SharingMode::eExclusive;

    vk::UniqueBuffer dev_buf;
    try {
        dev_buf = ctx->device().createBufferUnique(bci);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    const vk::MemoryRequirements req = ctx->device().getBufferMemoryRequirements(dev_buf.get());
    const auto want = vk::MemoryPropertyFlagBits::eDeviceLocal;
    std::uint32_t mem_index = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t i = 0; i < ctx->memory_properties().memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) != 0 &&
            (ctx->memory_properties().memoryTypes[i].propertyFlags & want) == want) {
            mem_index = i;
            break;
        }
    }
    if (mem_index == std::numeric_limits<std::uint32_t>::max()) {
        for (std::uint32_t i = 0; i < ctx->memory_properties().memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) != 0) {
                mem_index = i;
                break;
            }
        }
    }
    if (mem_index == std::numeric_limits<std::uint32_t>::max()) {
        return Status::Internal("no memory type for device-local buffer");
    }

    vk::MemoryAllocateInfo mai{};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_index;

    vk::UniqueDeviceMemory dev_mem;
    try {
        dev_mem = ctx->device().allocateMemoryUnique(mai);
        ctx->device().bindBufferMemory(dev_buf.get(), dev_mem.get(), 0);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    VulkanStagingPool::StagingSlice slice{};
    Status st = pool->Acquire(bytes.size(), 256, &slice);
    if (!st.ok()) {
        return st;
    }
    std::memcpy(slice.mapped, bytes.data(), bytes.size());

    vk::CommandBufferAllocateInfo alloc{};
    alloc.commandPool = ctx->transfer_cmd_pool();
    alloc.level = vk::CommandBufferLevel::ePrimary;
    alloc.commandBufferCount = 1;

    std::vector<vk::CommandBuffer> cmds;
    try {
        cmds = ctx->device().allocateCommandBuffers(alloc);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    vk::CommandBuffer cmd = cmds[0];

    vk::CommandBufferBeginInfo begin{};
    begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    cmd.begin(begin);

    vk::BufferCopy region{};
    region.srcOffset = slice.offset;
    region.dstOffset = 0;
    region.size = bytes.size();
    cmd.copyBuffer(slice.buffer, dev_buf.get(), region);

    cmd.end();

    try {
        ctx->device().resetFences(ctx->transfer_fence());
        vk::SubmitInfo sub{};
        sub.commandBufferCount = 1;
        sub.pCommandBuffers = &cmd;
        ctx->queue().submit({sub}, ctx->transfer_fence());
        (void)ctx->device().waitForFences({ctx->transfer_fence()}, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
        ctx->device().freeCommandBuffers(ctx->transfer_cmd_pool(), cmd);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    out_device->device = ctx->device();
    out_device->buffer = std::move(dev_buf);
    out_device->mem = std::move(dev_mem);
    out_device->mapped = nullptr;
    out_device->size = bytes.size();
    out_device->path = HostBufferPath::kDeviceLocalFromStaging;
    return Status::Ok();
}

std::uint64_t VulkanMemoryBridge::PinHostBuffer(std::shared_ptr<HostBuffer> buf) {
    return pomai::core::GpuBufferPinRegistry::Instance().Pin(std::move(buf));
}

void VulkanMemoryBridge::UnpinHostBuffer(std::uint64_t session_id) {
    pomai::core::GpuBufferPinRegistry::Instance().Unpin(session_id);
}

}  // namespace pomai::compute::vulkan
