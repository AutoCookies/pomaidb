#include "compute/vulkan/vulkan_staging_pool.h"

#include <algorithm>
#include <limits>

#include "compute/vulkan/vulkan_device_context.h"

namespace pomai::compute::vulkan {

VulkanStagingPool::~VulkanStagingPool() {
    if (ctx_ && memory_ && mapped_base_) {
        ctx_->device().unmapMemory(memory_.get());
        mapped_base_ = nullptr;
    }
}

namespace {

constexpr std::uint64_t kAlign = 256;

std::uint64_t AlignUp(std::uint64_t v, std::uint64_t a) {
    return (v + a - 1u) / a * a;
}

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

}  // namespace

Status VulkanStagingPool::Create(VulkanComputeContext* ctx, std::uint64_t total_bytes, VulkanStagingPool* out) {
    if (!ctx || !out || total_bytes == 0) {
        return Status::InvalidArgument("VulkanStagingPool::Create invalid args");
    }
    out->ctx_ = ctx;
    out->capacity_ = total_bytes;
    out->cursor_ = 0;
    out->telemetry_ = {};

    vk::BufferCreateInfo bci{};
    bci.size = total_bytes;
    bci.usage = vk::BufferUsageFlagBits::eTransferSrc;
    bci.sharingMode = vk::SharingMode::eExclusive;

    try {
        out->buffer_ = ctx->device().createBufferUnique(bci);
    } catch (const vk::SystemError& e) {
        return Status::Internal(e.what());
    }

    const vk::MemoryRequirements req = ctx->device().getBufferMemoryRequirements(out->buffer_.get());
    const auto props = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const std::uint32_t mem_index =
        FindMemoryType(ctx->memory_properties(), req.memoryTypeBits, props);
    if (mem_index == std::numeric_limits<std::uint32_t>::max()) {
        out->buffer_.reset();
        return Status::Internal("no HOST_VISIBLE|HOST_COHERENT memory type for staging");
    }

    vk::MemoryAllocateInfo mai{};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_index;

    try {
        out->memory_ = ctx->device().allocateMemoryUnique(mai);
        ctx->device().bindBufferMemory(out->buffer_.get(), out->memory_.get(), 0);
        out->mapped_base_ = ctx->device().mapMemory(out->memory_.get(), 0, req.size);
    } catch (const vk::SystemError& e) {
        out->memory_.reset();
        out->buffer_.reset();
        return Status::Internal(e.what());
    }

    return Status::Ok();
}

Status VulkanStagingPool::Acquire(std::uint64_t size, std::uint64_t alignment, StagingSlice* out_slice) {
    if (!out_slice || !ctx_ || !buffer_.get() || size == 0) {
        return Status::InvalidArgument("VulkanStagingPool::Acquire invalid args");
    }
    const std::uint64_t align = std::max(alignment, kAlign);
    const std::uint64_t aligned_size = AlignUp(size, align);
    if (aligned_size > capacity_) {
        telemetry_.acquire_failures++;
        return Status::ResourceExhausted("staging request larger than pool capacity");
    }

    std::uint64_t pos = AlignUp(cursor_, align);
    if (pos + aligned_size > capacity_) {
        try {
            ctx_->device().waitIdle();
        } catch (const vk::SystemError& e) {
            telemetry_.acquire_failures++;
            return Status::Internal(e.what());
        }
        telemetry_.wrap_count++;
        pos = 0;
        cursor_ = 0;
    }
    if (pos + aligned_size > capacity_) {
        telemetry_.acquire_failures++;
        return Status::ResourceExhausted("staging acquire after wrap still too large");
    }

    StagingSlice s{};
    s.buffer = buffer_.get();
    s.offset = pos;
    s.size = aligned_size;
    s.mapped = static_cast<char*>(mapped_base_) + static_cast<std::ptrdiff_t>(pos);

    cursor_ = pos + aligned_size;
    telemetry_.acquires++;
    telemetry_.peak_used = std::max(telemetry_.peak_used, cursor_);

    *out_slice = s;
    return Status::Ok();
}

}  // namespace pomai::compute::vulkan
