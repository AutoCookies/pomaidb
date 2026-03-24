#include "compute/vulkan/vulkan_device_context.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include "compute/vulkan/vulkan_init.h"

#include "compute/vulkan/loader/volk.h"

namespace pomai::compute::vulkan {

namespace {

constexpr const char kExtExternalMemoryHost[] = "VK_EXT_external_memory_host";

uint32_t FindComputeTransferQueueFamily(vk::PhysicalDevice pd) {
    const auto props = pd.getQueueFamilyProperties();
    for (uint32_t i = 0; i < props.size(); ++i) {
        const auto flags = props[i].queueFlags;
        if ((flags & vk::QueueFlagBits::eCompute) && (flags & vk::QueueFlagBits::eTransfer)) {
            return i;
        }
    }
    for (uint32_t i = 0; i < props.size(); ++i) {
        if (props[i].queueFlags & vk::QueueFlagBits::eCompute) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

bool DeviceHasExtension(vk::PhysicalDevice pd, const char* name) {
    const auto exts = pd.enumerateDeviceExtensionProperties();
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName.data(), name) == 0) {
            return true;
        }
    }
    return false;
}

void SortDevicesIntegratedFirst(std::vector<vk::PhysicalDevice>* devices) {
    std::sort(devices->begin(), devices->end(), [](vk::PhysicalDevice a, vk::PhysicalDevice b) {
        const auto pa = a.getProperties();
        const auto pb = b.getProperties();
        const bool ai = (pa.deviceType == vk::PhysicalDeviceType::eIntegratedGpu);
        const bool bi = (pb.deviceType == vk::PhysicalDeviceType::eIntegratedGpu);
        if (ai != bi) {
            return ai && !bi;
        }
        return pa.deviceID < pb.deviceID;
    });
}

}  // namespace

Status VulkanComputeContext::Create(const BridgeOptions& opts, VulkanComputeContext* out) {
    if (!out) {
        return Status::InvalidArgument("VulkanComputeContext::Create out null");
    }
    out->bridge_opts_ = opts;

    vk::UniqueInstance inst;
    Status st = InitVulkanInstance(&inst);
    if (!st.ok()) {
        return st;
    }
    out->instance_ = std::move(inst);

    std::vector<vk::PhysicalDevice> devices;
    st = EnumeratePhysicalDevices(out->instance_.get(), &devices);
    if (!st.ok()) {
        out->instance_.reset();
        return st;
    }

    SortDevicesIntegratedFirst(&devices);

    vk::PhysicalDevice chosen{};
    uint32_t qfam = std::numeric_limits<uint32_t>::max();
    bool found = false;
    for (auto pd : devices) {
        const uint32_t q = FindComputeTransferQueueFamily(pd);
        if (q == std::numeric_limits<uint32_t>::max()) {
            continue;
        }
        chosen = pd;
        qfam = q;
        found = true;
        break;
    }
    if (!found) {
        out->instance_.reset();
        return Status::NotFound("no Vulkan physical device with compute queue");
    }

    out->physical_device_ = chosen;
    out->queue_family_index_ = qfam;
    out->memory_properties_ = chosen.getMemoryProperties();

    const auto props = chosen.getProperties();
    out->capabilities_.integrated_gpu = (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu);
    out->capabilities_.memory_heap_count = out->memory_properties_.memoryHeapCount;
    out->capabilities_.memory_type_count = out->memory_properties_.memoryTypeCount;

    out->ext_external_memory_host_ = DeviceHasExtension(chosen, kExtExternalMemoryHost);
    out->capabilities_.ext_external_memory_host = out->ext_external_memory_host_;

    if (out->ext_external_memory_host_) {
        vk::PhysicalDeviceExternalMemoryHostPropertiesEXT host_ext{};
        vk::PhysicalDeviceProperties2 props2{};
        props2.pNext = &host_ext;
        chosen.getProperties2(&props2);
        out->min_host_import_alignment_ = host_ext.minImportedHostPointerAlignment;
        if (out->min_host_import_alignment_ == 0) {
            out->min_host_import_alignment_ = 1;
        }
        out->capabilities_.min_host_alignment = out->min_host_import_alignment_;
    } else {
        out->min_host_import_alignment_ = 1;
        out->capabilities_.min_host_alignment = 1;
    }

    float priority = 1.0f;
    vk::DeviceQueueCreateInfo qci{};
    qci.queueFamilyIndex = qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    std::vector<const char*> device_exts;
    if (out->ext_external_memory_host_) {
        device_exts.push_back(kExtExternalMemoryHost);
    }

    vk::DeviceCreateInfo dci{};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
    dci.ppEnabledExtensionNames = device_exts.empty() ? nullptr : device_exts.data();

    try {
        out->device_ = chosen.createDeviceUnique(dci);
    } catch (const vk::SystemError& e) {
        out->instance_.reset();
        std::string msg = "vkCreateDevice failed: ";
        msg += e.what();
        return Status::Internal(msg);
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(out->device_.get());
    volkLoadDevice(static_cast<VkDevice>(out->device_.get()));

    out->queue_ = out->device_->getQueue(qfam, 0);

    try {
        vk::CommandPoolCreateInfo cpci{};
        cpci.queueFamilyIndex = qfam;
        out->transfer_cmd_pool_ = out->device_->createCommandPoolUnique(cpci);

        vk::FenceCreateInfo fci{};
        out->transfer_fence_ = out->device_->createFenceUnique(fci);
    } catch (const vk::SystemError& e) {
        out->device_.reset();
        out->instance_.reset();
        std::string msg = "Vulkan command pool/fence failed: ";
        msg += e.what();
        return Status::Internal(msg);
    }

    return Status::Ok();
}

}  // namespace pomai::compute::vulkan
