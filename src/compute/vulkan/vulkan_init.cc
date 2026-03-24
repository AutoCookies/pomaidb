#include "compute/vulkan/vulkan_config.h"

#include <vector>

#include "palloc_compat.h"
#include "pomai/status.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace pomai::compute::vulkan {

Status InitVulkanInstance(vk::UniqueInstance* out_instance) {
    if (!out_instance) return Status::InvalidArgument("out_instance null");
    if (volkInitialize() != VK_SUCCESS) return Status::Internal("volkInitialize failed");
    if (!vkGetInstanceProcAddr) return Status::Internal("vkGetInstanceProcAddr missing");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    vk::ApplicationInfo app_info;
    app_info.pApplicationName = "PomaiDB";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "PomaiDB";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;

    vk::InstanceCreateInfo create_info;
    create_info.pApplicationInfo = &app_info;

    *out_instance = vk::createInstanceUnique(create_info);
    if (!(*out_instance)) return Status::Internal("vk::createInstanceUnique failed");

    volkLoadInstance(static_cast<VkInstance>(out_instance->get()));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(out_instance->get());
    return Status::Ok();
}

Status EnumeratePhysicalDevices(const vk::Instance& instance, std::vector<vk::PhysicalDevice>* out_devices) {
    if (!out_devices) return Status::InvalidArgument("out_devices null");
    *out_devices = instance.enumeratePhysicalDevices();
    return out_devices->empty() ? Status::NotFound("no Vulkan physical devices found") : Status::Ok();
}

// Example host-side wrapper that uses palloc for temporary scratch.
Status BuildQueueFamilyScratch(std::size_t count, std::uint32_t** out_indices) {
    if (!out_indices) return Status::InvalidArgument("out_indices null");
    *out_indices = static_cast<std::uint32_t*>(
        palloc_malloc_aligned(sizeof(std::uint32_t) * count, alignof(std::uint32_t)));
    if (!*out_indices) return Status::ResourceExhausted("palloc scratch alloc failed");
    return Status::Ok();
}

}  // namespace pomai::compute::vulkan

