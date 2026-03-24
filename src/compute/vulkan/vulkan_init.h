#pragma once

#include <cstdint>
#include <vector>

#include "compute/vulkan/vulkan_config.h"
#include "pomai/status.h"

namespace pomai::compute::vulkan {

Status InitVulkanInstance(vk::UniqueInstance* out_instance);
Status EnumeratePhysicalDevices(const vk::Instance& instance, std::vector<vk::PhysicalDevice>* out_devices);
Status BuildQueueFamilyScratch(std::size_t count, std::uint32_t** out_indices);

}  // namespace pomai::compute::vulkan

