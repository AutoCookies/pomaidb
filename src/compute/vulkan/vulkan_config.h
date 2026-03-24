#pragma once

// PomaiDB Vulkan policy:
// - Use internal volk loader as the dispatch backend.
// - Use dynamic dispatch for Vulkan-Hpp.
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif

#include "compute/vulkan/loader/volk.h"
#include <vulkan/vulkan.hpp>

