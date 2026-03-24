#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pomai/options.h"
#include "pomai/status.h"

namespace pomai::core {

struct MeshLodLevel {
    std::uint8_t level = 0;  // 0 is base (high-poly), 1..N are decimated.
    std::vector<float> xyz;
};

class MeshLodManager {
public:
    // Builds [base, lod1, lod2, ...] from input xyz vertices.
    static Status BuildLods(std::span<const float> xyz,
                            const pomai::DBOptions& opts,
                            std::vector<MeshLodLevel>* out);
};

}  // namespace pomai::core

