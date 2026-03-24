#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pomai/status.h"

namespace pomai::mesh {

struct SimplifyOutput {
    std::vector<std::uint32_t> indices;
    std::vector<float> vertices_xyz;
};

// Edge-first internal API: simplify triangle mesh and compact buffers.
Status SimplifyMesh(std::span<const float> vertices_xyz, std::span<const std::uint32_t> indices,
                    std::size_t target_index_count, SimplifyOutput* out);

// Reorder triangles for better locality in CPU traversal.
Status OptimizeIndexOrder(std::span<const std::uint32_t> indices, std::size_t vertex_count,
                          std::vector<std::uint32_t>* out_indices);

// Remove unused vertices and remap indices.
Status CompactVertexBuffer(std::span<const float> vertices_xyz, std::span<const std::uint32_t> indices,
                           SimplifyOutput* out);

}  // namespace pomai::mesh

