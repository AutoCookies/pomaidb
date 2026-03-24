#include "core/mesh/mesh_lod_manager.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "core/mesh/simplifier.h"

namespace pomai::core {
namespace {

std::size_t ClampTargetVertexCount(std::size_t source_vertices, std::size_t target) {
    if (source_vertices < 3) return source_vertices;
    const std::size_t clamped = std::max<std::size_t>(3, std::min(target, source_vertices));
    return (clamped / 3u) * 3u;
}

std::vector<unsigned int> BuildTriangleIndices(std::size_t vertex_count) {
    const std::size_t triangle_vertices = (vertex_count / 3u) * 3u;
    std::vector<unsigned int> indices(triangle_vertices);
    std::iota(indices.begin(), indices.end(), 0u);
    return indices;
}

std::vector<float> BuildVertexBufferFromIndices(const std::vector<unsigned int>& indices,
                                                std::span<const float> xyz) {
    std::vector<float> out;
    out.reserve(indices.size() * 3u);
    for (unsigned int idx : indices) {
        const std::size_t i = static_cast<std::size_t>(idx) * 3u;
        if (i + 2u >= xyz.size()) break;
        out.push_back(xyz[i]);
        out.push_back(xyz[i + 1u]);
        out.push_back(xyz[i + 2u]);
    }
    return out;
}

}  // namespace

Status MeshLodManager::BuildLods(std::span<const float> xyz, const pomai::DBOptions& opts,
                                 std::vector<MeshLodLevel>* out) {
    if (!out) return Status::InvalidArgument("lod output null");
    out->clear();
    if (xyz.empty() || xyz.size() % 3u != 0u) {
        return Status::InvalidArgument("mesh xyz must be multiple of 3");
    }

    const std::size_t vertex_count = xyz.size() / 3u;
    MeshLodLevel base;
    base.level = 0;
    base.xyz.assign(xyz.begin(), xyz.end());
    out->push_back(std::move(base));
    if (vertex_count < 9u) return Status::Ok();

    const std::vector<unsigned int> src_indices = BuildTriangleIndices(vertex_count);
    if (src_indices.size() < 6u) return Status::Ok();

    // Edge-friendly default profile: 50% then 25% vertex budget.
    // If DB is configured with very small mesh capacity, build only one LOD.
    std::vector<double> ratios = {0.35, 0.15};
    if (opts.max_mesh_objects <= 1024u) ratios.resize(1);

    std::vector<std::uint32_t> current_indices(src_indices.begin(), src_indices.end());
    std::uint8_t lod_level = 1;
    for (double ratio : ratios) {
        const std::size_t target_vertex_count = ClampTargetVertexCount(
            vertex_count, static_cast<std::size_t>(std::llround(static_cast<double>(vertex_count) * ratio)));
        const std::size_t target_index_count = (target_vertex_count / 3u) * 3u;
        if (target_index_count < 3u || target_index_count >= current_indices.size()) continue;

        pomai::mesh::SimplifyOutput simplified;
        auto st = pomai::mesh::SimplifyMesh(xyz, current_indices, target_index_count, &simplified);
        if (!st.ok() || simplified.indices.size() < 3u) continue;
        if (simplified.indices.size() >= current_indices.size()) continue;

        std::vector<std::uint32_t> optimized;
        st = pomai::mesh::OptimizeIndexOrder(simplified.indices, simplified.vertices_xyz.size() / 3u, &optimized);
        if (!st.ok()) continue;

        pomai::mesh::SimplifyOutput compacted;
        st = pomai::mesh::CompactVertexBuffer(simplified.vertices_xyz, optimized, &compacted);
        if (!st.ok()) continue;

        std::span<const float> final_xyz(compacted.vertices_xyz.data(), compacted.vertices_xyz.size());

        MeshLodLevel lod;
        lod.level = lod_level++;
        lod.xyz = BuildVertexBufferFromIndices(compacted.indices, final_xyz);
        if (lod.xyz.size() < 9u || lod.xyz.size() >= out->back().xyz.size()) continue;
        out->push_back(std::move(lod));
        current_indices = std::move(compacted.indices);
    }

    return Status::Ok();
}

}  // namespace pomai::core

