#include "core/mesh/simplifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

#include "palloc_compat.h"

namespace pomai::mesh {
namespace {

template <typename T>
class PallocArray {
public:
    PallocArray() = default;
    explicit PallocArray(std::size_t n) { reset(n); }
    ~PallocArray() {
        if (data_) palloc_free(data_);
    }

    void reset(std::size_t n) {
        if (data_) palloc_free(data_);
        data_ = nullptr;
        size_ = 0;
        if (n == 0) return;
        data_ = static_cast<T*>(palloc_malloc_aligned(sizeof(T) * n, alignof(T)));
        if (!data_) return;
        size_ = n;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return size_; }
    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

struct CellCoord {
    int x = 0;
    int y = 0;
    int z = 0;
    bool operator==(const CellCoord& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct CellCoordHash {
    std::size_t operator()(const CellCoord& c) const {
        const std::uint64_t h1 = static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.x) * 73856093u);
        const std::uint64_t h2 = static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.y) * 19349663u);
        const std::uint64_t h3 = static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.z) * 83492791u);
        return static_cast<std::size_t>(h1 ^ h2 ^ h3);
    }
};

inline std::size_t TrianglesFromIndices(std::size_t index_count) { return index_count / 3u; }

}  // namespace

Status SimplifyMesh(std::span<const float> vertices_xyz, std::span<const std::uint32_t> indices,
                    std::size_t target_index_count, SimplifyOutput* out) {
    if (!out) return Status::InvalidArgument("simplify output null");
    out->indices.clear();
    out->vertices_xyz.clear();
    if (vertices_xyz.empty() || vertices_xyz.size() % 3u != 0u) return Status::InvalidArgument("invalid vertices");
    if (indices.empty() || indices.size() % 3u != 0u) return Status::InvalidArgument("invalid indices");

    const std::size_t vertex_count = vertices_xyz.size() / 3u;
    const std::size_t source_triangles = TrianglesFromIndices(indices.size());
    const std::size_t target_triangles = std::max<std::size_t>(1u, target_index_count / 3u);
    if (target_triangles >= source_triangles) {
        out->indices.assign(indices.begin(), indices.end());
        out->vertices_xyz.assign(vertices_xyz.begin(), vertices_xyz.end());
        return Status::Ok();
    }

    float minx = vertices_xyz[0], maxx = vertices_xyz[0];
    float miny = vertices_xyz[1], maxy = vertices_xyz[1];
    float minz = vertices_xyz[2], maxz = vertices_xyz[2];
    for (std::size_t i = 3; i < vertices_xyz.size(); i += 3) {
        minx = std::min(minx, vertices_xyz[i]);
        maxx = std::max(maxx, vertices_xyz[i]);
        miny = std::min(miny, vertices_xyz[i + 1]);
        maxy = std::max(maxy, vertices_xyz[i + 1]);
        minz = std::min(minz, vertices_xyz[i + 2]);
        maxz = std::max(maxz, vertices_xyz[i + 2]);
    }

    const float extent_x = std::max(1e-6f, maxx - minx);
    const float extent_y = std::max(1e-6f, maxy - miny);
    const float extent_z = std::max(1e-6f, maxz - minz);
    const float target_ratio = static_cast<float>(target_triangles) / static_cast<float>(source_triangles);
    const float grid_scale = std::max(2.0f, std::cbrt(1.0f / std::max(0.01f, target_ratio)) * 2.0f);

    PallocArray<std::uint32_t> remap(vertex_count);
    if (!remap.data()) return Status::ResourceExhausted("palloc remap alloc failed");
    std::fill(remap.data(), remap.data() + vertex_count, std::numeric_limits<std::uint32_t>::max());

    std::unordered_map<CellCoord, std::uint32_t, CellCoordHash> cluster_to_vertex;
    cluster_to_vertex.reserve(vertex_count / 2u + 1u);

    out->vertices_xyz.reserve(target_triangles * 9u);
    for (std::size_t vi = 0; vi < vertex_count; ++vi) {
        const float x = vertices_xyz[vi * 3u];
        const float y = vertices_xyz[vi * 3u + 1u];
        const float z = vertices_xyz[vi * 3u + 2u];
        const CellCoord key{
            static_cast<int>(((x - minx) / extent_x) * grid_scale),
            static_cast<int>(((y - miny) / extent_y) * grid_scale),
            static_cast<int>(((z - minz) / extent_z) * grid_scale),
        };
        auto it = cluster_to_vertex.find(key);
        if (it == cluster_to_vertex.end()) {
            const std::uint32_t new_id = static_cast<std::uint32_t>(out->vertices_xyz.size() / 3u);
            cluster_to_vertex.emplace(key, new_id);
            out->vertices_xyz.push_back(x);
            out->vertices_xyz.push_back(y);
            out->vertices_xyz.push_back(z);
            remap[vi] = new_id;
        } else {
            remap[vi] = it->second;
        }
    }

    out->indices.reserve(indices.size());
    for (std::size_t i = 0; i < indices.size(); i += 3u) {
        const std::uint32_t a = remap[indices[i]];
        const std::uint32_t b = remap[indices[i + 1u]];
        const std::uint32_t c = remap[indices[i + 2u]];
        if (a == b || b == c || a == c) continue;
        out->indices.push_back(a);
        out->indices.push_back(b);
        out->indices.push_back(c);
        if (out->indices.size() >= target_index_count) break;
    }

    if (out->indices.size() < 3u) {
        out->indices.assign(indices.begin(), indices.begin() + std::min<std::size_t>(indices.size(), target_index_count));
        out->vertices_xyz.assign(vertices_xyz.begin(), vertices_xyz.end());
    }
    return Status::Ok();
}

Status OptimizeIndexOrder(std::span<const std::uint32_t> indices, std::size_t vertex_count,
                          std::vector<std::uint32_t>* out_indices) {
    if (!out_indices) return Status::InvalidArgument("output null");
    if (indices.empty() || indices.size() % 3u != 0u) return Status::InvalidArgument("invalid indices");

    // Single-threaded edge-safe triangle bucket reorder by min vertex id.
    // This improves locality for cache-friendly scans without extra threads.
    struct TriRef {
        std::uint32_t a;
        std::uint32_t b;
        std::uint32_t c;
        std::uint32_t key;
    };
    std::vector<TriRef> tris;
    tris.reserve(indices.size() / 3u);
    for (std::size_t i = 0; i < indices.size(); i += 3u) {
        const std::uint32_t a = indices[i];
        const std::uint32_t b = indices[i + 1u];
        const std::uint32_t c = indices[i + 2u];
        const std::uint32_t k = std::min(a, std::min(b, c));
        tris.push_back({a, b, c, k});
    }
    const std::uint32_t bucket_shift = vertex_count > 0 ? static_cast<std::uint32_t>(std::max<std::size_t>(0, static_cast<std::size_t>(std::log2(vertex_count)) - 10u)) : 0u;
    std::stable_sort(tris.begin(), tris.end(), [bucket_shift](const TriRef& l, const TriRef& r) {
        return (l.key >> bucket_shift) < (r.key >> bucket_shift);
    });

    out_indices->clear();
    out_indices->reserve(indices.size());
    for (const auto& t : tris) {
        out_indices->push_back(t.a);
        out_indices->push_back(t.b);
        out_indices->push_back(t.c);
    }
    return Status::Ok();
}

Status CompactVertexBuffer(std::span<const float> vertices_xyz, std::span<const std::uint32_t> indices,
                           SimplifyOutput* out) {
    if (!out) return Status::InvalidArgument("output null");
    out->indices.clear();
    out->vertices_xyz.clear();
    if (vertices_xyz.empty() || vertices_xyz.size() % 3u != 0u) return Status::InvalidArgument("invalid vertices");
    if (indices.empty() || indices.size() % 3u != 0u) return Status::InvalidArgument("invalid indices");
    const std::size_t vertex_count = vertices_xyz.size() / 3u;

    PallocArray<std::uint32_t> remap(vertex_count);
    if (!remap.data()) return Status::ResourceExhausted("palloc remap alloc failed");
    std::fill(remap.data(), remap.data() + vertex_count, std::numeric_limits<std::uint32_t>::max());

    out->indices.reserve(indices.size());
    out->vertices_xyz.reserve(vertices_xyz.size());
    for (std::uint32_t old_i : indices) {
        if (old_i >= vertex_count) return Status::InvalidArgument("index out of range");
        std::uint32_t& mapped = remap[old_i];
        if (mapped == std::numeric_limits<std::uint32_t>::max()) {
            mapped = static_cast<std::uint32_t>(out->vertices_xyz.size() / 3u);
            const std::size_t base = static_cast<std::size_t>(old_i) * 3u;
            out->vertices_xyz.push_back(vertices_xyz[base]);
            out->vertices_xyz.push_back(vertices_xyz[base + 1u]);
            out->vertices_xyz.push_back(vertices_xyz[base + 2u]);
        }
        out->indices.push_back(mapped);
    }
    return Status::Ok();
}

}  // namespace pomai::mesh

