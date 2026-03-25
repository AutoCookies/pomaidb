#pragma once

#include <cstdint>
#include <cstddef>
#include <deque>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/status.h"

namespace pomai::core {

class MeshEngine {
public:
    MeshEngine(std::string path, std::size_t max_meshes, const pomai::DBOptions& db_options);
    Status Open();
    Status Close();
    Status Put(std::uint64_t mesh_id, std::span<const float> xyz);
    Status Rmsd(std::uint64_t mesh_a, std::uint64_t mesh_b, double* out) const;
    Status Intersect(std::uint64_t mesh_a, std::uint64_t mesh_b, bool* out) const;
    Status Volume(std::uint64_t mesh_id, double* out) const;
    Status Rmsd(std::uint64_t mesh_a, std::uint64_t mesh_b, const MeshQueryOptions& opts, double* out) const;
    Status Intersect(std::uint64_t mesh_a, std::uint64_t mesh_b, const MeshQueryOptions& opts, bool* out) const;
    Status Volume(std::uint64_t mesh_id, const MeshQueryOptions& opts, double* out) const;

    void QueueLodBuild(std::uint64_t mesh_id);
    Status ProcessLodJobs(std::size_t max_jobs);
    std::size_t PendingLodJobs() const { return lod_queue_.size(); }

    void ForEach(const std::function<void(std::uint64_t id, std::size_t base_floats)>& fn) const;

private:
    struct MeshRecord {
        std::vector<float> base_xyz;
        std::unordered_map<std::uint8_t, std::vector<float>> lod_xyz;
        std::uint64_t revision = 0;
    };

    const std::vector<float>* ResolveMesh(std::uint64_t mesh_id, const MeshQueryOptions& opts) const;
    std::uint8_t ChooseLodLevel(const MeshRecord& rec, const MeshQueryOptions& opts) const;
    Status AppendBaseRecord(std::uint64_t mesh_id, std::span<const float> xyz, std::uint64_t revision);
    Status AppendLodRecord(std::uint64_t mesh_id, std::uint64_t revision, std::uint8_t level, std::span<const float> xyz);

    std::string path_;
    std::size_t max_meshes_;
    pomai::DBOptions db_options_;
    std::unordered_map<std::uint64_t, MeshRecord> meshes_;
    std::deque<std::uint64_t> lod_queue_;
};

} // namespace pomai::core

