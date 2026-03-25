#include "core/mesh/mesh_engine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>

#include "core/mesh/mesh_lod_manager.h"
#include "core/simd/simd_dispatch.h"

namespace pomai::core {
namespace {

constexpr std::uint8_t kBaseTag = 0xA1;
constexpr std::uint8_t kLodTag = 0xA2;

void PushUnique(std::deque<std::uint64_t>* q, std::uint64_t id) {
    if (!q) return;
    if (std::find(q->begin(), q->end(), id) == q->end()) q->push_back(id);
}

void BBox(const std::vector<float>& m, double* minx, double* miny, double* minz, double* maxx, double* maxy,
          double* maxz) {
    *minx = *maxx = m[0];
    *miny = *maxy = m[1];
    *minz = *maxz = m[2];
    for (std::size_t i = 3; i < m.size(); i += 3) {
        *minx = std::min<double>(*minx, m[i]);
        *miny = std::min<double>(*miny, m[i + 1]);
        *minz = std::min<double>(*minz, m[i + 2]);
        *maxx = std::max<double>(*maxx, m[i]);
        *maxy = std::max<double>(*maxy, m[i + 1]);
        *maxz = std::max<double>(*maxz, m[i + 2]);
    }
}

}  // namespace

MeshEngine::MeshEngine(std::string path, std::size_t max_meshes, const pomai::DBOptions& db_options)
    : path_(std::move(path)), max_meshes_(max_meshes), db_options_(db_options) {}

Status MeshEngine::Open() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path_, ec);
    if (ec) return Status::IOError("mesh create dir failed");

    // Legacy format: [id][n][xyz...]
    {
        std::ifstream in(path_ + "/mesh.log", std::ios::binary);
        if (in.good()) {
            while (true) {
                std::uint64_t id = 0;
                std::uint32_t n = 0;
                if (!in.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
                if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) break;
                std::vector<float> xyz(n);
                if (n > 0 &&
                    !in.read(reinterpret_cast<char*>(xyz.data()), static_cast<std::streamsize>(n * sizeof(float)))) {
                    break;
                }
                auto& rec = meshes_[id];
                rec.base_xyz = std::move(xyz);
                rec.lod_xyz.clear();
                ++rec.revision;
                if (max_meshes_ > 0 && meshes_.size() > max_meshes_) meshes_.erase(meshes_.begin());
            }
        }
    }

    // New LOD log format (versioned): [tag][id][revision][level][n][xyz...]
    {
        std::ifstream in(path_ + "/mesh_lod.log", std::ios::binary);
        if (!in.good()) return Status::Ok();
        while (true) {
            std::uint8_t tag = 0;
            std::uint64_t id = 0;
            std::uint64_t rev = 0;
            std::uint8_t level = 0;
            std::uint32_t n = 0;
            if (!in.read(reinterpret_cast<char*>(&tag), sizeof(tag))) break;
            if (!in.read(reinterpret_cast<char*>(&id), sizeof(id))) break;
            if (!in.read(reinterpret_cast<char*>(&rev), sizeof(rev))) break;
            if (!in.read(reinterpret_cast<char*>(&level), sizeof(level))) break;
            if (!in.read(reinterpret_cast<char*>(&n), sizeof(n))) break;
            std::vector<float> xyz(n);
            if (n > 0 &&
                !in.read(reinterpret_cast<char*>(xyz.data()), static_cast<std::streamsize>(n * sizeof(float)))) {
                break;
            }
            auto& rec = meshes_[id];
            if (rev < rec.revision) continue;
            rec.revision = rev;
            if (tag == kBaseTag || level == 0) {
                rec.base_xyz = std::move(xyz);
                rec.lod_xyz.clear();
            } else if (tag == kLodTag) {
                rec.lod_xyz[level] = std::move(xyz);
            }
        }
    }
    return Status::Ok();
}

Status MeshEngine::Close() { return Status::Ok(); }

Status MeshEngine::Put(std::uint64_t mesh_id, std::span<const float> xyz) {
    if (xyz.empty() || xyz.size() % 3 != 0) return Status::InvalidArgument("mesh xyz must be multiple of 3");
    if (max_meshes_ > 0 && meshes_.size() >= max_meshes_ && meshes_.find(mesh_id) == meshes_.end()) meshes_.erase(meshes_.begin());

    auto& rec = meshes_[mesh_id];
    rec.base_xyz.assign(xyz.begin(), xyz.end());
    rec.lod_xyz.clear();
    ++rec.revision;
    auto st = AppendBaseRecord(mesh_id, xyz, rec.revision);
    if (!st.ok()) return st;
    QueueLodBuild(mesh_id);
    return Status::Ok();
}

Status MeshEngine::AppendBaseRecord(std::uint64_t mesh_id, std::span<const float> xyz, std::uint64_t revision) {
    std::ofstream out_legacy(path_ + "/mesh.log", std::ios::binary | std::ios::app);
    if (!out_legacy.good()) return Status::IOError("mesh append failed");
    std::uint32_t n = static_cast<std::uint32_t>(xyz.size());
    out_legacy.write(reinterpret_cast<const char*>(&mesh_id), sizeof(mesh_id));
    out_legacy.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out_legacy.write(reinterpret_cast<const char*>(xyz.data()), static_cast<std::streamsize>(xyz.size_bytes()));

    std::ofstream out(path_ + "/mesh_lod.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("mesh lod append failed");
    const std::uint8_t tag = kBaseTag;
    const std::uint8_t level = 0;
    out.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    out.write(reinterpret_cast<const char*>(&mesh_id), sizeof(mesh_id));
    out.write(reinterpret_cast<const char*>(&revision), sizeof(revision));
    out.write(reinterpret_cast<const char*>(&level), sizeof(level));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(xyz.data()), static_cast<std::streamsize>(xyz.size_bytes()));
    return Status::Ok();
}

Status MeshEngine::AppendLodRecord(std::uint64_t mesh_id, std::uint64_t revision, std::uint8_t level,
                                   std::span<const float> xyz) {
    std::ofstream out(path_ + "/mesh_lod.log", std::ios::binary | std::ios::app);
    if (!out.good()) return Status::IOError("mesh lod append failed");
    const std::uint8_t tag = kLodTag;
    const std::uint32_t n = static_cast<std::uint32_t>(xyz.size());
    out.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
    out.write(reinterpret_cast<const char*>(&mesh_id), sizeof(mesh_id));
    out.write(reinterpret_cast<const char*>(&revision), sizeof(revision));
    out.write(reinterpret_cast<const char*>(&level), sizeof(level));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(xyz.data()), static_cast<std::streamsize>(xyz.size_bytes()));
    return Status::Ok();
}

void MeshEngine::QueueLodBuild(std::uint64_t mesh_id) {
    if (db_options_.mesh_lod_max_queue > 0 && lod_queue_.size() >= db_options_.mesh_lod_max_queue) return;
    PushUnique(&lod_queue_, mesh_id);
}

Status MeshEngine::ProcessLodJobs(std::size_t max_jobs) {
    std::size_t budget = max_jobs == 0 ? 1 : max_jobs;
    while (budget > 0 && !lod_queue_.empty()) {
        const std::uint64_t mesh_id = lod_queue_.front();
        lod_queue_.pop_front();
        auto it = meshes_.find(mesh_id);
        if (it == meshes_.end()) {
            --budget;
            continue;
        }
        std::vector<MeshLodLevel> lods;
        auto st = MeshLodManager::BuildLods(it->second.base_xyz, db_options_, &lods);
        if (!st.ok()) {
            --budget;
            continue;
        }
        const std::uint64_t rev = it->second.revision;
        it->second.lod_xyz.clear();
        for (const auto& lod : lods) {
            if (lod.level == 0) continue;
            it->second.lod_xyz[lod.level] = lod.xyz;
            (void)AppendLodRecord(mesh_id, rev, lod.level, lod.xyz);
        }
        --budget;
    }
    return Status::Ok();
}

std::uint8_t MeshEngine::ChooseLodLevel(const MeshRecord& rec, const MeshQueryOptions& opts) const {
    if (opts.detail == pomai::MeshDetailPreference::kHighDetail) return 0;
    if (rec.lod_xyz.empty()) return 0;
    // latency-first default: choose the lowest-detail available.
    std::uint8_t chosen = 0;
    for (const auto& kv : rec.lod_xyz) chosen = std::max(chosen, kv.first);
    return chosen;
}

const std::vector<float>* MeshEngine::ResolveMesh(std::uint64_t mesh_id, const MeshQueryOptions& opts) const {
    auto it = meshes_.find(mesh_id);
    if (it == meshes_.end()) return nullptr;
    const std::uint8_t level = ChooseLodLevel(it->second, opts);
    if (level == 0) return &it->second.base_xyz;
    auto lit = it->second.lod_xyz.find(level);
    if (lit == it->second.lod_xyz.end()) return &it->second.base_xyz;
    return &lit->second;
}

Status MeshEngine::Rmsd(std::uint64_t mesh_a, std::uint64_t mesh_b, double* out) const {
    return Rmsd(mesh_a, mesh_b, MeshQueryOptions{}, out);
}

Status MeshEngine::Rmsd(std::uint64_t mesh_a, std::uint64_t mesh_b, const MeshQueryOptions& opts, double* out) const {
    if (!out) return Status::InvalidArgument("mesh out null");
    const auto* a = ResolveMesh(mesh_a, opts);
    const auto* b = ResolveMesh(mesh_b, opts);
    if (!a || !b) return Status::NotFound("mesh not found");
    const std::size_t n = std::min(a->size(), b->size());
    if (n < 3) return Status::InvalidArgument("mesh size mismatch");
    *out = simd::MeshRmsdF32(a->data(), b->data(), n / 3u);
    return Status::Ok();
}

Status MeshEngine::Intersect(std::uint64_t mesh_a, std::uint64_t mesh_b, bool* out) const {
    return Intersect(mesh_a, mesh_b, MeshQueryOptions{}, out);
}

Status MeshEngine::Intersect(std::uint64_t mesh_a, std::uint64_t mesh_b, const MeshQueryOptions& opts,
                             bool* out) const {
    if (!out) return Status::InvalidArgument("mesh out null");
    const auto* a = ResolveMesh(mesh_a, opts);
    const auto* b = ResolveMesh(mesh_b, opts);
    if (!a || !b) return Status::NotFound("mesh not found");
    if (a->size() < 3 || b->size() < 3) return Status::InvalidArgument("mesh empty");
    double a0,a1,a2,a3,a4,a5,b0,b1,b2,b3,b4,b5;
    BBox(*a, &a0, &a1, &a2, &a3, &a4, &a5);
    BBox(*b, &b0, &b1, &b2, &b3, &b4, &b5);
    *out = (a0 <= b3 && a3 >= b0) && (a1 <= b4 && a4 >= b1) && (a2 <= b5 && a5 >= b2);
    return Status::Ok();
}

Status MeshEngine::Volume(std::uint64_t mesh_id, double* out) const {
    return Volume(mesh_id, MeshQueryOptions{}, out);
}

Status MeshEngine::Volume(std::uint64_t mesh_id, const MeshQueryOptions& opts, double* out) const {
    if (!out) return Status::InvalidArgument("mesh out null");
    const auto* m = ResolveMesh(mesh_id, opts);
    if (!m) return Status::NotFound("mesh not found");
    if (m->size() < 3) return Status::InvalidArgument("mesh empty");
    double minx = 0.0, miny = 0.0, minz = 0.0, maxx = 0.0, maxy = 0.0, maxz = 0.0;
    BBox(*m, &minx, &miny, &minz, &maxx, &maxy, &maxz);
    *out = (maxx - minx) * (maxy - miny) * (maxz - minz);
    return Status::Ok();
}

void MeshEngine::ForEach(const std::function<void(std::uint64_t id, std::size_t base_floats)>& fn) const {
    for (const auto& [id, rec] : meshes_) fn(id, rec.base_xyz.size());
}

} // namespace pomai::core

