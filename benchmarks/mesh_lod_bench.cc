#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#include "pomai/pomai.h"

namespace {

std::vector<float> BuildMesh(std::size_t triangles) {
    std::vector<float> xyz;
    xyz.reserve(triangles * 9u);
    for (std::size_t i = 0; i < triangles; ++i) {
        const float x = static_cast<float>(i % 256u);
        const float y = static_cast<float>(i / 256u);
        xyz.insert(xyz.end(), {x, y, 0.0f, x + 0.5f, y, 0.0f, x, y + 0.5f, 0.0f});
    }
    return xyz;
}

double RunVolumeLoop(pomai::DB* db, const pomai::MeshQueryOptions& q, std::size_t iters) {
    double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iters; ++i) {
        double v = 0.0;
        (void)db->MeshVolume("mesh", 1, q, &v);
        sink += v;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1000.0;
    if (sink < 0.0) std::fprintf(stderr, "sink=%f\n", sink);
    return ms;
}

}  // namespace

int main() {
    pomai::DBOptions opt;
    opt.path = "/tmp/pomaidb_mesh_lod_bench";
    opt.dim = 4;
    opt.shard_count = 1;
    opt.mesh_lod_jobs_per_tick = 8;
    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) return 1;

    pomai::MembraneSpec mesh;
    mesh.name = "mesh";
    mesh.kind = pomai::MembraneKind::kMesh;
    mesh.shard_count = 1;
    st = db->CreateMembrane(mesh);
    if (!st.ok() && st.code() != pomai::ErrorCode::kAlreadyExists) return 1;
    st = db->OpenMembrane("mesh");
    if (!st.ok()) return 1;

    const auto verts = BuildMesh(4096);
    st = db->MeshPut("mesh", 1, verts);
    if (!st.ok()) return 1;
    // Pump a few calls so scheduler builds LOD.
    for (int i = 0; i < 8; ++i) {
        double tmp = 0.0;
        (void)db->MeshVolume("mesh", 1, &tmp);
    }

    pomai::MeshQueryOptions auto_q;
    auto_q.detail = pomai::MeshDetailPreference::kAutoLatencyFirst;
    pomai::MeshQueryOptions hi_q;
    hi_q.detail = pomai::MeshDetailPreference::kHighDetail;

    const std::size_t iters = 5000;
    const double auto_ms = RunVolumeLoop(db.get(), auto_q, iters);
    const double hi_ms = RunVolumeLoop(db.get(), hi_q, iters);
    std::printf("mesh_lod_bench iterations=%zu auto_latency_ms=%.3f high_detail_ms=%.3f\n", iters, auto_ms, hi_ms);

    (void)db->Close();
    return 0;
}

