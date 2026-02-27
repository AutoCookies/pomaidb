// hnsw_index.cc — HnswIndex implementation wrapping faiss::IndexHNSWFlat.
//
// Phases 3 & 4:
//  - Phase 3: Add/Search delegation to faiss::IndexHNSWFlat
//  - Phase 4: Save/Load via faiss::write_index / faiss::read_index

#include "core/index/hnsw_index.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

// Pull complete FAISS headers only in this TU
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

namespace pomai::index {

// ── Constructor / Destructor ──────────────────────────────────────────────────
HnswIndex::HnswIndex(uint32_t dim, HnswOptions opts)
    : dim_(dim), opts_(opts)
{
    // faiss::IndexHNSWFlat(d, M) — uses L2 by default; we switch to IP below
    // based on pomai convention (inner product by default).
    index_ = std::make_unique<faiss::IndexHNSWFlat>(
        static_cast<int>(dim_), opts_.M, faiss::METRIC_INNER_PRODUCT);
    index_->hnsw.efConstruction = opts_.ef_construction;
    index_->hnsw.efSearch       = opts_.ef_search;
}

HnswIndex::~HnswIndex() = default;

// ── Build Phase ───────────────────────────────────────────────────────────────
pomai::Status HnswIndex::Add(VectorId id, std::span<const float> vec)
{
    if (vec.size() != dim_)
        return pomai::Status::InvalidArgument("vector dim mismatch");
    index_->add(1, vec.data());
    id_map_.push_back(id);
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::AddBatch(const VectorId* ids,
                                   const float*    vecs,
                                   std::size_t     n)
{
    if (n == 0) return pomai::Status::Ok();
    index_->add(static_cast<faiss::idx_t>(n), vecs);
    id_map_.insert(id_map_.end(), ids, ids + n);
    return pomai::Status::Ok();
}

// ── Query Phase ───────────────────────────────────────────────────────────────
pomai::Status HnswIndex::Search(std::span<const float> query,
                                 uint32_t               topk,
                                 int                    ef_search,
                                 std::vector<VectorId>* out_ids,
                                 std::vector<float>*    out_dists) const
{
    if (query.size() != dim_)
        return pomai::Status::InvalidArgument("query dim mismatch");
    if (id_map_.empty())
        return pomai::Status::Ok();

    const uint32_t k = std::min<uint32_t>(topk, static_cast<uint32_t>(id_map_.size()));

    // Temporarily override efSearch if caller requests it
    const int saved_ef = index_->hnsw.efSearch;
    if (ef_search > 0) index_->hnsw.efSearch = ef_search;

    std::vector<faiss::idx_t> faiss_ids(k, -1);
    std::vector<float>        faiss_dists(k, 0.0f);
    index_->search(1, query.data(), static_cast<faiss::idx_t>(k),
                   faiss_dists.data(), faiss_ids.data());

    index_->hnsw.efSearch = saved_ef;

    out_ids->clear();
    out_dists->clear();
    for (uint32_t i = 0; i < k; ++i) {
        if (faiss_ids[i] < 0) break;
        out_ids->push_back(id_map_[static_cast<std::size_t>(faiss_ids[i])]);
        out_dists->push_back(faiss_dists[i]);
    }
    return pomai::Status::Ok();
}

// ── Persistence (Phase 4) ─────────────────────────────────────────────────────
// .hnsw sidecar file format:
//   [FAISS native binary (faiss::write_index)]
//   [uint64 n_ids]
//   [VectorId × n_ids]

pomai::Status HnswIndex::Save(const std::string& path) const
{
    try {
        faiss::write_index(index_.get(), path.c_str());
    } catch (const std::exception& ex) {
        return pomai::Status::IOError(std::string("HNSW write failed: ") + ex.what());
    }

    // Append id_map to the same file
    std::ofstream f(path, std::ios::binary | std::ios::app);
    if (!f) return pomai::Status::IOError("Cannot append id_map to " + path);
    const uint64_t n = static_cast<uint64_t>(id_map_.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(reinterpret_cast<const char*>(id_map_.data()),
            n * sizeof(VectorId));
    if (!f) return pomai::Status::IOError("id_map write failed: " + path);
    return pomai::Status::Ok();
}

pomai::Status HnswIndex::Load(const std::string& path,
                               std::unique_ptr<HnswIndex>* out)
{
    // Read FAISS section
    faiss::Index* raw = nullptr;
    try {
        raw = faiss::read_index(path.c_str());
    } catch (const std::exception& ex) {
        return pomai::Status::IOError(std::string("HNSW read failed: ") + ex.what());
    }
    auto* hnsw_idx = dynamic_cast<faiss::IndexHNSWFlat*>(raw);
    if (!hnsw_idx) {
        delete raw;
        return pomai::Status::Corruption("File is not an IndexHNSWFlat: " + path);
    }

    // faiss::write_index writes an exact byte count; we need the offset of
    // the id_map section. We use FAISS's reader to get the file size up to
    // the index, then read the id_map from the remainder.
    //
    // Simplest approach: faiss::write_index / read_index via C FILE interface
    // writes everything to the file from position 0; the id_map was appended.
    // Use a second pass from the end file.
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        delete raw;
        return pomai::Status::IOError("Cannot re-open for id_map: " + path);
    }

    // The id_map is at the tail: [uint64 n][VectorId × n]
    // Seek backward from end of file.
    f.seekg(0, std::ios::end);
    const auto file_size = f.tellg();
    if (file_size < static_cast<std::streamoff>(sizeof(uint64_t))) {
        delete raw;
        return pomai::Status::Corruption("File too small: " + path);
    }
    f.seekg(-static_cast<std::streamoff>(sizeof(uint64_t)), std::ios::end);
    uint64_t n_ids = 0;
    f.read(reinterpret_cast<char*>(&n_ids), sizeof(n_ids));
    const auto id_bytes = static_cast<std::streamoff>(n_ids * sizeof(VectorId));
    f.seekg(-(static_cast<std::streamoff>(sizeof(uint64_t)) + id_bytes),
            std::ios::end);
    std::vector<VectorId> id_map(n_ids);
    f.read(reinterpret_cast<char*>(id_map.data()), id_bytes);
    if (!f) {
        delete raw;
        return pomai::Status::IOError("id_map read failed: " + path);
    }

    HnswOptions opts;
    opts.M              = hnsw_idx->hnsw.nb_neighbors(0) / 2;
    opts.ef_construction= hnsw_idx->hnsw.efConstruction;
    opts.ef_search      = hnsw_idx->hnsw.efSearch;

    auto result       = std::make_unique<HnswIndex>(
        static_cast<uint32_t>(hnsw_idx->d), opts);
    result->index_.reset(hnsw_idx);
    result->id_map_ = std::move(id_map);
    *out = std::move(result);
    return pomai::Status::Ok();
}

} // namespace pomai::index
