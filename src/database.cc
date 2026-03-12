// PomaiDB Embedded: single-instance storage engine implementation.
// One Arena, one WAL, one index — strictly sequential data flow.

#include "pomai/database.h"

#include <cstdlib>
#include <utility>

#include "core/distance.h"
#include "core/shard/runtime.h"
#include "core/snapshot_wrapper.h"
#include "pomai/env.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"
#include "util/logging.h"

namespace pomai {

namespace {

constexpr std::size_t kArenaBlockBytes = 1u << 20;  // 1 MiB
constexpr std::size_t kWalSegmentBytes = 64u << 20; // 64 MiB

constexpr std::size_t kDefaultMaxMemtableMbNormal = 256u;
constexpr std::size_t kDefaultMaxMemtableMbLowMem = 64u;
constexpr std::uint8_t kDefaultPressureThresholdPercent = 80u;

// Parse optional env int; 0 or unset = use default.
static std::uint32_t EnvU32(const char* name, std::uint32_t default_val) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return static_cast<std::uint32_t>(default_val);
    return static_cast<std::uint32_t>(std::atoi(v));
}

static bool EnvBool(const char* name, bool default_val) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return default_val;
    return (v[0] == '1' || v[0] == 't' || v[0] == 'T');
}

} // namespace

// Single-instance storage: one WAL, one MemTable, one VectorRuntime.
class StorageEngine {
public:
    StorageEngine() = default;
    ~StorageEngine() = default;

    StorageEngine(const StorageEngine&) = delete;
    StorageEngine& operator=(const StorageEngine&) = delete;

    Status Open(const EmbeddedOptions& options) {
        if (runtime_) return Status::InvalidArgument("already opened");

        pomai::Env* env = options.env ? options.env : pomai::Env::Default();
        Status st = env->CreateDirIfMissing(options.path);
        if (!st.ok()) return st;

        if (options.dim == 0)
            return Status::InvalidArgument("dim must be > 0");

        auto wal = std::make_unique<storage::Wal>(
            env, options.path, 0u, kWalSegmentBytes, options.fsync);
        st = wal->Open();
        if (!st.ok()) return st;

        auto mem = std::make_unique<table::MemTable>(
            options.dim, kArenaBlockBytes);
        st = wal->ReplayInto(*mem);
        if (!st.ok()) return st;

        std::string data_dir = options.path;
        auto rt = std::make_unique<core::VectorRuntime>(
            0u, std::move(data_dir), options.dim,
            MembraneKind::kVector, options.metric,
            std::move(wal), std::move(mem), options.index_params);
        runtime_ = std::move(rt);
        return runtime_->Start();
    }

    void Close() { runtime_.reset(); }

    Status Append(VectorId id, std::span<const float> vec) {
        return runtime_ ? runtime_->Put(id, vec)
                       : Status::InvalidArgument("not opened");
    }

    Status Append(VectorId id, std::span<const float> vec,
                  const Metadata& meta) {
        return runtime_ ? runtime_->Put(id, vec, meta)
                        : Status::InvalidArgument("not opened");
    }

    Status AppendBatch(const std::vector<VectorId>& ids,
                       const std::vector<std::span<const float>>& vectors) {
        return runtime_ ? runtime_->PutBatch(ids, vectors)
                        : Status::InvalidArgument("not opened");
    }

    Status Get(VectorId id, std::vector<float>* out,
               Metadata* out_meta) {
        return runtime_ ? runtime_->Get(id, out, out_meta)
                        : Status::InvalidArgument("not opened");
    }

    Status Exists(VectorId id, bool* exists) {
        return runtime_ ? runtime_->Exists(id, exists)
                       : Status::InvalidArgument("not opened");
    }

    Status Delete(VectorId id) {
        return runtime_ ? runtime_->Delete(id)
                        : Status::InvalidArgument("not opened");
    }

    Status Flush() {
        return runtime_ ? runtime_->Flush()
                        : Status::InvalidArgument("not opened");
    }

    Status Freeze() {
        return runtime_ ? runtime_->Freeze()
                        : Status::InvalidArgument("not opened");
    }

    std::size_t MemTableBytesUsed() const {
        return runtime_ ? runtime_->MemTableBytesUsed() : 0u;
    }

    Status GetSnapshot(std::shared_ptr<Snapshot>* out) {
        if (!runtime_) return Status::InvalidArgument("not opened");
        if (!out) return Status::InvalidArgument("output is null");
        auto internal = runtime_->GetSnapshot();
        *out = std::make_shared<core::SnapshotWrapper>(std::move(internal));
        return Status::Ok();
    }

    Status NewIterator(const std::shared_ptr<Snapshot>& snap,
                       std::unique_ptr<SnapshotIterator>* out) {
        if (!runtime_) return Status::InvalidArgument("not opened");
        if (!out) return Status::InvalidArgument("output is null");
        if (!snap) return Status::InvalidArgument("snapshot is null");
        auto* wrapper = dynamic_cast<core::SnapshotWrapper*>(snap.get());
        if (!wrapper) return Status::InvalidArgument("invalid snapshot type");
        return runtime_->NewIterator(wrapper->GetInternal(), out);
    }

    Status Search(std::span<const float> query, std::uint32_t topk,
                  const SearchOptions& opts,
                  SearchResult* out) {
        if (!runtime_) return Status::InvalidArgument("not opened");
        if (!out) return Status::InvalidArgument("output is null");
        out->Clear();
        out->hits.clear();
        std::vector<SearchHit> hits;
        auto st = runtime_->Search(query, topk, opts, &hits);
        if (!st.ok()) return st;
        out->hits = std::move(hits);
        out->routed_shards_count = 1;
        out->routed_buckets_count = 0;
        return Status::Ok();
    }

    Status SearchBatch(std::span<const float> queries, std::uint32_t num_queries,
                      std::uint32_t topk, const SearchOptions& opts,
                      std::vector<SearchResult>* out) {
        if (!runtime_) return Status::InvalidArgument("not opened");
        if (!out) return Status::InvalidArgument("output is null");
        out->clear();
        out->resize(num_queries);
        if (num_queries == 0 || topk == 0) return Status::Ok();
        std::vector<uint32_t> indices(num_queries);
        for (uint32_t i = 0; i < num_queries; ++i) indices[i] = i;
        std::vector<std::vector<SearchHit>> per_query;
        auto st = runtime_->SearchBatchLocal(queries, indices, topk, opts, &per_query);
        if (!st.ok()) return st;
        for (uint32_t i = 0; i < num_queries; ++i) {
            (*out)[i].hits = std::move(per_query[i]);
            (*out)[i].routed_shards_count = 1;
        }
        return Status::Ok();
    }

private:
    std::unique_ptr<core::VectorRuntime> runtime_;
};

// -----------------------------------------------------------------------------
// Database
// -----------------------------------------------------------------------------

Database::Database() = default;

Database::~Database() {
    (void)Close();
}

Status Database::Open(const EmbeddedOptions& options) {
    if (opened_) return Status::InvalidArgument("already opened");
    if (options.path.empty())
        return Status::InvalidArgument("path empty");
    if (options.dim == 0)
        return Status::InvalidArgument("dim must be > 0");

    core::InitDistance();
    storage_engine_ = std::make_unique<StorageEngine>();
    auto st = storage_engine_->Open(options);
    if (!st.ok()) {
        storage_engine_.reset();
        return st;
    }

    // Memtable backpressure: from options or env. 0 = disabled (no limit).
    std::size_t max_mb = options.max_memtable_mb != 0
        ? options.max_memtable_mb
        : EnvU32("POMAI_MAX_MEMTABLE_MB",
                 EnvBool("POMAI_BENCH_LOW_MEMORY", false) ? kDefaultMaxMemtableMbLowMem
                                                          : kDefaultMaxMemtableMbNormal);
    max_memtable_bytes_ = max_mb * 1024u * 1024u;
    if (options.memtable_flush_threshold_mb != 0) {
        pressure_threshold_bytes_ = options.memtable_flush_threshold_mb * 1024u * 1024u;
    } else {
        std::uint8_t pct = options.pressure_threshold_percent != 0
            ? options.pressure_threshold_percent
            : static_cast<std::uint8_t>(EnvU32("POMAI_MEMTABLE_PRESSURE_THRESHOLD", kDefaultPressureThresholdPercent));
        if (pct > 100u) pct = 100u;
        pressure_threshold_bytes_ = (max_memtable_bytes_ * pct) / 100u;
    }
    auto_freeze_on_pressure_ = options.auto_freeze_on_pressure || EnvBool("POMAI_AUTO_FREEZE_ON_PRESSURE", true);

    opened_ = true;
    return Status::Ok();
}

Status Database::Close() {
    if (!opened_) return Status::Ok();
    if (storage_engine_) storage_engine_->Close();
    storage_engine_.reset();
    opened_ = false;
    return Status::Ok();
}

Status Database::Flush() {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Flush();
}

Status Database::Freeze() {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Freeze();
}

Status Database::MaybeApplyBackpressure() {
    if (max_memtable_bytes_ == 0) return Status::Ok();
    const std::size_t used = storage_engine_->MemTableBytesUsed();
    if (used < pressure_threshold_bytes_) return Status::Ok();
    if (auto_freeze_on_pressure_) {
        const unsigned used_mb = static_cast<unsigned>(used / (1024u * 1024u));
        POMAI_LOG_WARN("Memtable pressure detected (Usage: {} MB). Triggering Auto-Freeze.", used_mb);
        return storage_engine_->Freeze();
    }
    return Status::ResourceExhausted(
        "Memtable pressure high - call Freeze() or TryFreezeIfPressured()");
}

Status Database::TryFreezeIfPressured() {
    if (!opened_) return Status::InvalidArgument("not opened");
    if (max_memtable_bytes_ == 0) return Status::Ok();
    if (storage_engine_->MemTableBytesUsed() < pressure_threshold_bytes_)
        return Status::Ok();
    return storage_engine_->Freeze();
}

std::size_t Database::GetMemTableBytesUsed() const {
    return storage_engine_ ? storage_engine_->MemTableBytesUsed() : 0u;
}

Status Database::GetSnapshot(std::shared_ptr<Snapshot>* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->GetSnapshot(out);
}

Status Database::NewIterator(const std::shared_ptr<Snapshot>& snap,
                             std::unique_ptr<SnapshotIterator>* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->NewIterator(snap, out);
}

Status Database::AddVector(VectorId id, std::span<const float> vec) {
    if (!opened_) return Status::InvalidArgument("not opened");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    return storage_engine_->Append(id, vec);
}

Status Database::AddVector(VectorId id, std::span<const float> vec,
                          const Metadata& meta) {
    if (!opened_) return Status::InvalidArgument("not opened");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    return storage_engine_->Append(id, vec, meta);
}

Status Database::AddVectorBatch(
    const std::vector<VectorId>& ids,
    const std::vector<std::span<const float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("not opened");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    return storage_engine_->AppendBatch(ids, vectors);
}

Status Database::PutBatch(const std::vector<VectorId>& ids,
                          const std::vector<std::vector<float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("not opened");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    if (ids.size() != vectors.size())
        return Status::InvalidArgument("PutBatch: ids and vectors size mismatch");
    if (ids.empty()) return Status::Ok();
    std::vector<std::span<const float>> spans;
    spans.reserve(ids.size());
    for (const auto& v : vectors)
        spans.push_back(std::span<const float>(v));
    return storage_engine_->AppendBatch(ids, spans);
}

Status Database::PutBatch(std::span<const VectorId> ids,
                          std::span<const float> vectors,
                          std::size_t dimension) {
    if (!opened_) return Status::InvalidArgument("not opened");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    if (dimension == 0) {
        return Status::InvalidArgument("PutBatch: dimension must be > 0");
    }
    if (ids.empty()) return Status::Ok();
    const std::size_t expected = static_cast<std::size_t>(ids.size()) * dimension;
    if (vectors.size() != expected) {
        return Status::InvalidArgument("PutBatch: ids * dimension mismatch");
    }

    // Zero-copy: build spans into the flattened buffer without copying vector data.
    std::vector<VectorId> owned_ids(ids.begin(), ids.end());
    std::vector<std::span<const float>> spans;
    spans.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const float* base = vectors.data() + i * dimension;
        spans.emplace_back(base, dimension);
    }
    return storage_engine_->AppendBatch(owned_ids, spans);
}

Status Database::Get(VectorId id, std::vector<float>* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Get(id, out, nullptr);
}

Status Database::Get(VectorId id, std::vector<float>* out,
                    Metadata* out_meta) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Get(id, out, out_meta);
}

Status Database::Exists(VectorId id, bool* exists) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Exists(id, exists);
}

Status Database::Delete(VectorId id) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Delete(id);
}

Status Database::Search(std::span<const float> query,
                        std::uint32_t topk,
                        SearchResult* out) {
    return Search(query, topk, SearchOptions{}, out);
}

Status Database::Search(std::span<const float> query,
                        std::uint32_t topk,
                        const SearchOptions& opts,
                        SearchResult* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->Search(query, topk, opts, out);
}

Status Database::SearchBatch(std::span<const float> queries,
                             std::uint32_t num_queries,
                             std::uint32_t topk,
                             const SearchOptions& opts,
                             std::vector<SearchResult>* out) {
    if (!opened_) return Status::InvalidArgument("not opened");
    return storage_engine_->SearchBatch(queries, num_queries, topk, opts, out);
}

} // namespace pomai
