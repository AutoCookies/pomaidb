#include "core/vector_engine/vector_engine.h"

#include <utility>

#include "compute/vulkan/vulkan_device_context.h"
#include "core/distance.h"
#include "core/memory/pin_manager.h"
#include "core/shard/runtime.h"
#include "core/snapshot_wrapper.h"
#include "pomai/env.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"

namespace pomai::core {

namespace {

constexpr std::size_t kArenaBlockBytes = 1u << 20;  // 1 MiB
constexpr std::size_t kWalSegmentBytes = 64u << 20; // 64 MiB

} // namespace

VectorEngine::VectorEngine(pomai::DBOptions opt,
                           pomai::MembraneKind kind,
                           pomai::MetricType metric,
                           uint32_t ttl_sec,
                           uint32_t retention_max_count,
                           uint64_t retention_max_bytes,
                           uint64_t sync_lsn)
    : opt_(std::move(opt)),
      kind_(kind),
      metric_(metric),
      ttl_sec_(ttl_sec),
      retention_max_count_(retention_max_count),
      retention_max_bytes_(retention_max_bytes),
      sync_lsn_(sync_lsn) {}

VectorEngine::~VectorEngine() = default;

Status VectorEngine::EnsureOpen() const {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (kind_ != pomai::MembraneKind::kVector) {
        return Status::InvalidArgument("vector_engine requires VECTOR membrane");
    }
    if (opt_.dim == 0) {
        return Status::InvalidArgument("vector_engine requires dim > 0");
    }
    return Status::Ok();
}

Status VectorEngine::ValidateVector(std::span<const float> vec) const {
    if (static_cast<std::uint32_t>(vec.size()) != opt_.dim) {
        return Status::InvalidArgument("vector_engine dim mismatch");
    }
    return Status::Ok();
}

Status VectorEngine::Open() {
    if (opened_) return Status::Ok();
    core::InitDistance();
    return OpenLocked();
}

Status VectorEngine::OpenLocked() {
    if (kind_ != pomai::MembraneKind::kVector) {
        return Status::InvalidArgument("vector_engine only supports VECTOR membranes");
    }
    if (opt_.dim == 0) {
        return Status::InvalidArgument("vector_engine requires dim > 0");
    }

    pomai::Env* env = opt_.env ? opt_.env : pomai::Env::Default();
    Status st_env = env->CreateDirIfMissing(opt_.path);
    if (!st_env.ok())
        return Status::IOError("vector_engine CreateDirIfMissing failed");

    auto wal = std::make_unique<storage::Wal>(
        env, opt_.path, /*log_id=*/0u, kWalSegmentBytes, opt_.fsync,
        opt_.enable_encryption_at_rest, opt_.encryption_key_hex);
    Status st = wal->Open();
    if (!st.ok()) return st;

    // quantize_inmem_ only makes sense when the downstream segment/index
    // is also quantized. Otherwise we would decode back to lossy floats
    // and break exact float expectations (tests compare with EXPECT_EQ).
    const bool quantize_inmem =
        opt_.enable_quantization && (opt_.index_params.quant_type != pomai::QuantizationType::kNone);

    auto mem = std::make_unique<table::MemTable>(
        opt_.dim, kArenaBlockBytes, nullptr, quantize_inmem);
    st = wal->ReplayInto(*mem);
    if (!st.ok()) return st;

    // Segments and manifest live under "<membrane_path>/data".
    auto data_dir_path = std::filesystem::path(opt_.path) / "data";
    std::error_code ec2;
    std::filesystem::create_directories(data_dir_path, ec2);
    std::string data_dir = data_dir_path.string();
    auto rt = std::make_unique<VectorRuntime>(
        /*runtime_id=*/0u,
        std::move(data_dir),
        opt_.dim,
        kind_,
        metric_,
        std::move(wal),
        std::move(mem),
        opt_.index_params,
        sync_lsn_,
        opt_.endurance_aware_maintenance,
        opt_.write_budget_bytes_per_hour,
        opt_.endurance_compaction_bias,
        quantize_inmem,
        opt_.write_coalesce_window_us,
        opt_.write_coalesce_batch_size,
        ttl_sec_,
        retention_max_count_,
        retention_max_bytes_);

    runtime_ = std::move(rt);
    st = runtime_->Start();
    if (!st.ok()) {
        runtime_.reset();
        return st;
    }

    if (opt_.vulkan_enable_memory_bridge) {
        auto vctx = std::make_unique<pomai::compute::vulkan::VulkanComputeContext>();
        pomai::compute::vulkan::BridgeOptions bopt;
        bopt.prefer_unified_memory = opt_.vulkan_prefer_unified_memory;
        bopt.staging_pool_mb = opt_.vulkan_staging_pool_mb;
        bopt.zero_copy_min_bytes = opt_.vulkan_zero_copy_min_bytes;
        Status vv = pomai::compute::vulkan::VulkanComputeContext::Create(bopt, vctx.get());
        if (vv.ok()) {
            vulkan_ctx_ = std::move(vctx);
        }
    }

    opened_ = true;
    return Status::Ok();
}

Status VectorEngine::Close() {
    if (!opened_) return Status::Ok();
    vulkan_ctx_.reset();
    runtime_.reset();
    opened_ = false;
    return Status::Ok();
}

// Ingestion -------------------------------------------------------------------

Status VectorEngine::Put(VectorId id, std::span<const float> vec) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    st = ValidateVector(vec);
    if (!st.ok()) return st;
    return runtime_->Put(id, vec);
}

Status VectorEngine::Put(VectorId id,
                         std::span<const float> vec,
                         const pomai::Metadata& meta) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    st = ValidateVector(vec);
    if (!st.ok()) return st;
    return runtime_->Put(id, vec, meta);
}

Status VectorEngine::PutBatch(const std::vector<VectorId>& ids,
                              const std::vector<std::span<const float>>& vectors) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (ids.size() != vectors.size()) {
        return Status::InvalidArgument("vector_engine ids/vectors size mismatch");
    }
    for (const auto& v : vectors) {
        st = ValidateVector(v);
        if (!st.ok()) return st;
    }
    if (ids.empty()) return Status::Ok();
    return runtime_->PutBatch(ids, vectors);
}

Status VectorEngine::PutBatch(const std::vector<VectorId>& ids,
                              const std::vector<std::vector<float>>& vectors) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (ids.size() != vectors.size()) {
        return Status::InvalidArgument("vector_engine ids/vectors size mismatch");
    }
    for (const auto& v : vectors) {
        st = ValidateVector(std::span<const float>(v));
        if (!st.ok()) return st;
    }
    if (ids.empty()) return Status::Ok();
    std::vector<std::span<const float>> spans;
    spans.reserve(ids.size());
    for (const auto& v : vectors)
        spans.push_back(std::span<const float>(v));
    return runtime_->PutBatch(ids, spans);
}

// Point lookups ---------------------------------------------------------------

Status VectorEngine::Get(VectorId id, std::vector<float>* out) {
    return Get(id, out, nullptr);
}

Status VectorEngine::Get(VectorId id,
                         std::vector<float>* out,
                         pomai::Metadata* out_meta) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (!out) return Status::InvalidArgument("vector_engine output is null");
    return runtime_->Get(id, out, out_meta);
}

Status VectorEngine::Exists(VectorId id, bool* exists) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (!exists) return Status::InvalidArgument("vector_engine exists output is null");
    return runtime_->Exists(id, exists);
}

Status VectorEngine::Delete(VectorId id) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    return runtime_->Delete(id);
}

// Maintenance -----------------------------------------------------------------

Status VectorEngine::Flush() {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    return runtime_->Flush();
}

Status VectorEngine::Freeze() {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    return runtime_->Freeze();
}

Status VectorEngine::Compact() {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    return runtime_->Compact();
}

Status VectorEngine::PushSync(SyncReceiver* receiver) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    return runtime_->PushSync(receiver);
}

uint64_t VectorEngine::GetLastSyncedLSN() const {
    return runtime_ ? runtime_->GetLastSyncedLSN() : sync_lsn_;
}

std::size_t VectorEngine::MemTableBytesUsed() const noexcept {
    return runtime_ ? runtime_->MemTableBytesUsed() : 0u;
}

// Snapshots & iteration -------------------------------------------------------

Status VectorEngine::GetSnapshot(std::shared_ptr<pomai::Snapshot>* out) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (!out) return Status::InvalidArgument("vector_engine output is null");
    auto internal = runtime_->GetSnapshot();
    *out = std::make_shared<SnapshotWrapper>(std::move(internal));
    return Status::Ok();
}

Status VectorEngine::NewIterator(std::unique_ptr<pomai::SnapshotIterator>* out) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (!out) return Status::InvalidArgument("vector_engine output is null");
    return runtime_->NewIterator(out);
}

Status VectorEngine::NewIterator(const std::shared_ptr<pomai::Snapshot>& snap,
                                 std::unique_ptr<pomai::SnapshotIterator>* out) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (!out) return Status::InvalidArgument("vector_engine output is null");
    if (!snap) return Status::InvalidArgument("vector_engine snapshot is null");

    auto wrapper = std::dynamic_pointer_cast<SnapshotWrapper>(snap);
    if (!wrapper) {
        return Status::InvalidArgument("vector_engine invalid snapshot type");
    }
    return runtime_->NewIterator(wrapper->GetInternal(), out);
}

// Search ----------------------------------------------------------------------

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            pomai::SearchResult* out) {
    return Search(query, topk, SearchOptions{}, out);
}

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            const SearchOptions& opts,
                            pomai::SearchResult* out) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (!out) return Status::InvalidArgument("vector_engine output is null");

    out->Clear();
    if (static_cast<std::uint32_t>(query.size()) != opt_.dim) {
        return Status::InvalidArgument("vector_engine dim mismatch");
    }
    if (topk == 0) return Status::Ok();

    std::vector<SearchHit> hits;
    st = runtime_->Search(query, topk, opts, &hits);
    if (!st.ok()) return st;

    out->hits = std::move(hits);
    out->routed_shards_count = 1;
    out->total_shards_count = 1;
    out->pruned_shards_count = (!opts.partition_device_id.empty() || !opts.partition_location_id.empty()) ? 0 : 0;
    out->routing_probe_centroids = 0;
    out->routed_buckets_count = runtime_->LastQueryCandidatesScanned();

    if (opts.zero_copy) {
        std::shared_ptr<pomai::Snapshot> active_snap;
        auto st_snap = GetSnapshot(&active_snap);
        if (!st_snap.ok()) return st_snap;
        auto snap_wrapper = std::dynamic_pointer_cast<SnapshotWrapper>(active_snap);
        if (snap_wrapper) {
            auto internal_snap = snap_wrapper->GetInternal();
            out->zero_copy_session_id = core::MemoryPinManager::Instance().Pin(active_snap);
            out->zero_copy_pointers.reserve(out->hits.size());

            for (const auto& hit : out->hits) {
                pomai::SemanticPointer ptr;
                if (runtime_->GetSemanticPointer(internal_snap, hit.id, &ptr).ok()) {
                    ptr.session_id = out->zero_copy_session_id;
                    out->zero_copy_pointers.push_back(ptr);
                } else {
                    out->zero_copy_pointers.push_back(ptr);
                }
            }
        }
    }

    return Status::Ok();
}

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            const SearchOptions& opts,
                            pomai::SearchHitSink& sink) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;

    if (static_cast<std::uint32_t>(query.size()) != opt_.dim) {
        return Status::InvalidArgument("vector_engine dim mismatch");
    }
    if (topk == 0) return Status::Ok();

    // Re-use core VectorRuntime internal search logic directly
    return runtime_->Search(query, topk, opts, sink);
}

Status VectorEngine::SearchBatch(std::span<const float> queries,
                                 uint32_t num_queries,
                                 uint32_t topk,
                                 std::vector<pomai::SearchResult>* out) {
    return SearchBatch(queries, num_queries, topk, SearchOptions{}, out);
}

Status VectorEngine::SearchBatch(std::span<const float> queries,
                                 uint32_t num_queries,
                                 uint32_t topk,
                                 const SearchOptions& opts,
                                 std::vector<pomai::SearchResult>* out) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (!out) return Status::InvalidArgument("vector_engine output is null");

    if (queries.size() != static_cast<std::size_t>(num_queries) * opt_.dim) {
        return Status::InvalidArgument("vector_engine queries dimension mismatch for batch search");
    }

    out->clear();
    out->resize(num_queries);

    if (num_queries == 0 || topk == 0) {
        return Status::Ok();
    }

    std::vector<uint32_t> indices(num_queries);
    for (uint32_t i = 0; i < num_queries; ++i) {
        indices[i] = i;
    }

    std::vector<std::vector<pomai::SearchHit>> per_query;
    st = runtime_->SearchBatchLocal(queries, indices, topk, opts, &per_query);
    if (!st.ok()) return st;

    for (uint32_t i = 0; i < num_queries; ++i) {
        (*out)[i].hits = std::move(per_query[i]);
        (*out)[i].routed_shards_count = 1;
        (*out)[i].total_shards_count = 1;
        (*out)[i].pruned_shards_count = (!opts.partition_device_id.empty() || !opts.partition_location_id.empty()) ? 0 : 0;
        (*out)[i].routing_probe_centroids = 0;
        (*out)[i].routed_buckets_count = runtime_->LastQueryCandidatesScanned();
    }

    if (opts.zero_copy) {
        std::shared_ptr<pomai::Snapshot> active_snap;
        auto st_snap = GetSnapshot(&active_snap);
        if (!st_snap.ok()) return st_snap;
        auto snap_wrapper = std::dynamic_pointer_cast<SnapshotWrapper>(active_snap);
        if (snap_wrapper) {
            auto internal_snap = snap_wrapper->GetInternal();
            auto session_id = core::MemoryPinManager::Instance().Pin(active_snap);

            for (uint32_t i = 0; i < num_queries; ++i) {
                (*out)[i].zero_copy_session_id = session_id;
                (*out)[i].zero_copy_pointers.reserve((*out)[i].hits.size());

                for (const auto& hit : (*out)[i].hits) {
                    pomai::SemanticPointer ptr;
                    if (runtime_->GetSemanticPointer(internal_snap, hit.id, &ptr).ok()) {
                        ptr.session_id = session_id;
                        (*out)[i].zero_copy_pointers.push_back(ptr);
                    } else {
                        (*out)[i].zero_copy_pointers.push_back(ptr);
                    }
                }
            }
        }
    }

    return Status::Ok();
}

Status VectorEngine::SearchLexical(const std::string& query,
                                   std::uint32_t topk,
                                   std::vector<LexicalHit>* out) {
    auto st = EnsureOpen();
    if (!st.ok()) return st;
    if (!out) return Status::InvalidArgument("vector_engine lexical output is null");
    return runtime_->SearchLexical(query, topk, out);
}

} // namespace pomai::core

