#include "pomai/database.h"
#include "pomai/options.h"
#include "pomai/status.h"
#include "pomai/hooks.h"
#include "pomai/env.h"
#include "core/membrane/manager.h"
#include "core/concurrency/scheduler.h"
#include "core/graph/graph_membrane_impl.h"
#include "core/shard/runtime.h"
#include "src/table/memtable.h"
#include "storage/wal/wal.h"
#include "core/hooks/auto_edge_hook.h"
#include "core/graph/bitset_frontier.h"
#include "core/query/query_planner.h"
#include "core/storage/internal_engine.h"
#include <memory>
#include <vector>
#include <queue>
#include <unordered_set>
#include <iostream>

namespace pomai {

Status StorageEngine::Open(const EmbeddedOptions& options) {
    auto env = options.env ? options.env : Env::Default();
    auto v_path = options.path + "/vectors";
    
    auto wal = std::make_unique<storage::Wal>(env, v_path, 0, 1024ULL * 1024 * 1024, options.fsync);
    auto st = wal->Open();
    if (!st.ok()) return st;

    auto mem = std::make_unique<table::MemTable>(options.dim, 128ULL * 1024 * 1024);
    
    runtime_ = std::make_unique<core::VectorRuntime>(
        0, v_path, options.dim, 
        MembraneKind::kVector,
        options.metric, std::move(wal), std::move(mem), options.index_params);
        
    st = runtime_->Start();
    if (!st.ok()) return st;

    auto g_path = options.path + "/graph";
    auto g_wal = std::make_unique<storage::Wal>(env, g_path, 1, 1024ULL * 1024 * 1024, options.fsync);
    st = g_wal->Open();
    if (!st.ok()) return st;

    graph_runtime_ = std::make_unique<core::GraphMembraneImpl>(std::move(g_wal));
    planner_ = std::make_unique<core::QueryPlanner>(this);
    return Status::Ok();
}

Status StorageEngine::SearchMultiModal(std::string_view membrane, const MultiModalQuery& query, SearchResult* out) {
    return planner_ ? planner_->Execute(membrane, query, out) : Status::InvalidArgument("not opened");
}

void StorageEngine::Close() {
    runtime_.reset();
    graph_runtime_.reset();
}

Status StorageEngine::Flush() { return runtime_ ? runtime_->Flush() : Status::Ok(); }
Status StorageEngine::Freeze() { return runtime_ ? runtime_->Freeze() : Status::Ok(); }

Status StorageEngine::Append(VectorId id, std::span<const float> vec) {
    if (!runtime_) return Status::InvalidArgument("not opened");
    (void)runtime_->BeginBatch();
    auto st = runtime_->Put(id, vec);
    if (st.ok()) {
        for (auto& h : hooks_) h->OnPostPut(id, vec, Metadata());
    }
    (void)runtime_->EndBatch();
    return st;
}

Status StorageEngine::Append(VectorId id, std::span<const float> vec, const Metadata& meta) {
    if (!runtime_) return Status::InvalidArgument("not opened");
    (void)runtime_->BeginBatch();
    auto st = runtime_->Put(id, vec, meta);
    if (st.ok()) {
        for (auto& h : hooks_) h->OnPostPut(id, vec, meta);
    }
    (void)runtime_->EndBatch();
    return st;
}

Status StorageEngine::AppendBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors) {
    if (!runtime_) return Status::InvalidArgument("not opened");
    (void)runtime_->BeginBatch();
    auto st = runtime_->PutBatch(ids, vectors);
    if (st.ok()) {
        for (size_t i = 0; i < ids.size(); ++i) {
            for (auto& h : hooks_) h->OnPostPut(ids[i], vectors[i], Metadata());
        }
    }
    (void)runtime_->EndBatch();
    return st;
}

Status StorageEngine::Get(VectorId id, std::vector<float>* out, Metadata* meta) {
    return runtime_ ? runtime_->Get(id, out, meta) : Status::InvalidArgument("not opened");
}

Status StorageEngine::Exists(VectorId id, bool* exists) {
    return runtime_ ? runtime_->Exists(id, exists) : Status::InvalidArgument("not opened");
}

Status StorageEngine::Delete(VectorId id) {
    return runtime_ ? runtime_->Delete(id) : Status::InvalidArgument("not opened");
}

Status StorageEngine::Search(std::string_view /*membrane*/, std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) {
    if (!runtime_ || !out) return Status::InvalidArgument("invalid args");
    std::vector<SearchHit> hits;
    auto st = runtime_->Search(query, topk, opts, &hits);
    if (st.ok()) {
        out->hits = std::move(hits);
        out->routed_shards_count = 1;
    }
    return st;
}

Status StorageEngine::SearchLexical(std::string_view /*membrane*/, const std::string& query, uint32_t topk, std::vector<core::LexicalHit>* out) {
    if (!runtime_ || !out) return Status::InvalidArgument("invalid args");
    return runtime_->SearchLexical(query, topk, out);
}

Status StorageEngine::Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) {
    return Search("__default__", query, topk, opts, out);
}

Status StorageEngine::PushSync(core::SyncReceiver* receiver) {
    return runtime_ ? runtime_->PushSync(receiver) : Status::Ok();
}

Status StorageEngine::AddVertex(VertexId id, TagId tag, const Metadata& meta) {
    return graph_runtime_ ? graph_runtime_->AddVertex(id, tag, meta) : Status::InvalidArgument("no graph");
}

Status StorageEngine::AddEdge(VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta) {
    return graph_runtime_ ? graph_runtime_->AddEdge(src, dst, type, rank, meta) : Status::InvalidArgument("no graph");
}

Status StorageEngine::GetNeighbors(std::string_view /*membrane*/, VertexId src, std::vector<pomai::Neighbor>* out) {
    return GetNeighbors(src, out);
}

Status StorageEngine::GetNeighbors(std::string_view /*membrane*/, VertexId src, EdgeType type, std::vector<pomai::Neighbor>* out) {
    return GetNeighbors(src, type, out);
}

Status StorageEngine::GetNeighbors(VertexId src, std::vector<pomai::Neighbor>* out) {
    return graph_runtime_ ? graph_runtime_->GetNeighbors(src, out) : Status::InvalidArgument("no graph");
}

Status StorageEngine::GetNeighbors(VertexId src, EdgeType type, std::vector<pomai::Neighbor>* out) {
    return graph_runtime_ ? graph_runtime_->GetNeighbors(src, type, out) : Status::InvalidArgument("no graph");
}

Status StorageEngine::GetSnapshot(std::shared_ptr<Snapshot>* out) {
    if (!out) return Status::InvalidArgument("out pointer is null");
    if (!runtime_) return Status::InvalidArgument("not opened");
    auto v_snap = runtime_->GetSnapshot();
    if (!v_snap) return Status::Internal("failed to get snapshot");
    *out = std::move(v_snap);
    return Status::Ok();
}

Status StorageEngine::NewIterator(const std::shared_ptr<Snapshot>& snap, std::unique_ptr<SnapshotIterator>* out) {
    if (!runtime_) return Status::InvalidArgument("not opened");
    auto v_snap = std::static_pointer_cast<core::VectorSnapshot>(snap);
    return runtime_->NewIterator(v_snap, out);
}

std::size_t StorageEngine::GetMemTableBytesUsed() const {
    return runtime_ ? runtime_->GetStats().mem_used : 0;
}

void StorageEngine::AddPostPutHook(std::shared_ptr<PostPutHook> hook) {
    hooks_.push_back(std::move(hook));
}

// Tasks
class SyncTask : public core::DatabaseTask {
public:
    SyncTask(StorageEngine* engine, std::shared_ptr<core::SyncReceiver> receiver)
        : engine_(engine), receiver_(std::move(receiver)) {}
    Status Run() override { return engine_->PushSync(receiver_.get()); }
    std::string Name() const override { return "SyncTask"; }
private:
    StorageEngine* engine_;
    std::shared_ptr<core::SyncReceiver> receiver_;
};

class MaintenanceTask : public core::DatabaseTask {
public:
    explicit MaintenanceTask(Database* db) : db_(db) {}
    Status Run() override { return db_->MaybeApplyBackpressure(); }
    std::string Name() const override { return "Maintenance"; }
private:
    Database* db_;
};

struct Database::Impl {
    core::TaskScheduler scheduler;
    std::shared_ptr<core::SyncReceiver> sync_receiver;
};

Database::Database() : opened_(false), impl_(std::make_unique<Impl>()) {}
Database::~Database() { (void)Close(); }

Status Database::Open(const EmbeddedOptions& options) {
    if (opened_) return Status::InvalidArgument("already open");
    if (options.dim == 0) return Status::InvalidArgument("dimension must be greater than 0");
    if (options.path.empty()) return Status::InvalidArgument("path cannot be empty");
    
    std::uint32_t max_mb = options.max_memtable_mb;
    if (max_mb == 0) max_mb = 128u; // Default
    max_memtable_bytes_ = max_mb * 1024ULL * 1024ULL;
    
    std::uint8_t threshold_pct = options.pressure_threshold_percent;
    if (threshold_pct == 0) threshold_pct = 80u;
    pressure_threshold_bytes_ = (max_memtable_bytes_ * threshold_pct) / 100u;
    
    auto_freeze_on_pressure_ = options.auto_freeze_on_pressure;

    storage_engine_ = std::make_unique<StorageEngine>();
    auto st = storage_engine_->Open(options);
    if (!st.ok()) return st;

    opened_ = true;
    if (impl_->sync_receiver) {
        impl_->scheduler.RegisterPeriodic(std::make_unique<SyncTask>(storage_engine_.get(), impl_->sync_receiver), std::chrono::seconds(10));
    }
    impl_->scheduler.RegisterPeriodic(std::make_unique<MaintenanceTask>(this), std::chrono::seconds(5));
    
    if (options.enable_auto_edge) {
        AddPostPutHook(std::make_shared<core::AutoEdgeHook>(this));
    }
    return Status::Ok();
}

Status Database::Close() {
    if (!opened_) return Status::Ok();
    storage_engine_->Close();
    storage_engine_.reset();
    opened_ = false;
    return Status::Ok();
}

Status Database::Flush() { return opened_ ? storage_engine_->Flush() : Status::InvalidArgument("closed"); }
Status Database::Freeze() { return opened_ ? storage_engine_->Freeze() : Status::InvalidArgument("closed"); }

Status Database::TryFreezeIfPressured() {
    if (!opened_) return Status::InvalidArgument("closed");
    if (GetMemTableBytesUsed() > pressure_threshold_bytes_) {
        return Freeze();
    }
    return Status::Ok();
}

Status Database::MaybeApplyBackpressure() {
    if (!opened_) return Status::InvalidArgument("closed");
    if (GetMemTableBytesUsed() > pressure_threshold_bytes_) {
        if (auto_freeze_on_pressure_) {
            return Freeze();
        } else {
            return Status::ResourceExhausted("memtable pressure");
        }
    }
    return Status::Ok();
}

std::size_t Database::GetMemTableBytesUsed() const {
    return opened_ ? storage_engine_->GetMemTableBytesUsed() : 0;
}

Status Database::AddVector(VectorId id, std::span<const float> vec) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    st = storage_engine_->Append(id, vec);
    impl_->scheduler.Poll();
    return st;
}

Status Database::AddVector(VectorId id, std::span<const float> vec, const Metadata& meta) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    st = storage_engine_->Append(id, vec, meta);
    impl_->scheduler.Poll();
    return st;
}

Status Database::AddVectorBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    st = storage_engine_->AppendBatch(ids, vectors);
    impl_->scheduler.Poll();
    return st;
}

Status Database::PutBatch(const std::vector<VectorId>& ids, const std::vector<std::vector<float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    if (ids.size() != vectors.size()) return Status::InvalidArgument("mismatch");
    std::vector<std::span<const float>> spans;
    for (const auto& v : vectors) spans.push_back(v);
    st = storage_engine_->AppendBatch(ids, spans);
    impl_->scheduler.Poll();
    return st;
}

Status Database::PutBatch(std::span<const VectorId> ids, std::span<const float> vectors, std::size_t dimension) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = MaybeApplyBackpressure();
    if (!st.ok()) return st;
    if (dimension == 0 || ids.empty() || vectors.size() != ids.size() * dimension) return Status::InvalidArgument("invalid args");
    std::vector<VectorId> owned_ids(ids.begin(), ids.end());
    std::vector<std::span<const float>> spans;
    for (size_t i = 0; i < ids.size(); ++i) spans.emplace_back(vectors.data() + i * dimension, dimension);
    st = storage_engine_->AppendBatch(owned_ids, spans);
    impl_->scheduler.Poll();
    return st;
}

Status Database::Get(VectorId id, std::vector<float>* out) { return opened_ ? storage_engine_->Get(id, out, nullptr) : Status::InvalidArgument("closed"); }
Status Database::Get(VectorId id, std::vector<float>* out, Metadata* meta) { return opened_ ? storage_engine_->Get(id, out, meta) : Status::InvalidArgument("closed"); }

Status Database::Exists(VectorId id, bool* exists) {
    return opened_ ? storage_engine_->Exists(id, exists) : Status::InvalidArgument("closed");
}

Status Database::Delete(VectorId id) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = storage_engine_->Delete(id);
    impl_->scheduler.Poll();
    return st;
}

Status Database::Search(std::span<const float> query, uint32_t topk, SearchResult* out) {
    return Search(query, topk, SearchOptions(), out);
}

Status Database::Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) {
    return opened_ ? storage_engine_->Search("__default__", query, topk, opts, out) : Status::InvalidArgument("closed");
}

Status Database::SearchBatch(std::span<const float> queries, uint32_t num_queries, uint32_t topk, const SearchOptions& opts, std::vector<SearchResult>* out) {
    if (!opened_) return Status::InvalidArgument("closed");
    if (!out) return Status::InvalidArgument("out null");
    out->resize(num_queries);
    size_t dim = queries.size() / num_queries;
    for (uint32_t i = 0; i < num_queries; ++i) {
        auto st = storage_engine_->Search("__default__", queries.subspan(i * dim, dim), topk, opts, &(*out)[i]);
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status Database::SearchGraphRAG(std::span<const float> query, std::uint32_t topk,
                              const SearchOptions& opts, uint32_t k_hops,
                              std::vector<SearchResult>* out) {
    if (!opened_) return Status::InvalidArgument("closed");
    
    // Legacy implementation redirected to the new Planner logic
    MultiModalQuery mmq;
    mmq.vector.assign(query.begin(), query.end());
    mmq.top_k = topk;
    mmq.graph_hops = k_hops;
    
    SearchResult res;
    auto st = storage_engine_->SearchMultiModal("__default__", mmq, &res);
    if (st.ok() && out) {
        out->clear();
        out->push_back(std::move(res));
    }
    return st;
}

Status Database::SearchMultiModal(const MultiModalQuery& query, SearchResult* out) {
    return SearchMultiModal("__default__", query, out);
}

Status Database::SearchMultiModal(std::string_view membrane, const MultiModalQuery& query, SearchResult* out) {
    if (!opened_) return Status::InvalidArgument("closed");
    return storage_engine_->SearchMultiModal(membrane, query, out);
}

Status Database::AddVertex(VertexId id, TagId tag, const Metadata& meta) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = storage_engine_->AddVertex(id, tag, meta);
    impl_->scheduler.Poll();
    return st;
}

Status Database::AddEdge(VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta) {
    if (!opened_) return Status::InvalidArgument("closed");
    auto st = storage_engine_->AddEdge(src, dst, type, rank, meta);
    impl_->scheduler.Poll();
    return st;
}

Status Database::GetNeighbors(VertexId src, std::vector<Neighbor>* out) {
    return opened_ ? storage_engine_->GetNeighbors(src, out) : Status::InvalidArgument("closed");
}

Status Database::GetNeighbors(VertexId src, EdgeType type, std::vector<Neighbor>* out) {
    return opened_ ? storage_engine_->GetNeighbors(src, type, out) : Status::InvalidArgument("closed");
}

Status Database::GetSnapshot(std::shared_ptr<Snapshot>* out) {
    return opened_ ? storage_engine_->GetSnapshot(out) : Status::InvalidArgument("closed");
}

Status Database::NewIterator(const std::shared_ptr<Snapshot>& snap, std::unique_ptr<SnapshotIterator>* out) {
    return opened_ ? storage_engine_->NewIterator(snap, out) : Status::InvalidArgument("closed");
}

void Database::RegisterSyncReceiver(std::shared_ptr<core::SyncReceiver> receiver) {
    impl_->sync_receiver = std::move(receiver);
    if (opened_ && storage_engine_) {
        impl_->scheduler.RegisterPeriodic(std::make_unique<SyncTask>(storage_engine_.get(), impl_->sync_receiver), std::chrono::seconds(10));
    }
}

void Database::AddPostPutHook(std::shared_ptr<PostPutHook> hook) {
    if (opened_) storage_engine_->AddPostPutHook(std::move(hook));
}

} // namespace pomai
