#include "pomai/database.h"
#include "pomai/options.h"
#include "pomai/status.h"
#include "pomai/hooks.h"
#include "pomai/env.h"
#include "core/membrane/manager.h"
#include "core/concurrency/scheduler.h"
#include "core/graph/graph_membrane_impl.h"
#include "core/shard/runtime.h"
#include "table/memtable.h"
#include "storage/wal/wal.h"
#include "core/hooks/auto_edge_hook.h"
#include "core/graph/bitset_frontier.h"
#include "core/query/query_planner.h"
#include "core/storage/internal_engine.h"
#include "core/kernel/pods/vector_pod.h"
#include "core/kernel/pods/graph_pod.h"
#include "core/kernel/pods/query_pod.h"
#include "pomai/metadata.h"
#include "pomai/search.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>

namespace pomai {

Status StorageEngine::Open(const EmbeddedOptions& options) {
    auto env = options.env ? options.env : Env::Default();
    auto v_path = options.path + "/vectors";
    
    auto wal = std::make_unique<storage::Wal>(env, v_path, 0, 1024ULL * 1024 * 1024, options.fsync,
        options.enable_encryption_at_rest, options.encryption_key_hex);
    auto st = wal->Open();
    if (!st.ok()) return st;

    auto mem = std::make_unique<table::MemTable>(options.dim, 128ULL * 1024 * 1024);
    
    auto v_runtime = std::make_unique<core::VectorRuntime>(
        0, v_path, options.dim, 
        MembraneKind::kVector,
        options.metric, std::move(wal), std::move(mem), options.index_params);
        
    kernel_.RegisterPod(std::make_unique<core::VectorPod>(std::move(v_runtime)));

    auto g_path = options.path + "/graph";
    auto g_wal = std::make_unique<storage::Wal>(env, g_path, 1, 1024ULL * 1024 * 1024, options.fsync,
        options.enable_encryption_at_rest, options.encryption_key_hex);
    st = g_wal->Open();
    if (!st.ok()) return st;

    auto g_runtime = std::make_unique<core::GraphMembraneImpl>(std::move(g_wal));
    kernel_.RegisterPod(std::make_unique<core::GraphPod>(std::move(g_runtime)));

    auto planner = std::make_unique<core::QueryPlanner>(this);
    kernel_.RegisterPod(std::make_unique<core::QueryPod>(std::move(planner)));

    return Status::Ok();
}

Status StorageEngine::SearchMultiModal(std::string_view membrane, const MultiModalQuery& query, SearchResult* out) {
    const MultiModalQuery* q_ptr = &query;
    core::Message msg = core::Message::Create(core::PodId::kQuery, core::Op::kSearchMultiModal, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&q_ptr), sizeof(void*)));
    msg.membrane_id = membrane;
    msg.result_ptr = out;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

void StorageEngine::Close() {
    kernel_.Stop();
}

Status StorageEngine::Flush() {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kFlush);
    msg.result_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return st;
}

Status StorageEngine::Freeze() {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kFreeze);
    msg.result_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return st;
}

Status StorageEngine::Append(VectorId id, std::span<const float> vec, const Metadata& meta) {
    Status st = Status::Ok();
    struct P {
        VectorId id;
        const float* vec_data;
        size_t vec_size;
        const Metadata* meta;
    } p = {id, vec.data(), vec.size(), &meta};

    core::Message msg = core::Message::Create(core::PodId::kIndex, 0x0F /* kPutWithMeta */, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&p), sizeof(P)));
    msg.result_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();

    if (st.ok()) {
        for (auto& h : hooks_) h->OnPostPut(id, vec, meta);
    }
    return st;
}

Status StorageEngine::Append(VectorId id, std::span<const float> vec) {
    return Append(id, vec, Metadata());
}

Status StorageEngine::AppendBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors) {
    Status st = Status::Ok();
    struct P {
        const std::vector<VectorId>* ids;
        const std::vector<std::span<const float>>* vectors;
    } payload = {&ids, &vectors};

    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kPutBatch, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&payload), sizeof(payload)));
    msg.result_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();

    if (st.ok()) {
        for (size_t i = 0; i < ids.size(); ++i) {
            for (auto& h : hooks_) h->OnPostPut(ids[i], vectors[i], Metadata());
        }
    }
    return st;
}

Status StorageEngine::Get(VectorId id, std::vector<float>* out, Metadata* meta) {
    Status st = Status::Ok();
    struct P {
        VectorId id;
        std::vector<float>* out_vec;
        Metadata* out_meta;
    } p = {id, out, meta};

    core::Message msg = core::Message::Create(core::PodId::kIndex, 0x10 /* kGetWithMeta */, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&p), sizeof(P)));
    msg.result_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return st;
}

Status StorageEngine::Exists(VectorId id, bool* exists) {
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kExists, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&id), sizeof(id)));
    msg.result_ptr = exists;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

Status StorageEngine::Delete(VectorId id) {
    Status st = Status::Ok();
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kDelete, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&id), sizeof(id)));
    msg.result_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return st;
}

Status StorageEngine::Search(std::string_view membrane, std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) {
    struct P {
        uint32_t topk;
        const float* query_data;
        size_t query_size;
        const SearchOptions* opts;
    } p = {topk, query.data(), query.size(), &opts};

    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kSearch, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&p), sizeof(P)));
    msg.membrane_id = membrane;
    msg.result_ptr = out;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

Status StorageEngine::SearchLexical(std::string_view membrane, const std::string& query, uint32_t topk, std::vector<core::LexicalHit>* out) {
    // Op::kSearchLexical = 0x0C
    std::vector<uint8_t> payload(4 + query.size());
    std::memcpy(payload.data(), &topk, 4);
    std::memcpy(payload.data() + 4, query.data(), query.size());

    core::Message msg = core::Message::Create(core::PodId::kIndex, 0x0C /* kSearchLexical */, payload);
    msg.membrane_id = membrane;
    msg.result_ptr = out;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

Status StorageEngine::Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) {
    return Search("__default__", query, topk, opts, out);
}

Status StorageEngine::AddVertex(VertexId id, TagId tag, const Metadata& meta) {
    Status st = Status::Ok();
    struct { VertexId id; TagId tag; } payload = {id, tag};
    
    core::Message msg = core::Message::Create(core::PodId::kGraph, core::Op::kAddVertex, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&payload), sizeof(payload)));
    msg.result_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return st;
}

Status StorageEngine::AddEdge(VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta) {
    Status st = Status::Ok();
    struct { VertexId src; VertexId dst; EdgeType type; uint32_t rank; } payload = {src, dst, type, rank};

    core::Message msg = core::Message::Create(core::PodId::kGraph, core::Op::kAddEdge, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&payload), sizeof(payload)));
    msg.result_ptr = &st;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return st;
}

Status StorageEngine::GetNeighbors(std::string_view /*membrane*/, VertexId src, std::vector<pomai::Neighbor>* out) {
    return GetNeighbors(src, out);
}

Status StorageEngine::GetNeighbors(std::string_view /*membrane*/, VertexId src, EdgeType type, std::vector<pomai::Neighbor>* out) {
    return GetNeighbors(src, type, out);
}

std::optional<core::LinkedObject> StorageEngine::ResolveLinkedByVectorId(uint64_t /*vector_id*/) const {
    return std::nullopt;
}


// (Method removed to avoid duplication and signature mismatch)

Status StorageEngine::GetNeighbors(VertexId src, std::vector<pomai::Neighbor>* out) {
    core::Message msg = core::Message::Create(core::PodId::kGraph, core::Op::kGetNeighbors, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&src), sizeof(src)));
    msg.result_ptr = out;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

Status StorageEngine::GetNeighbors(VertexId src, EdgeType type, std::vector<pomai::Neighbor>* out) {
    struct { VertexId src; EdgeType type; } payload = {src, type};
    core::Message msg = core::Message::Create(core::PodId::kGraph, core::Op::kGetNeighborsWithType, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&payload), sizeof(payload)));
    msg.result_ptr = out;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

Status StorageEngine::GetSnapshot(std::shared_ptr<Snapshot>* out) {
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kGetSnapshot);
    msg.result_ptr = out;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

Status StorageEngine::NewIterator(const std::shared_ptr<Snapshot>& snap, std::unique_ptr<SnapshotIterator>* out) {
    const std::shared_ptr<Snapshot>* s_ptr = &snap;
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kNewIterator, 
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&s_ptr), sizeof(void*)));
    msg.result_ptr = out;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

Status StorageEngine::PushSync(core::SyncReceiver* receiver) {
    core::Message msg = core::Message::Create(core::PodId::kIndex, core::Op::kSync);
    msg.result_ptr = receiver;
    kernel_.Enqueue(std::move(msg));
    kernel_.ProcessAll();
    return Status::Ok();
}

std::size_t StorageEngine::GetMemTableBytesUsed() const {
    auto* pod = static_cast<core::VectorPod*>(const_cast<core::MicroKernel&>(kernel_).GetPod(core::PodId::kIndex));
    return pod ? pod->GetMemTableBytesUsed() : 0;
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
        AddPostPutHook(std::make_shared<core::AutoEdgeHook>(storage_engine_.get()));
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

Status Database::Get(VectorId id, std::vector<float>* out) { 
    if (!out) return Status::InvalidArgument("out cannot be null");
    return opened_ ? storage_engine_->Get(id, out, nullptr) : Status::InvalidArgument("closed"); 
}
Status Database::Get(VectorId id, std::vector<float>* out, Metadata* meta) { 
    if (!out) return Status::InvalidArgument("out cannot be null");
    return opened_ ? storage_engine_->Get(id, out, meta) : Status::InvalidArgument("closed"); 
}

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
    if (!out) return Status::InvalidArgument("out cannot be null");
    return Search(query, topk, SearchOptions(), out);
}

Status Database::Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) {
    if (!out) return Status::InvalidArgument("out cannot be null");
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
    if (!out) return Status::InvalidArgument("out cannot be null");
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
