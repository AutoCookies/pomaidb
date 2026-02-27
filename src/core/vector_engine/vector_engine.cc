#include "core/vector_engine/vector_engine.h"

#include <algorithm>
#include <filesystem>
#include <future>
#include <limits>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>

#include "core/routing/kmeans_lite.h"
#include "core/routing/routing_persist.h"
#include "core/shard/runtime.h"
#include "core/shard/shard.h"
#include "core/snapshot_wrapper.h"
#include "core/memory/pin_manager.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"
#include "util/logging.h"
#include "core/distance.h"

namespace pomai::core {
namespace {
constexpr std::size_t kMailboxCap = 4096;
constexpr std::size_t kArenaBlockBytes = 1u << 20;  // 1 MiB
constexpr std::size_t kWalSegmentBytes = 64u << 20; // 64 MiB
constexpr std::uint64_t kPersistEveryPuts = 50000;

struct WorseHit {
    bool operator()(const pomai::SearchHit& a, const pomai::SearchHit& b) const {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.id > b.id;
    }
};

bool IsBetterHit(const pomai::SearchHit& a, const pomai::SearchHit& b) {
    if (a.score != b.score) {
        return a.score > b.score;
    }
    return a.id < b.id;
}

static std::vector<pomai::SearchHit> MergeTopK(const std::vector<std::vector<pomai::SearchHit>>& per,
                                               std::uint32_t k) {
    std::vector<pomai::SearchHit> out;
    if (k == 0) {
        return out;
    }
    std::priority_queue<pomai::SearchHit, std::vector<pomai::SearchHit>, WorseHit> heap;
    for (const auto& hits : per) {
        for (const auto& hit : hits) {
            if (heap.size() < k) {
                heap.push(hit);
                continue;
            }
            if (IsBetterHit(hit, heap.top())) {
                heap.pop();
                heap.push(hit);
            }
        }
    }

    out.reserve(heap.size());
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return IsBetterHit(a, b); });
    return out;
}
} // namespace

VectorEngine::VectorEngine(pomai::DBOptions opt, pomai::MembraneKind kind, pomai::MetricType metric)
    : opt_(std::move(opt)), kind_(kind), metric_(metric) {}
VectorEngine::~VectorEngine() = default;

std::uint32_t VectorEngine::ShardOf(VectorId id, std::uint32_t shard_count) {
    return shard_count == 0 ? 0u : static_cast<std::uint32_t>(id % shard_count);
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
    if (opt_.dim == 0) return Status::InvalidArgument("vector_engine requires dim > 0");
    if (opt_.shard_count == 0) return Status::InvalidArgument("vector_engine requires shard_count > 0");

    std::error_code ec;
    bool created_root_dir = false;
    if (!std::filesystem::exists(opt_.path, ec)) {
        if (!std::filesystem::create_directories(opt_.path, ec)) {
            return Status::IOError("vector_engine create_directories failed");
        }
        created_root_dir = true;
    } else if (ec) {
        return Status::IOError("vector_engine stat failed: " + ec.message());
    }

    if (!opt_.routing_enabled) {
        routing_mode_.store(routing::RoutingMode::kDisabled);
    } else {
        auto loaded = routing::LoadRoutingTable(opt_.path);
        if (loaded.has_value() && loaded->Valid() && loaded->dim == opt_.dim) {
            auto table = std::make_shared<routing::RoutingTable>(std::move(*loaded));
            routing_mutable_ = std::make_shared<routing::RoutingTable>(*table);
            routing_current_ = routing_mutable_;
            auto prev = routing::LoadRoutingPrevTable(opt_.path);
            if (prev.has_value() && prev->Valid() && prev->dim == opt_.dim) {
                routing_prev_ = std::make_shared<routing::RoutingTable>(std::move(*prev));
            }
            routing_mode_.store(routing::RoutingMode::kReady);
        } else {
            const std::uint32_t rk = std::max(1u, opt_.routing_k == 0 ? (2u * opt_.shard_count) : opt_.routing_k);
            warmup_target_ = rk * std::max(1u, opt_.routing_warmup_mult);
            warmup_reservoir_.reserve(static_cast<std::size_t>(warmup_target_) * opt_.dim);
            routing_mode_.store(routing::RoutingMode::kWarmup);
        }
    }

    shards_.clear();
    shards_.reserve(opt_.shard_count);

    size_t threads = opt_.search_threads;
    if (threads == 0) {
        size_t hw = std::thread::hardware_concurrency();
        if (hw == 0) {
            hw = 1;
        }
        size_t target = hw > 1 ? (hw - 1) : 1;
        // Do not cap by shard_count for batch searches. Queries can run in parallel.
        threads = target;
        if (threads == 0) {
            threads = 1;
        }
    }
    size_t segment_threads = std::max<size_t>(1, threads / 2);
    if (threads > 1) {
        search_pool_ = std::make_unique<util::ThreadPool>(threads);
    } else {
        search_pool_.reset();
    }
    if (segment_threads > 1) {
        segment_pool_ = std::make_unique<util::ThreadPool>(segment_threads);
    } else {
        segment_pool_.reset();
    }

    Status first_error = Status::Ok();
    for (std::uint32_t i = 0; i < opt_.shard_count; ++i) {
        auto wal = std::make_unique<storage::Wal>(opt_.path, i, kWalSegmentBytes, opt_.fsync);
        auto st = wal->Open();
        if (!st.ok()) {
            first_error = Status(st.code(), "vector_engine wal open failed: " + st.message());
            break;
        }
        auto mem = std::make_unique<table::MemTable>(opt_.dim, kArenaBlockBytes);
        st = wal->ReplayInto(*mem);
        if (!st.ok()) {
            first_error = Status(st.code(), "vector_engine wal replay failed: " + st.message());
            break;
        }
        auto shard_dir = (std::filesystem::path(opt_.path) / "shards" / std::to_string(i)).string();
        std::filesystem::create_directories(shard_dir, ec);

        auto rt = std::make_unique<ShardRuntime>(i, shard_dir, opt_.dim, kind_, metric_, std::move(wal), std::move(mem),
                                                 kMailboxCap, opt_.index_params, search_pool_.get(),
                                                 segment_pool_.get());
        auto shard = std::make_unique<Shard>(std::move(rt));

        st = shard->Start();
        if (!st.ok()) {
            first_error = Status(st.code(), "vector_engine shard start failed: " + st.message());
            break;
        }
        shards_.push_back(std::move(shard));
    }

    if (!first_error.ok()) {
        shards_.clear();
        search_pool_.reset();
        segment_pool_.reset();
        if (created_root_dir) {
            std::error_code ignore;
            std::filesystem::remove_all(opt_.path, ignore);
        }
        return first_error;
    }

    opened_ = true;
    return Status::Ok();
}

Status VectorEngine::Close() {
    if (!opened_) return Status::Ok();
    if (routing_mode_.load() == routing::RoutingMode::kReady && routing_mutable_) {
        shard_router_.Update([&]{
            (void)routing::SaveRoutingTableAtomic(opt_.path, *routing_mutable_, opt_.routing_keep_prev != 0);
        });
    }
    shards_.clear();
    search_pool_.reset();
    segment_pool_.reset();
    opened_ = false;
    return Status::Ok();
}

void VectorEngine::MaybeWarmupAndInitRouting(std::span<const float> vec) {
    if (routing_mode_.load() != routing::RoutingMode::kWarmup) return;
    if (warmup_count_ < warmup_target_) {
        warmup_reservoir_.insert(warmup_reservoir_.end(), vec.begin(), vec.end());
        ++warmup_count_;
    }
    if (warmup_count_ < warmup_target_) return;

    shard_router_.Update([&]{
        if (routing_mode_.load() == routing::RoutingMode::kReady) return;
        const std::uint32_t rk = std::max(1u, opt_.routing_k == 0 ? (2u * opt_.shard_count) : opt_.routing_k);
        auto built = routing::BuildInitialTable(std::span<const float>(warmup_reservoir_.data(), warmup_reservoir_.size()),
                                                warmup_count_, opt_.dim, rk, opt_.shard_count, 5, 12345);
        routing_prev_ = routing_current_;
        routing_mutable_ = std::make_shared<routing::RoutingTable>(built);
        routing_current_ = routing_mutable_;
        routing_mode_.store(routing::RoutingMode::kReady);
        (void)routing::SaveRoutingTableAtomic(opt_.path, built, opt_.routing_keep_prev != 0);
        util::Log(util::LogLevel::kInfo,
                  "[routing] mode=READY warmup_size=" + std::to_string(warmup_count_) +
                      " k=" + std::to_string(built.k));
    });
}

std::uint32_t VectorEngine::RouteShardForVector(VectorId id, std::span<const float> vec) {
    if (!opt_.routing_enabled || routing_mode_.load() != routing::RoutingMode::kReady || !routing_current_) {
        if (opt_.routing_enabled) MaybeWarmupAndInitRouting(vec);
        return ShardOf(id, opt_.shard_count);
    }

    auto table = routing_current_;
    const std::uint32_t sid = table->RouteVector(vec);

    {
        shard_router_.Update([&]{
            if (routing_mutable_) routing::OnlineUpdate(routing_mutable_.get(), vec);
        });
    }
    ++puts_since_persist_;
    MaybePersistRoutingAsync();
    return sid;
}

void VectorEngine::MaybePersistRoutingAsync() {
    if (puts_since_persist_ < kPersistEveryPuts) return;
    if (routing_persist_inflight_.load(std::memory_order_relaxed) || !routing_mutable_) return;
    puts_since_persist_ = 0;
    routing_persist_inflight_.store(true, std::memory_order_relaxed);
    auto snapshot = std::make_shared<routing::RoutingTable>(*routing_mutable_);
    if (search_pool_) {
        (void)search_pool_->Enqueue([this, snapshot]() {
            auto st = routing::SaveRoutingTableAtomic(opt_.path, *snapshot, opt_.routing_keep_prev != 0);
            if (!st.ok()) {
                util::Log(util::LogLevel::kWarn, "[routing] persist failed: " + st.message());
            }
            routing_persist_inflight_.store(false, std::memory_order_release);
            return Status::Ok();
        });
    } else {
        routing_persist_inflight_.store(false, std::memory_order_relaxed);
    }
}

Status VectorEngine::Put(VectorId id, std::span<const float> vec) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (kind_ != pomai::MembraneKind::kVector) {
        return Status::InvalidArgument("vector_engine requires VECTOR membrane");
    }
    if (static_cast<std::uint32_t>(vec.size()) != opt_.dim) {
        return Status::InvalidArgument("vector_engine dim mismatch");
    }
    const auto sid = RouteShardForVector(id, vec);
    return shards_[sid]->Put(id, vec);
}

Status VectorEngine::Put(VectorId id, std::span<const float> vec, const pomai::Metadata& meta) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (kind_ != pomai::MembraneKind::kVector) {
        return Status::InvalidArgument("vector_engine requires VECTOR membrane");
    }
    if (static_cast<std::uint32_t>(vec.size()) != opt_.dim) {
        return Status::InvalidArgument("vector_engine dim mismatch");
    }
    const auto sid = RouteShardForVector(id, vec);
    return shards_[sid]->Put(id, vec, meta);
}

Status VectorEngine::PutBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (kind_ != pomai::MembraneKind::kVector) {
        return Status::InvalidArgument("vector_engine requires VECTOR membrane");
    }
    if (ids.size() != vectors.size()) {
        return Status::InvalidArgument("vector_engine ids/vectors size mismatch");
    }
    if (ids.empty()) return Status::Ok();

    uint32_t shard_count = opt_.shard_count;
    std::vector<std::vector<VectorId>> shard_ids(shard_count);
    std::vector<std::vector<std::span<const float>>> shard_vecs(shard_count);
    size_t reserve_size = (ids.size() / shard_count) + 1;
    for (uint32_t i = 0; i < shard_count; ++i) {
        shard_ids[i].reserve(reserve_size);
        shard_vecs[i].reserve(reserve_size);
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        if (static_cast<uint32_t>(vectors[i].size()) != opt_.dim) {
            return Status::InvalidArgument("vector_engine dim mismatch");
        }
        const uint32_t s = RouteShardForVector(ids[i], vectors[i]);
        shard_ids[s].push_back(ids[i]);
        shard_vecs[s].push_back(vectors[i]);
    }

    for (uint32_t i = 0; i < shard_count; ++i) {
        if (shard_ids[i].empty()) continue;
        Status st = shards_[i]->PutBatch(shard_ids[i], shard_vecs[i]);
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status VectorEngine::Get(VectorId id, std::vector<float>* out) { return Get(id, out, nullptr); }

Status VectorEngine::Get(VectorId id, std::vector<float>* out, pomai::Metadata* out_meta) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (!out) return Status::InvalidArgument("vector_engine output is null");

    Status last = Status::NotFound("id not found");
    for (auto& s : shards_) {
        std::vector<float> tmp;
        pomai::Metadata meta;
        auto st = s->Get(id, &tmp, out_meta ? &meta : nullptr);
        if (st.ok()) {
            *out = std::move(tmp);
            if (out_meta) *out_meta = std::move(meta);
            return Status::Ok();
        }
        last = st;
    }
    return last;
}

Status VectorEngine::Exists(VectorId id, bool* exists) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (!exists) return Status::InvalidArgument("vector_engine exists output is null");
    *exists = false;
    for (auto& s : shards_) {
        bool e = false;
        auto st = s->Exists(id, &e);
        if (!st.ok()) return st;
        if (e) {
            *exists = true;
            return Status::Ok();
        }
    }
    return Status::Ok();
}

Status VectorEngine::Delete(VectorId id) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    Status first_error = Status::Ok();
    for (auto& s : shards_) {
        auto st = s->Delete(id);
        if (!st.ok() && first_error.ok()) first_error = st;
    }
    if (!first_error.ok()) return first_error;

    bool exists = false;
    auto exst = Exists(id, &exists);
    if (!exst.ok()) return exst;
    if (exists) return Status::Aborted("delete incomplete across shards");
    return Status::Ok();
}

Status VectorEngine::Flush() {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    for (auto& s : shards_) {
        auto st = s->Flush();
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status VectorEngine::Freeze() {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (shards_.empty()) return Status::Ok();

    // Phase 2: Parallel freeze — fan out to all shards concurrently.
    // Each shard's freeze is handled by its own actor (RunLoop); we just
    // enqueue the command from multiple threads simultaneously.
    if (search_pool_ && shards_.size() > 1) {
        std::vector<std::future<Status>> futs;
        futs.reserve(shards_.size());
        for (auto& s : shards_) {
            futs.push_back(search_pool_->Enqueue([&s]() { return s->Freeze(); }));
        }
        for (auto& f : futs) {
            Status st = f.get();
            if (!st.ok()) return st;
        }
        return Status::Ok();
    }

    // Fallback: sequential (no thread pool or single shard).
    for (auto& s : shards_) {
        Status st = s->Freeze();
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status VectorEngine::Compact() {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    for (auto& s : shards_) {
        Status st = s->Compact();
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status VectorEngine::NewIterator(std::unique_ptr<pomai::SnapshotIterator>* out) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (!out) return Status::InvalidArgument("vector_engine output is null");
    if (shards_.empty()) return Status::Internal("no shards available");
    return shards_[0]->NewIterator(out);
}

Status VectorEngine::GetSnapshot(std::shared_ptr<pomai::Snapshot>* out) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (!out) return Status::InvalidArgument("vector_engine output is null");
    if (shards_.empty()) return Status::Internal("no shards available");
    auto s = shards_[0]->GetSnapshot();
    *out = std::make_shared<SnapshotWrapper>(std::move(s));
    return Status::Ok();
}

Status VectorEngine::NewIterator(const std::shared_ptr<pomai::Snapshot>& snap,
                           std::unique_ptr<pomai::SnapshotIterator>* out) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (!out) return Status::InvalidArgument("vector_engine output is null");
    if (!snap) return Status::InvalidArgument("vector_engine snapshot is null");

    auto wrapper = std::dynamic_pointer_cast<SnapshotWrapper>(snap);
    if (!wrapper) return Status::InvalidArgument("vector_engine invalid snapshot type");
    if (shards_.empty()) return Status::Internal("no shards available");
    return shards_[0]->NewIterator(wrapper->GetInternal(), out);
}

Status VectorEngine::Search(std::span<const float> query, std::uint32_t topk, pomai::SearchResult* out) {
    return Search(query, topk, SearchOptions{}, out);
}

std::vector<std::uint32_t> VectorEngine::BuildProbeShards(std::span<const float> query,
                                                          const SearchOptions& opts) {
    if (opts.force_fanout || routing_mode_.load() != routing::RoutingMode::kReady || !routing_current_) {
        std::vector<std::uint32_t> all(opt_.shard_count);
        for (std::uint32_t i = 0; i < opt_.shard_count; ++i) all[i] = i;
        routed_probe_centroids_last_query_.store(0);
        routed_shards_last_query_count_.store(opt_.shard_count);
        return all;
    }

    auto table = routing_current_;
    std::uint32_t probe = opts.routing_probe_override ? opts.routing_probe_override
                                                      : (opt_.routing_probe ? opt_.routing_probe : 2u);
    probe = std::max(1u, std::min(probe, table->k));

    auto top2 = table->ClosestCentroids(query, std::min(2u, table->k));
    if (top2.size() == 2) {
        const float d1 = table->DistanceSq(query, top2[0]);
        const float d2 = table->DistanceSq(query, top2[1]);
        if ((d2 - d1) < 0.05f && table->k >= 3) probe = std::max(probe, 3u);
    }

    std::unordered_set<std::uint32_t> shard_set;
    auto cur = table->ClosestCentroids(query, probe);
    for (std::uint32_t c : cur) shard_set.insert(table->owner_shard[c]);

    auto prev = routing_prev_;
    if (prev && prev->Valid()) {
        auto prevc = prev->ClosestCentroids(query, std::min(probe, prev->k));
        for (std::uint32_t c : prevc) shard_set.insert(prev->owner_shard[c]);
    }

    std::vector<std::uint32_t> out;
    out.reserve(shard_set.size());
    for (auto sid : shard_set) out.push_back(sid);
    std::sort(out.begin(), out.end());

    routed_probe_centroids_last_query_.store(probe);
    routed_shards_last_query_count_.store(static_cast<std::uint32_t>(out.size()));
    return out;
}

Status VectorEngine::Search(std::span<const float> query,
                            std::uint32_t topk,
                            const SearchOptions& opts,
                            pomai::SearchResult* out) {
    return SearchInternal(query, topk, opts, out, true);
}

Status VectorEngine::SearchInternal(std::span<const float> query,
                                    std::uint32_t topk,
                                    const SearchOptions& opts,
                                    pomai::SearchResult* out,
                                    bool use_pool) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (!out) return Status::InvalidArgument("vector_engine output is null");

    out->Clear();
    if (static_cast<std::uint32_t>(query.size()) != opt_.dim) {
        return Status::InvalidArgument("vector_engine dim mismatch");
    }
    if (topk == 0) return Status::Ok();

    const auto probe_shards = BuildProbeShards(query, opts);
    std::vector<std::vector<pomai::SearchHit>> per(probe_shards.size());
    std::vector<std::future<pomai::Status>> futures;
    futures.reserve(probe_shards.size());

    if (search_pool_ && use_pool && probe_shards.size() > 1) {
        for (std::size_t i = 0; i < probe_shards.size(); ++i) {
            const std::uint32_t sid = probe_shards[i];
            futures.push_back(search_pool_->Enqueue([this, query, topk, opts, &per, sid, i] { return shards_[sid]->SearchLocal(query, topk, opts, &per[i]); }));
        }
    } else {
        for (std::size_t i = 0; i < probe_shards.size(); ++i) {
            const std::uint32_t sid = probe_shards[i];
            futures.push_back(std::async(std::launch::deferred, [this, query, topk, opts, &per, sid, i] { return shards_[sid]->SearchLocal(query, topk, opts, &per[i]); }));
        }
    }

    std::uint64_t candidates_scanned = 0;
    for (size_t i = 0; i < futures.size(); ++i) {
        Status st = futures[i].get();
        candidates_scanned += shards_[probe_shards[i]]->LastQueryCandidatesScanned();
        if (!st.ok()) {
            out->errors.push_back({probe_shards[i], st.message()});
        }
    }

    out->hits = MergeTopK(per, topk);
    out->routed_shards_count = routed_shards_last_query_count_.load();
    out->routing_probe_centroids = routed_probe_centroids_last_query_.load();
    out->routed_buckets_count = candidates_scanned;

    if (opts.zero_copy) {
        std::shared_ptr<pomai::Snapshot> active_snap;
        GetSnapshot(&active_snap);
        auto snap_wrapper = std::dynamic_pointer_cast<SnapshotWrapper>(active_snap);
        if (snap_wrapper) {
            auto internal_snap = snap_wrapper->GetInternal();
            out->zero_copy_session_id = core::MemoryPinManager::Instance().Pin(active_snap);
            out->zero_copy_pointers.reserve(out->hits.size());
            
            for (const auto& hit : out->hits) {
                pomai::SemanticPointer ptr;
                bool found = false;
                for (auto sid : probe_shards) {
                    if (shards_[sid]->GetSemanticPointer(internal_snap, hit.id, &ptr).ok()) {
                        ptr.session_id = out->zero_copy_session_id;
                        out->zero_copy_pointers.push_back(ptr);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    out->zero_copy_pointers.push_back(ptr);
                }
            }
        }
    }

    return Status::Ok();
}
Status VectorEngine::SearchBatch(std::span<const float> queries, uint32_t num_queries, 
                                 uint32_t topk, std::vector<pomai::SearchResult>* out) {
    return SearchBatch(queries, num_queries, topk, SearchOptions{}, out);
}

Status VectorEngine::SearchBatch(std::span<const float> queries, uint32_t num_queries, 
                                 uint32_t topk, const SearchOptions& opts, std::vector<pomai::SearchResult>* out) {
    if (!opened_) return Status::InvalidArgument("vector_engine not opened");
    if (!out) return Status::InvalidArgument("vector_engine output is null");
    
    // queries span must contain (num_queries * dim) floats.
    if (queries.size() != static_cast<size_t>(num_queries) * opt_.dim) {
        return Status::InvalidArgument("vector_engine queries dimension mismatch for batch search");
    }

    out->clear();
    out->resize(num_queries);

    if (num_queries == 0 || topk == 0) {
        return Status::Ok();
    }

    // 1. Group queries by shard
    std::vector<std::vector<uint32_t>> queries_by_shard(opt_.shard_count);
    for (uint32_t i = 0; i < num_queries; ++i) {
        std::span<const float> single_query(queries.data() + (i * opt_.dim), opt_.dim);
        auto probe_shards = BuildProbeShards(single_query, opts);
        for (auto sid : probe_shards) {
            queries_by_shard[sid].push_back(i);
        }
    }

    // 2. Dispatch tasks to shards
    std::vector<std::future<Status>> futures;
    // shard_results[shard_id][query_index] -> hits
    std::vector<std::vector<std::vector<pomai::SearchHit>>> shard_results(opt_.shard_count);

    for (uint32_t sid = 0; sid < opt_.shard_count; ++sid) {
        if (queries_by_shard[sid].empty()) continue;
        
        shard_results[sid].resize(num_queries);

        auto task = [this, sid, queries, &queries_by_shard, topk, opts, &shard_results]() {
            return shards_[sid]->SearchBatchLocal(queries, queries_by_shard[sid], topk, opts, &shard_results[sid]);
        };

        if (search_pool_) {
            futures.push_back(search_pool_->Enqueue(task));
        } else {
            // Use std::async as fallback if no pool
            futures.push_back(std::async(std::launch::async, task));
        }
    }
    
    // 3. Wait for all shard tasks
    for (auto& f : futures) {
        Status st = f.get();
        if (!st.ok()) return st;
    }

    // 4. Merge results for each query
    for (uint32_t i = 0; i < num_queries; ++i) {
        std::vector<std::vector<pomai::SearchHit>> per_query_hits;
        for (uint32_t sid = 0; sid < opt_.shard_count; ++sid) {
            if (!shard_results[sid].empty() && !shard_results[sid][i].empty()) {
                per_query_hits.push_back(std::move(shard_results[sid][i]));
            }
        }
        out->at(i).hits = MergeTopK(per_query_hits, topk);
        // Track candidates scanned if needed (optional)
    }

    // 5. Zero-copy support
    if (opts.zero_copy) {
        std::shared_ptr<pomai::Snapshot> active_snap;
        GetSnapshot(&active_snap);
        auto snap_wrapper = std::dynamic_pointer_cast<SnapshotWrapper>(active_snap);
        if (snap_wrapper) {
            auto internal_snap = snap_wrapper->GetInternal();
            auto session_id = core::MemoryPinManager::Instance().Pin(active_snap);
            
            for (uint32_t i = 0; i < num_queries; ++i) {
                out->at(i).zero_copy_session_id = session_id;
                out->at(i).zero_copy_pointers.reserve(out->at(i).hits.size());
                
                std::span<const float> single_query(queries.data() + (i * opt_.dim), opt_.dim);
                auto probe_shards = BuildProbeShards(single_query, opts);

                for (const auto& hit : out->at(i).hits) {
                    pomai::SemanticPointer ptr;
                    bool found = false;
                    for (auto sid : probe_shards) {
                        if (shards_[sid]->GetSemanticPointer(internal_snap, hit.id, &ptr).ok()) {
                            ptr.session_id = session_id;
                            out->at(i).zero_copy_pointers.push_back(ptr);
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        out->at(i).zero_copy_pointers.push_back(ptr);
                    }
                }
            }
        }
    }

    return Status::Ok();
}

} // namespace pomai::core
