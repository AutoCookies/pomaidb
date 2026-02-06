#include "core/engine/engine.h"

#include <algorithm>
#include <filesystem>
#include <future>
#include <iostream>
#include <limits>
#include <thread>
#include <unordered_set>
#include <utility>

#include "core/routing/kmeans_lite.h"
#include "core/routing/routing_persist.h"
#include "core/shard/runtime.h"
#include "core/shard/shard.h"
#include "core/snapshot_wrapper.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"

namespace pomai::core {
namespace {
constexpr std::size_t kMailboxCap = 4096;
constexpr std::size_t kArenaBlockBytes = 1u << 20;  // 1 MiB
constexpr std::size_t kWalSegmentBytes = 64u << 20; // 64 MiB
constexpr std::uint64_t kPersistEveryPuts = 50000;

static void MergeTopK(std::vector<pomai::SearchHit>* all, std::uint32_t k) {
    if (!all) return;
    if (all->size() <= k) {
        std::sort(all->begin(), all->end(), [](const auto& a, const auto& b) { return a.score > b.score; });
        return;
    }
    std::nth_element(all->begin(), all->begin() + static_cast<std::ptrdiff_t>(k), all->end(),
                     [](const auto& a, const auto& b) { return a.score > b.score; });
    all->resize(k);
    std::sort(all->begin(), all->end(), [](const auto& a, const auto& b) { return a.score > b.score; });
}
} // namespace

Engine::Engine(pomai::DBOptions opt) : opt_(std::move(opt)) {}
Engine::~Engine() = default;

std::uint32_t Engine::ShardOf(VectorId id, std::uint32_t shard_count) {
    return shard_count == 0 ? 0u : static_cast<std::uint32_t>(id % shard_count);
}

Status Engine::Open() {
    if (opened_) return Status::Ok();
    return OpenLocked();
}

Status Engine::OpenLocked() {
    if (opt_.dim == 0) return Status::InvalidArgument("dim must be > 0");
    if (opt_.shard_count == 0) return Status::InvalidArgument("shard_count must be > 0");

    std::error_code ec;
    bool created_root_dir = false;
    if (!std::filesystem::exists(opt_.path, ec)) {
        if (!std::filesystem::create_directories(opt_.path, ec)) return Status::IOError("create_directories failed");
        created_root_dir = true;
    } else if (ec) {
        return Status::IOError("stat failed: " + ec.message());
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

    size_t threads = std::thread::hardware_concurrency();
    if (threads < 4) threads = 4;
    search_pool_ = std::make_unique<util::ThreadPool>(threads);
    segment_pool_ = std::make_unique<util::ThreadPool>(threads);

    Status first_error = Status::Ok();
    for (std::uint32_t i = 0; i < opt_.shard_count; ++i) {
        auto wal = std::make_unique<storage::Wal>(opt_.path, i, kWalSegmentBytes, opt_.fsync);
        auto st = wal->Open();
        if (!st.ok()) {
            first_error = st;
            break;
        }
        auto mem = std::make_unique<table::MemTable>(opt_.dim, kArenaBlockBytes);
        st = wal->ReplayInto(*mem);
        if (!st.ok()) {
            first_error = st;
            break;
        }
        auto shard_dir = (std::filesystem::path(opt_.path) / "shards" / std::to_string(i)).string();
        std::filesystem::create_directories(shard_dir, ec);

        auto rt = std::make_unique<ShardRuntime>(i, shard_dir, opt_.dim, std::move(wal), std::move(mem),
                                                 kMailboxCap, opt_.index_params, search_pool_.get(),
                                                 segment_pool_.get());
        auto shard = std::make_unique<Shard>(std::move(rt));

        st = shard->Start();
        if (!st.ok()) {
            first_error = st;
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

Status Engine::Close() {
    if (!opened_) return Status::Ok();
    if (routing_mode_.load() == routing::RoutingMode::kReady && routing_mutable_) {
        std::lock_guard<std::mutex> lg(routing_mu_);
        (void)routing::SaveRoutingTableAtomic(opt_.path, *routing_mutable_, opt_.routing_keep_prev != 0);
    }
    search_pool_.reset();
    shards_.clear();
    opened_ = false;
    return Status::Ok();
}

void Engine::MaybeWarmupAndInitRouting(std::span<const float> vec) {
    if (routing_mode_.load() != routing::RoutingMode::kWarmup) return;
    if (warmup_count_ < warmup_target_) {
        warmup_reservoir_.insert(warmup_reservoir_.end(), vec.begin(), vec.end());
        ++warmup_count_;
    }
    if (warmup_count_ < warmup_target_) return;

    std::lock_guard<std::mutex> lg(routing_mu_);
    if (routing_mode_.load() == routing::RoutingMode::kReady) return;
    const std::uint32_t rk = std::max(1u, opt_.routing_k == 0 ? (2u * opt_.shard_count) : opt_.routing_k);
    auto built = routing::BuildInitialTable(std::span<const float>(warmup_reservoir_.data(), warmup_reservoir_.size()),
                                            warmup_count_, opt_.dim, rk, opt_.shard_count, 5, 12345);
    routing_prev_ = routing_current_;
    routing_mutable_ = std::make_shared<routing::RoutingTable>(built);
    routing_current_ = routing_mutable_;
    routing_mode_.store(routing::RoutingMode::kReady);
    (void)routing::SaveRoutingTableAtomic(opt_.path, built, opt_.routing_keep_prev != 0);
    std::cout << "[routing] mode=READY warmup_size=" << warmup_count_ << " k=" << built.k << "\n";
}

std::uint32_t Engine::RouteShardForVector(VectorId id, std::span<const float> vec) {
    if (!opt_.routing_enabled || routing_mode_.load() != routing::RoutingMode::kReady || !routing_current_) {
        if (opt_.routing_enabled) MaybeWarmupAndInitRouting(vec);
        return ShardOf(id, opt_.shard_count);
    }

    auto table = routing_current_;
    const std::uint32_t sid = table->RouteVector(vec);

    {
        std::lock_guard<std::mutex> lg(routing_mu_);
        if (routing_mutable_) routing::OnlineUpdate(routing_mutable_.get(), vec);
    }
    ++puts_since_persist_;
    MaybePersistRoutingAsync();
    return sid;
}

void Engine::MaybePersistRoutingAsync() {
    if (puts_since_persist_ < kPersistEveryPuts) return;
    std::lock_guard<std::mutex> lg(routing_mu_);
    if (routing_persist_inflight_ || !routing_mutable_) return;
    puts_since_persist_ = 0;
    routing_persist_inflight_ = true;
    auto snapshot = std::make_shared<routing::RoutingTable>(*routing_mutable_);
    if (search_pool_) {
        (void)search_pool_->Enqueue([this, snapshot]() {
            auto st = routing::SaveRoutingTableAtomic(opt_.path, *snapshot, opt_.routing_keep_prev != 0);
            if (!st.ok()) {
                std::cout << "[routing] persist failed: " << st.message() << "\n";
            }
            std::lock_guard<std::mutex> lk(routing_mu_);
            routing_persist_inflight_ = false;
            return Status::Ok();
        });
    } else {
        routing_persist_inflight_ = false;
    }
}

Status Engine::Put(VectorId id, std::span<const float> vec) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (static_cast<std::uint32_t>(vec.size()) != opt_.dim) return Status::InvalidArgument("dim mismatch");
    const auto sid = RouteShardForVector(id, vec);
    return shards_[sid]->Put(id, vec);
}

Status Engine::Put(VectorId id, std::span<const float> vec, const pomai::Metadata& meta) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (static_cast<std::uint32_t>(vec.size()) != opt_.dim) return Status::InvalidArgument("dim mismatch");
    const auto sid = RouteShardForVector(id, vec);
    return shards_[sid]->Put(id, vec, meta);
}

Status Engine::PutBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (ids.size() != vectors.size()) return Status::InvalidArgument("size mismatch");
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
        if (static_cast<uint32_t>(vectors[i].size()) != opt_.dim) return Status::InvalidArgument("dim mismatch");
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

Status Engine::Get(VectorId id, std::vector<float>* out) { return Get(id, out, nullptr); }

Status Engine::Get(VectorId id, std::vector<float>* out, pomai::Metadata* out_meta) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (!out) return Status::InvalidArgument("out=null");

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

Status Engine::Exists(VectorId id, bool* exists) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (!exists) return Status::InvalidArgument("exists=null");
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

Status Engine::Delete(VectorId id) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
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

Status Engine::Flush() {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    for (auto& s : shards_) {
        auto st = s->Flush();
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status Engine::Freeze() {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    for (auto& s : shards_) {
        Status st = s->Freeze();
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status Engine::Compact() {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    for (auto& s : shards_) {
        Status st = s->Compact();
        if (!st.ok()) return st;
    }
    return Status::Ok();
}

Status Engine::NewIterator(std::unique_ptr<pomai::SnapshotIterator>* out) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (!out) return Status::InvalidArgument("out is null");
    if (shards_.empty()) return Status::Internal("no shards available");
    return shards_[0]->NewIterator(out);
}

Status Engine::GetSnapshot(std::shared_ptr<pomai::Snapshot>* out) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (!out) return Status::InvalidArgument("out is null");
    if (shards_.empty()) return Status::Internal("no shards available");
    auto s = shards_[0]->GetSnapshot();
    *out = std::make_shared<SnapshotWrapper>(std::move(s));
    return Status::Ok();
}

Status Engine::NewIterator(const std::shared_ptr<pomai::Snapshot>& snap,
                           std::unique_ptr<pomai::SnapshotIterator>* out) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (!out) return Status::InvalidArgument("out is null");
    if (!snap) return Status::InvalidArgument("snap is null");

    auto wrapper = std::dynamic_pointer_cast<SnapshotWrapper>(snap);
    if (!wrapper) return Status::InvalidArgument("invalid snapshot type");
    if (shards_.empty()) return Status::Internal("no shards available");
    return shards_[0]->NewIterator(wrapper->GetInternal(), out);
}

Status Engine::Search(std::span<const float> query, std::uint32_t topk, pomai::SearchResult* out) {
    return Search(query, topk, SearchOptions{}, out);
}

std::vector<std::uint32_t> Engine::BuildProbeShards(std::span<const float> query, const SearchOptions& opts) {
    if (opts.force_fanout || routing_mode_.load() != routing::RoutingMode::kReady || !routing_current_) {
        std::vector<std::uint32_t> all(opt_.shard_count);
        for (std::uint32_t i = 0; i < opt_.shard_count; ++i) all[i] = i;
        routed_probe_centroids_last_query_.store(0);
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
    std::cout << "[routing] mode=READY routed_shards=" << out.size() << " probe=" << probe << "\n";
    return out;
}

Status Engine::Search(std::span<const float> query,
                      std::uint32_t topk,
                      const SearchOptions& opts,
                      pomai::SearchResult* out) {
    if (!opened_) return Status::InvalidArgument("engine not opened");
    if (!out) return Status::InvalidArgument("out=null");

    out->Clear();
    if (static_cast<std::uint32_t>(query.size()) != opt_.dim) return Status::InvalidArgument("dim mismatch");
    if (topk == 0) return Status::Ok();

    const auto probe_shards = BuildProbeShards(query, opts);
    std::vector<std::vector<pomai::SearchHit>> per(probe_shards.size());
    std::vector<std::future<pomai::Status>> futures;
    futures.reserve(probe_shards.size());

    for (std::size_t i = 0; i < probe_shards.size(); ++i) {
        const std::uint32_t sid = probe_shards[i];
        futures.push_back(search_pool_->Enqueue([&, sid, i] { return shards_[sid]->SearchLocal(query, topk, opts, &per[i]); }));
    }

    std::vector<pomai::SearchHit> merged;
    for (size_t i = 0; i < futures.size(); ++i) {
        Status st = futures[i].get();
        if (!st.ok()) {
            out->errors.push_back({probe_shards[i], st.message()});
        } else {
            merged.insert(merged.end(), per[i].begin(), per[i].end());
        }
    }

    MergeTopK(&merged, topk);
    out->hits = std::move(merged);
    return Status::Ok();
}

} // namespace pomai::core
