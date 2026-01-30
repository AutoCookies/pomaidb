#include <pomai/core/shard.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <chrono>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>
#include <random>
#include <limits>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <pomai/util/fixed_topk.h>
#include <pomai/util/search_utils.h>
#include <pomai/util/cpu_kernels.h>
#include <pomai/core/spatial_router.h>
#include <pomai/util/pomai_assert.h>
#include <pomai/util/memory_manager.h>

namespace pomai
{
    namespace
    {
        constexpr std::size_t kTargetGrainSize = 1000;
        constexpr std::size_t kMaxProbe = 128;
        constexpr std::size_t kOversample = 128;

        std::size_t TargetCentroidCount(std::size_t n)
        {
            if (n == 0)
                return 0;
            return std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(n)));
        }

        std::vector<Vector> SampleSnapshotVectors(const Seed::Snapshot &snap, std::size_t max_samples)
        {
            std::vector<Vector> out;
            if (!snap || snap->ids.empty())
                return out;
            if (!snap->is_quantized.load(std::memory_order_acquire) || snap->qdata.empty())
                return out;
            const std::size_t n = snap->ids.size();
            const std::size_t dim = snap->dim;
            const std::size_t target = std::min(max_samples, n);
            out.reserve(target);
            std::vector<float> buf(dim);
            std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
            std::uniform_int_distribution<std::size_t> dist;
            for (std::size_t row = 0; row < n; ++row)
            {
                Seed::DequantizeRow(snap, row, buf.data());
                if (out.size() < target)
                {
                    Vector v;
                    v.data = buf;
                    out.push_back(std::move(v));
                }
                else
                {
                    std::uniform_int_distribution<std::size_t> d(0, row);
                    std::size_t pick = d(rng);
                    if (pick < target)
                    {
                        std::uniform_int_distribution<std::size_t> idx(0, target - 1);
                        std::size_t j = pick % target;
                        out[j].data = buf;
                    }
                }
            }
            return out;
        }

    }

    Shard::Shard(std::string name,
                 std::size_t dim,
                 std::size_t queue_cap,
                 std::string wal_dir,
                 CompactionConfig compaction,
                 LogFn info,
                 LogFn error)
        : name_(std::move(name)),
          wal_dir_(std::move(wal_dir)),
          wal_(name_, wal_dir_, dim),
          seed_(dim),
          ingest_q_(queue_cap),
          build_pool_(nullptr),
          log_info_(std::move(info)),
          log_error_(std::move(error)),
          compaction_(compaction)
    {
        auto live_snap = seed_.MakeSnapshot();
        auto next = std::make_shared<ShardState>();
        next->live_snap = live_snap;
        next->live_grains.reset();
        PublishState(std::move(next));
        ScheduleLiveGrainBuild(live_snap);
    }

    Shard::~Shard()
    {
        Stop();
    }

    void Shard::Start()
    {
        try
        {
            WalReplayStats stats = checkpoint_lsn_ ? wal_.ReplayToSeed(seed_, *checkpoint_lsn_) : wal_.ReplayToSeed(seed_);
            if (log_info_)
                log_info_("[" + name_ + "] WAL Replay: records=" + std::to_string(stats.records_applied));
        }
        catch (...)
        {
        }
        if (!recovered_)
        {
            auto live_snap = seed_.MakeSnapshot();
            auto next = std::make_shared<ShardState>();
            next->live_snap = live_snap;
            next->live_grains.reset();
            PublishState(std::move(next));
            ScheduleLiveGrainBuild(live_snap);
            if (seed_.Count() > 0)
                MaybeFreezeSegment();
        }
        wal_.Start();
        owner_ = std::thread(&Shard::RunLoop, this);
        compactor_running_.store(true, std::memory_order_release);
        compactor_ = std::thread(&Shard::RunCompactionLoop, this);
    }

    void Shard::Stop()
    {
        ingest_q_.Close();
        if (owner_.joinable())
            owner_.join();
        compactor_running_.store(false, std::memory_order_release);
        if (compactor_.joinable())
            compactor_.join();
        wal_.Stop();
    }

    std::size_t Shard::ApproxCountUnsafe() const
    {
        return seed_.Count();
    }

    std::shared_ptr<const ShardState> Shard::SnapshotState() const
    {
        return state_.load(std::memory_order_acquire);
    }

    std::size_t Shard::CompactionBacklog() const
    {
        return compaction_backlog_.load(std::memory_order_relaxed);
    }

    std::uint64_t Shard::LastCompactionDurationMs() const
    {
        return last_compaction_ms_.load(std::memory_order_relaxed);
    }

    Lsn Shard::DurableLsn() const
    {
        return wal_.DurableLsn();
    }

    Lsn Shard::WrittenLsn() const
    {
        return wal_.WrittenLsn();
    }

    std::future<Lsn> Shard::EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        UpsertTask t;
        t.batch = std::move(batch);
        t.wait_durable = wait_durable;
        auto fut = t.done.get_future();
        if (!ingest_q_.Push(std::move(t)))
        {
            std::promise<Lsn> p;
            p.set_exception(std::make_exception_ptr(std::runtime_error("closed")));
            return p.get_future();
        }
        return fut;
    }

    std::future<bool> Shard::RequestCheckpoint()
    {
        UpsertTask t;
        t.is_checkpoint = true;
        t.checkpoint_done.emplace();
        auto fut = t.checkpoint_done->get_future();
        if (!ingest_q_.Push(std::move(t)))
        {
            std::promise<bool> p;
            p.set_exception(std::make_exception_ptr(std::runtime_error("closed")));
            return p.get_future();
        }
        return fut;
    }

    std::future<ShardCheckpointState> Shard::RequestCheckpointState()
    {
        UpsertTask t;
        t.is_checkpoint_state = true;
        t.checkpoint_state_done.emplace();
        auto fut = t.checkpoint_state_done->get_future();
        if (!ingest_q_.Push(std::move(t)))
        {
            std::promise<ShardCheckpointState> p;
            p.set_exception(std::make_exception_ptr(std::runtime_error("closed")));
            return p.get_future();
        }
        return fut;
    }

    void Shard::RequestEmergencyFreeze()
    {
        bool expected = false;
        if (!emergency_freeze_pending_.compare_exchange_strong(expected, true))
            return;
        UpsertTask t;
        t.is_emergency_freeze = true;
        if (!ingest_q_.Push(std::move(t)))
            emergency_freeze_pending_.store(false);
    }

    void Shard::MergeTopK(SearchResponse &out, const SearchResponse &in, std::size_t k)
    {
        if (k == 0)
        {
            out.items.clear();
            return;
        }
        if (in.items.empty())
            return;
        FixedTopK topk(k);
        for (const auto &it : out.items)
            topk.Push(it.score, it.id);
        for (const auto &it : in.items)
            topk.Push(it.score, it.id);
        topk.FillSorted(out.items);
    }

    std::shared_ptr<GrainIndex> Shard::BuildGrainIndex(const Seed::Snapshot &snap) const
    {
        if (!snap || snap->ids.empty())
            return nullptr;
        if (!snap->is_quantized.load(std::memory_order_acquire) || snap->qdata.empty())
        {
            if (log_info_)
                log_info_("[" + name_ + "] BuildGrainIndex: snapshot not quantized, skipping index build");
            return nullptr;
        }
        const std::size_t n = snap->ids.size();
        const std::size_t dim = snap->dim;
        POMAI_ASSERT(snap->qmins.size() == dim && snap->qscales.size() == dim, "BuildGrainIndex quantization dim mismatch");
        POMAI_ASSERT(snap->qdata.size() == n * dim, "BuildGrainIndex qdata size mismatch");
        const std::size_t k = TargetCentroidCount(n);
        if (k == 0)
            return nullptr;
        std::size_t sample_cap = std::min<std::size_t>(n, 20000);
        auto &mm = MemoryManager::Instance();
        const std::size_t total = mm.TotalUsage();
        const std::size_t hard = mm.HardWatermarkBytes();
        const std::size_t avail = (hard > total) ? (hard - total) : 0;
        const std::size_t max_samples_by_mem = (dim > 0) ? (avail / (dim * sizeof(float))) : 0;
        if (max_samples_by_mem == 0)
            return nullptr;
        sample_cap = std::min(sample_cap, max_samples_by_mem);
        std::vector<Vector> sample = SampleSnapshotVectors(snap, sample_cap);
        if (sample.empty())
        {
            if (log_info_)
                log_info_("[" + name_ + "] BuildGrainIndex: no valid samples, skipping index build");
            return nullptr;
        }
        std::vector<Vector> centroids;
        try
        {
            centroids = SpatialRouter::BuildKMeans(sample, std::min(k, sample.size()), 8);
        }
        catch (...)
        {
            return nullptr;
        }
        if (centroids.empty())
            return nullptr;
        std::vector<std::uint32_t> assignments(n);
        std::vector<std::uint32_t> counts(centroids.size(), 0);
        std::vector<float> buf(dim);
        for (std::size_t row = 0; row < n; ++row)
        {
            Seed::DequantizeRow(snap, row, buf.data());
            float best_d = std::numeric_limits<float>::infinity();
            std::size_t best = 0;
            for (std::size_t c = 0; c < centroids.size(); ++c)
            {
                float d = kernels::L2Sqr(buf.data(), centroids[c].data.data(), dim);
                if (d < best_d)
                {
                    best_d = d;
                    best = c;
                }
            }
            assignments[row] = static_cast<std::uint32_t>(best);
            counts[best]++;
        }
        auto grains = std::make_shared<GrainIndex>();
        grains->dim = snap->dim;
        grains->centroids = std::move(centroids);
        grains->offsets.resize(grains->centroids.size() + 1, 0);
        for (std::size_t c = 0; c < grains->centroids.size(); ++c)
            grains->offsets[c + 1] = grains->offsets[c] + counts[c];
        grains->postings.resize(n);
        std::vector<std::size_t> cursor = grains->offsets;
        for (std::size_t row = 0; row < n; ++row)
            grains->postings[cursor[assignments[row]]++] = static_cast<std::uint32_t>(row);
        grains->namespace_offsets.resize(grains->centroids.size() + 1, 0);
        if (!snap->namespace_ids.empty())
        {
            std::vector<std::vector<std::uint32_t>> ns_buckets(grains->centroids.size());
            for (std::size_t row = 0; row < n; ++row)
            {
                std::size_t c = assignments[row];
                ns_buckets[c].push_back(snap->namespace_ids[row]);
            }
            for (std::size_t c = 0; c < ns_buckets.size(); ++c)
            {
                auto &bucket = ns_buckets[c];
                std::sort(bucket.begin(), bucket.end());
                bucket.erase(std::unique(bucket.begin(), bucket.end()), bucket.end());
                grains->namespace_offsets[c + 1] = grains->namespace_offsets[c] + bucket.size();
            }
            grains->namespace_ids.resize(grains->namespace_offsets.back());
            std::size_t cursor_ns = 0;
            for (const auto &bucket : ns_buckets)
            {
                std::copy(bucket.begin(), bucket.end(), grains->namespace_ids.begin() + cursor_ns);
                cursor_ns += bucket.size();
            }
        }
        return grains;
    }

    SearchResponse Shard::SearchGrains(const Seed::Snapshot &snap, const GrainIndex &grains, const SearchRequest &req, const pomai::ai::Budget &budget) const
    {
        SearchResponse resp;
        const std::size_t dim = snap->dim;
        POMAI_ASSERT(req.query.data.size() == dim, "SearchGrains query dim mismatch");
        POMAI_ASSERT(grains.dim == dim, "SearchGrains grains dim mismatch");
        POMAI_ASSERT(snap->qdata.size() == snap->ids.size() * dim, "SearchGrains qdata size mismatch");
        if (req.metric != Metric::L2)
            return resp;
        const bool has_filter = req.filter && !req.filter->empty();
        const std::size_t candidate_k = has_filter && req.filtered_candidate_k > 0 ? std::max<std::size_t>(req.filtered_candidate_k, req.topk)
                                                                                   : NormalizeCandidateK(req);
        const std::size_t topk = std::min<std::size_t>(req.topk, candidate_k);
        const std::size_t max_visits = has_filter ? std::max<std::size_t>(req.filter_max_visits, candidate_k) : 0;
        const std::uint64_t time_budget_us = has_filter ? req.filter_time_budget_us : 0;
        const auto start_time = std::chrono::steady_clock::now();
        std::size_t probe = budget.bucket_budget > 0 ? budget.bucket_budget : std::min<std::size_t>(16, grains.centroids.size());
        probe = std::min({probe, grains.centroids.size(), (std::size_t)128});
        FixedTopK centroid_topk(probe);
        for (std::size_t c = 0; c < grains.centroids.size(); ++c)
        {
            float d = kernels::L2Sqr(req.query.data.data(), grains.centroids[c].data.data(), dim);
            centroid_topk.Push(-d, static_cast<Id>(c));
        }
        std::vector<std::uint8_t> qquant(dim);
        for (std::size_t d = 0; d < dim; ++d)
        {
            float qv = (snap->qscales[d] > 0.0f) ? ((req.query.data[d] - snap->qmins[d]) / snap->qscales[d]) : 0.0f;
            qquant[d] = static_cast<std::uint8_t>(std::clamp<int>(static_cast<int>(std::nearbyint(qv)), 0, 255));
        }
        FixedTopK candidates(candidate_k);
        const auto *c_data = centroid_topk.Data();
        bool time_budget_hit = false;
        bool visit_budget_hit = false;
        bool stop = false;
        std::size_t visit_count = 0;
        for (std::size_t i = 0; i < centroid_topk.Size(); ++i)
        {
            std::size_t c = static_cast<std::size_t>(c_data[i].id);
            if (has_filter && req.filter->namespace_id && !grains.namespace_ids.empty())
            {
                std::uint32_t ns = *req.filter->namespace_id;
                std::size_t ns_begin = grains.namespace_offsets[c];
                std::size_t ns_end = grains.namespace_offsets[c + 1];
                if (ns_begin == ns_end)
                    continue;
                if (!std::binary_search(grains.namespace_ids.begin() + ns_begin,
                                        grains.namespace_ids.begin() + ns_end,
                                        ns))
                    continue;
            }
            for (std::size_t j = grains.offsets[c]; j < grains.offsets[c + 1]; ++j)
            {
                if (has_filter)
                {
                    if (candidates.Size() >= candidate_k)
                    {
                        stop = true;
                        break;
                    }
                    if (max_visits > 0 && visit_count >= max_visits)
                    {
                        visit_budget_hit = true;
                        stop = true;
                        break;
                    }
                    if (time_budget_us > 0 && (visit_count % 64 == 0))
                    {
                        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - start_time)
                                           .count();
                        if (elapsed >= static_cast<std::int64_t>(time_budget_us))
                        {
                            time_budget_hit = true;
                            stop = true;
                            break;
                        }
                    }
                    ++visit_count;
                }
                std::size_t row = grains.postings[j];
                if (has_filter && !snap->MatchFilter(row, *req.filter))
                    continue;
                float d = kernels::L2Sqr_SQ8_AVX2(snap->qdata.data() + row * dim, qquant.data(), dim);
                candidates.Push(-d, static_cast<Id>(row));
            }
            if (stop)
                break;
        }
        FixedTopK final_topk(topk);
        thread_local std::vector<float, AlignedAllocator<float, 64>> dequant;
        if (dequant.size() < dim)
            dequant.resize(dim);
        for (std::size_t i = 0; i < candidates.Size(); ++i)
        {
            std::size_t row = static_cast<std::size_t>(candidates.Data()[i].id);
            Seed::DequantizeRow(snap, row, dequant.data());
            final_topk.Push(-kernels::L2Sqr(dequant.data(), req.query.data.data(), dim), snap->ids[row]);
        }
        final_topk.FillSorted(resp.items);
        SortAndDedupeResults(resp.items, topk);
        resp.stats.filtered_candidates = candidates.Size();
        const bool filtered_partial = has_filter && candidates.Size() < candidate_k;
        const bool budget_exhausted = has_filter && (time_budget_hit || visit_budget_hit) && filtered_partial;
        if (budget_exhausted && req.search_mode == SearchMode::Quality)
            throw std::runtime_error("filtered grain search budget exhausted");
        if (filtered_partial)
            resp.stats.filtered_partial = true;
        resp.stats.filtered_time_budget_hit = time_budget_hit;
        resp.stats.filtered_visit_budget_hit = visit_budget_hit;
        resp.stats.filtered_budget_exhausted = budget_exhausted;
        resp.partial = resp.partial || resp.stats.filtered_partial;
        resp.stats.partial = resp.partial;
        return resp;
    }

    SearchResponse Shard::Search(const SearchRequest &req, const pomai::ai::Budget &budget) const
    {
        const auto start = std::chrono::steady_clock::now();
        auto snapshot = state_.load(std::memory_order_acquire);
        SearchResponse out;
        if (!snapshot)
            return out;
        SearchRequest normalized = req;
        normalized.candidate_k = NormalizeCandidateK(req);
        normalized.max_rerank_k = NormalizeMaxRerankK(req);
        normalized.graph_ef = NormalizeGraphEf(req, normalized.candidate_k);
        normalized.metric = req.metric;
        pomai::ai::Budget effective_budget = budget;
        if (normalized.graph_ef > 0)
            effective_budget.ops_budget = normalized.graph_ef;
        for (const auto &s : snapshot->segments)
        {
            SearchResponse r;
            if (s.grains && s.snap)
                r = SearchGrains(s.snap, *s.grains, normalized, effective_budget);
            else if (s.index && normalized.filter && s.snap)
                r = s.index->SearchFiltered(normalized, effective_budget, *normalized.filter, *s.snap);
            else if (s.index)
                r = s.index->Search(normalized, effective_budget);
            else if (s.snap)
                r = Seed::SearchSnapshot(s.snap, normalized);
            MergeTopK(out, r, req.topk);
            out.partial = out.partial || r.partial;
            out.stats.partial = out.stats.partial || r.stats.partial;
            out.stats.filtered_partial = out.stats.filtered_partial || r.stats.filtered_partial;
            out.stats.filtered_time_budget_hit = out.stats.filtered_time_budget_hit || r.stats.filtered_time_budget_hit;
            out.stats.filtered_visit_budget_hit = out.stats.filtered_visit_budget_hit || r.stats.filtered_visit_budget_hit;
            out.stats.filtered_budget_exhausted = out.stats.filtered_budget_exhausted || r.stats.filtered_budget_exhausted;
        }
        SearchResponse lr;
        if (snapshot->live_snap && snapshot->live_grains)
            lr = SearchGrains(snapshot->live_snap, *snapshot->live_grains, normalized, effective_budget);
        else if (snapshot->live_snap)
            lr = Seed::SearchSnapshot(snapshot->live_snap, normalized);
        MergeTopK(out, lr, req.topk);
        out.partial = out.partial || lr.partial;
        out.stats.partial = out.stats.partial || lr.stats.partial;
        out.stats.filtered_partial = out.stats.filtered_partial || lr.stats.filtered_partial;
        out.stats.filtered_time_budget_hit = out.stats.filtered_time_budget_hit || lr.stats.filtered_time_budget_hit;
        out.stats.filtered_visit_budget_hit = out.stats.filtered_visit_budget_hit || lr.stats.filtered_visit_budget_hit;
        out.stats.filtered_budget_exhausted = out.stats.filtered_budget_exhausted || lr.stats.filtered_budget_exhausted;
        SortAndDedupeResults(out.items, req.topk);
        if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() > 40)
            const_cast<Shard *>(this)->RequestEmergencyFreeze();
        return out;
    }

    void Shard::RunLoop()
    {
        while (auto opt = ingest_q_.Pop())
        {
            UpsertTask task = std::move(*opt);
            if (task.is_checkpoint)
            {
                if (task.checkpoint_done)
                    task.checkpoint_done->set_value(true);
                continue;
            }
            if (task.is_checkpoint_state)
            {
                try
                {
                    wal_.WaitDurable(wal_.WrittenLsn());
                    ShardCheckpointState state;
                    state.shard_id = 0;
                    state.durable_lsn = wal_.WrittenLsn();
                    auto snapshot = state_.load(std::memory_order_acquire);
                    if (snapshot)
                    {
                        state.segments.reserve(snapshot->segments.size());
                        for (const auto &seg : snapshot->segments)
                        {
                            if (!seg.snap)
                                continue;
                            std::vector<float> qmaxs;
                            qmaxs.resize(seg.snap->qmins.size());
                            for (std::size_t d = 0; d < seg.snap->qmins.size(); ++d)
                                qmaxs[d] = seg.snap->qmins[d] + seg.snap->qscales[d] * 255.0f;
                            state.segments.push_back(Seed::PersistedState{
                                static_cast<std::uint32_t>(seg.snap->dim),
                                seg.snap->ids,
                                seg.snap->qdata,
                                seg.snap->qmins,
                                std::move(qmaxs),
                                seg.snap->qscales,
                                seg.snap->namespace_ids,
                                seg.snap->tag_offsets,
                                seg.snap->tag_ids,
                                true,
                                0,
                                0});
                        }
                    }
                    state.live = seed_.ExportPersistedState();
                    if (task.checkpoint_state_done)
                        task.checkpoint_state_done->set_value(std::move(state));
                }
                catch (...)
                {
                    if (task.checkpoint_state_done)
                        task.checkpoint_state_done->set_exception(std::current_exception());
                }
                continue;
            }
            if (task.is_emergency_freeze)
            {
                MaybeFreezeSegment();
                emergency_freeze_pending_.store(false);
                continue;
            }
            try
            {
                Lsn lsn = wal_.AppendUpserts(task.batch, task.wait_durable);
                seed_.ApplyUpserts(task.batch);
                std::uint64_t clamped = seed_.ConsumeOutOfRangeCount();
                if (clamped > 0 && log_error_)
                    log_error_("[" + name_ + "] quantization clamp rows=" + std::to_string(clamped));
                since_freeze_ += task.batch.size();
                if (since_freeze_ >= kFreezeEveryVectors)
                {
                    since_freeze_ = 0;
                    MaybeFreezeSegment();
                }
                else if (since_live_publish_++ >= kPublishLiveEveryVectors)
                {
                    since_live_publish_ = 0;
                    auto s = seed_.MakeSnapshot();
                    {
                        std::lock_guard<std::mutex> lk(writer_mu_);
                        auto prev = state_.load(std::memory_order_acquire);
                        auto next = std::make_shared<ShardState>(prev ? *prev : ShardState{});
                        next->live_snap = s;
                        next->live_grains.reset();
                        PublishState(std::move(next));
                    }
                    ScheduleLiveGrainBuild(s);
                }
                if (task.wait_durable)
                    wal_.WaitDurable(lsn);
                task.done.set_value(lsn);
            }
            catch (...)
            {
                task.done.set_exception(std::current_exception());
            }
        }
        MaybeFreezeSegment();
    }

    void Shard::MaybeFreezeSegment()
    {
        auto snap = seed_.MakeSnapshot();
        if (!snap || snap->ids.empty())
            return;
        auto current = state_.load(std::memory_order_acquire);
            if (current && current->segments.size() >= kMaxSegments)
            {
                if (log_error_)
                    log_error_("[" + name_ + "] segment cap reached, skipping freeze to bound segment count");
                return;
            }
        std::size_t pos;
        {
            std::lock_guard<std::mutex> lk(writer_mu_);
            auto prev = state_.load(std::memory_order_acquire);
            auto next = std::make_shared<ShardState>(prev ? *prev : ShardState{});
            next->segments.push_back({snap, nullptr, nullptr, 0, segment_epoch_.fetch_add(1, std::memory_order_relaxed)});
            pos = next->segments.size() - 1;
            PublishState(std::move(next));
        }
        if (build_pool_)
        {
            IndexBuildPool::Job job;
            job.segment_pos = pos;
            job.snap = snap;
            job.M = 48;
            job.ef_construction = 200;
            job.attach = [this](std::size_t p, Seed::Snapshot s, std::shared_ptr<pomai::core::OrbitIndex> i)
            {
                auto g = this->BuildGrainIndex(s);
                this->AttachIndex(p, std::move(s), std::move(i), std::move(g));
            };
            build_pool_->Enqueue(std::move(job));
        }
        Seed next_seed(seed_.Dim());
        next_seed.InheritBounds(seed_);
        seed_ = std::move(next_seed);
        auto live_snap = seed_.MakeSnapshot();
        {
            std::lock_guard<std::mutex> lk(writer_mu_);
            auto prev = state_.load(std::memory_order_acquire);
            auto next = std::make_shared<ShardState>(prev ? *prev : ShardState{});
            next->live_snap = live_snap;
            next->live_grains.reset();
            PublishState(std::move(next));
        }
        ScheduleLiveGrainBuild(live_snap);
    }

    void Shard::ScheduleLiveGrainBuild(const Seed::Snapshot &snap)
    {
        if (!snap)
            return;
        if (!build_pool_)
        {
            if (log_info_)
                log_info_("[" + name_ + "] ScheduleLiveGrainBuild: no build pool, skipping live grains build");
            return;
        }
        bool queued = build_pool_->EnqueueTask([this, snap]()
                                               {
                                                   auto grains = BuildGrainIndex(snap);
                                                   AttachLiveGrains(snap, std::move(grains));
                                               });
        if (!queued && log_info_)
            log_info_("[" + name_ + "] ScheduleLiveGrainBuild: queue full, skipping live grains build");
    }

    void Shard::AttachIndex(std::size_t pos, Seed::Snapshot snap, std::shared_ptr<pomai::core::OrbitIndex> idx, std::shared_ptr<GrainIndex> grains)
    {
        std::lock_guard<std::mutex> lk(writer_mu_);
        auto prev = state_.load(std::memory_order_acquire);
        if (!prev)
            return;
        if (pos >= prev->segments.size() || prev->segments[pos].snap != snap)
            return;
        auto next = std::make_shared<ShardState>(*prev);
        next->segments[pos].index = std::move(idx);
        next->segments[pos].grains = std::move(grains);
        PublishState(std::move(next));
    }

    void Shard::AttachLiveGrains(Seed::Snapshot snap, std::shared_ptr<GrainIndex> grains)
    {
        std::lock_guard<std::mutex> lk(writer_mu_);
        auto prev = state_.load(std::memory_order_acquire);
        if (!prev)
            return;
        if (prev->live_snap != snap)
            return;
        auto next = std::make_shared<ShardState>(*prev);
        next->live_grains = std::move(grains);
        PublishState(std::move(next));
    }

    void Shard::PublishState(std::shared_ptr<const ShardState> next)
    {
        state_.store(std::move(next), std::memory_order_release);
    }

    std::vector<Vector> Shard::SampleVectors(std::size_t max_samples) const
    {
        std::vector<Seed::Snapshot> snaps;
        auto snapshot = state_.load(std::memory_order_acquire);
        if (!snapshot)
            return {};
        for (const auto &s : snapshot->segments)
            if (s.snap)
                snaps.push_back(s.snap);
        if (snapshot->live_snap)
            snaps.push_back(snapshot->live_snap);
        std::vector<Vector> res;
        res.reserve(std::min(max_samples, (std::size_t)2000));
        std::mt19937_64 rng(std::random_device{}());
        std::size_t seen = 0;
        for (const auto &s : snaps)
        {
            if (!s)
                continue;
            for (std::size_t i = 0; i < s->ids.size(); ++i)
            {
                seen++;
                if (res.size() < max_samples)
                {
                    Vector v;
                    v.data.resize(s->dim);
                    Seed::DequantizeRow(s, i, v.data.data());
                    res.push_back(std::move(v));
                }
                else
                {
                    std::uniform_int_distribution<std::size_t> d(0, seen - 1);
                    std::size_t j = d(rng);
                    if (j < max_samples)
                        Seed::DequantizeRow(s, i, res[j].data.data());
                }
            }
        }
        return res;
    }

    void Shard::LoadFromCheckpoint(const ShardCheckpointState &state, Lsn checkpoint_lsn)
    {
        seed_.LoadPersistedState(state.live);
        auto next = std::make_shared<ShardState>();
        next->segments.reserve(state.segments.size());
        for (const auto &seg : state.segments)
        {
            auto snap = Seed::SnapshotFromState(seg);
            next->segments.push_back({snap, nullptr, nullptr, 0, segment_epoch_.fetch_add(1, std::memory_order_relaxed)});
        }
        next->live_snap = seed_.MakeSnapshot();
        next->live_grains.reset();
        PublishState(std::move(next));
        checkpoint_lsn_ = checkpoint_lsn;
        recovered_ = true;
    }

    std::size_t Shard::ComputeCompactionBacklog(const std::shared_ptr<const ShardState> &state) const
    {
        if (!state)
            return 0;
        const std::size_t merge_count = std::max<std::size_t>({1, compaction_.compaction_trigger_threshold, compaction_.level_fanout});
        std::unordered_map<std::uint32_t, std::size_t> counts;
        for (const auto &seg : state->segments)
            counts[seg.level]++;
        std::size_t backlog = 0;
        for (const auto &kv : counts)
        {
            if (kv.second >= merge_count)
                backlog += kv.second;
        }
        return backlog;
    }

    void Shard::RunCompactionLoop()
    {
        while (compactor_running_.load(std::memory_order_acquire))
        {
            MaybeScheduleCompaction();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    void Shard::MaybeScheduleCompaction()
    {
        auto snapshot = state_.load(std::memory_order_acquire);
        compaction_backlog_.store(ComputeCompactionBacklog(snapshot), std::memory_order_relaxed);
        if (!snapshot)
            return;
        if (compaction_.max_concurrent_compactions == 0)
            return;
        const std::size_t merge_count = std::max<std::size_t>({1, compaction_.compaction_trigger_threshold, compaction_.level_fanout});
        if (active_compactions_.load(std::memory_order_relaxed) >= compaction_.max_concurrent_compactions)
            return;
        std::unordered_map<std::uint32_t, std::size_t> counts;
        for (const auto &seg : snapshot->segments)
            counts[seg.level]++;
        for (const auto &kv : counts)
        {
            if (kv.second >= merge_count)
            {
                if (active_compactions_.fetch_add(1, std::memory_order_acq_rel) >= compaction_.max_concurrent_compactions)
                {
                    active_compactions_.fetch_sub(1, std::memory_order_acq_rel);
                    return;
                }
                bool ok = CompactLevel(kv.first);
                active_compactions_.fetch_sub(1, std::memory_order_acq_rel);
                if (!ok)
                    return;
            }
        }
    }

    Seed Shard::MergeSnapshots(const std::vector<Seed::Snapshot> &snaps) const
    {
        Seed merged(seed_.Dim());
        std::vector<float> buf(seed_.Dim());
        std::vector<UpsertRequest> batch;
        batch.reserve(1024);
        for (const auto &snap : snaps)
        {
            if (!snap)
                continue;
            for (std::size_t row = 0; row < snap->ids.size(); ++row)
            {
                Seed::DequantizeRow(snap, row, buf.data());
                Metadata meta{};
                if (row < snap->namespace_ids.size())
                    meta.namespace_id = snap->namespace_ids[row];
                std::uint32_t start = (row < snap->tag_offsets.size()) ? snap->tag_offsets[row] : 0;
                std::uint32_t end = (row + 1 < snap->tag_offsets.size()) ? snap->tag_offsets[row + 1] : start;
                if (start <= end && end <= snap->tag_ids.size())
                    meta.tag_ids.assign(snap->tag_ids.begin() + start, snap->tag_ids.begin() + end);
                UpsertRequest req;
                req.id = snap->ids[row];
                req.vec.data = buf;
                req.metadata = std::move(meta);
                batch.push_back(std::move(req));
                if (batch.size() >= 1024)
                {
                    merged.ApplyUpserts(batch);
                    batch.clear();
                }
            }
        }
        if (!batch.empty())
            merged.ApplyUpserts(batch);
        return merged;
    }

    bool Shard::CompactLevel(std::uint32_t level)
    {
        auto start = std::chrono::steady_clock::now();
        auto snapshot = state_.load(std::memory_order_acquire);
        if (!snapshot)
            return false;
        const std::size_t merge_count = std::max<std::size_t>({1, compaction_.compaction_trigger_threshold, compaction_.level_fanout});
        std::vector<std::size_t> positions;
        std::vector<Seed::Snapshot> snaps;
        for (std::size_t i = 0; i < snapshot->segments.size(); ++i)
        {
            const auto &seg = snapshot->segments[i];
            if (seg.level == level && seg.snap)
            {
                positions.push_back(i);
                snaps.push_back(seg.snap);
                if (positions.size() >= merge_count)
                    break;
            }
        }
        if (positions.size() < merge_count)
            return false;
        Seed merged = MergeSnapshots(snaps);
        auto merged_snap = merged.MakeSnapshot();
        auto grains = BuildGrainIndex(merged_snap);
        std::shared_ptr<pomai::core::OrbitIndex> index;
        if (merged_snap && !merged_snap->ids.empty())
        {
            auto flat = Seed::DequantizeSnapshot(merged_snap);
            index = std::make_shared<pomai::core::OrbitIndex>(merged_snap->dim);
            index->Build(flat, merged_snap->ids);
        }
        {
            std::lock_guard<std::mutex> lk(writer_mu_);
            auto current = state_.load(std::memory_order_acquire);
            if (!current)
                return false;
            bool still_present = true;
            for (std::size_t idx : positions)
            {
                if (idx >= current->segments.size() || current->segments[idx].snap != snapshot->segments[idx].snap)
                {
                    still_present = false;
                    break;
                }
            }
            if (!still_present)
                return false;
            auto next = std::make_shared<ShardState>(*current);
            std::vector<IndexedSegment> compacted;
            compacted.reserve(next->segments.size() - positions.size() + 1);
            std::unordered_set<std::size_t> remove;
            remove.reserve(positions.size());
            for (std::size_t idx : positions)
                remove.insert(idx);
            for (std::size_t i = 0; i < next->segments.size(); ++i)
            {
                if (remove.count(i) == 0)
                    compacted.push_back(next->segments[i]);
            }
            IndexedSegment merged_seg;
            merged_seg.snap = merged_snap;
            merged_seg.grains = std::move(grains);
            merged_seg.index = std::move(index);
            merged_seg.level = level + 1;
            merged_seg.created_at = segment_epoch_.fetch_add(1, std::memory_order_relaxed);
            compacted.push_back(std::move(merged_seg));
            next->segments = std::move(compacted);
            PublishState(std::move(next));
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        last_compaction_ms_.store(static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
        return true;
    }
}
