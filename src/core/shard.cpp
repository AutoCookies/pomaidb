#include "shard.h"
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
#include "fixed_topk.h"
#include "cpu_kernels.h"
#include "spatial_router.h"

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

        void CheckedWrite(int fd, const void *buf, size_t count)
        {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(buf);
            size_t rem = count;
            while (rem > 0)
            {
                ssize_t res = ::write(fd, p, rem);
                if (res < 0)
                {
                    if (errno == EINTR)
                        continue;
                    throw std::runtime_error("Disk write failed");
                }
                p += static_cast<size_t>(res);
                rem -= static_cast<size_t>(res);
            }
        }

        std::size_t PeakRssBytes()
        {
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) != 0)
                return 0;
            return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
        }
    }

    Shard::Shard(std::string name, std::size_t dim, std::size_t queue_cap, std::string wal_dir, LogFn info, LogFn error)
        : name_(std::move(name)), wal_dir_(std::move(wal_dir)), wal_(name_, wal_dir_, dim), seed_(dim), ingest_q_(queue_cap), build_pool_(nullptr), log_info_(std::move(info)), log_error_(std::move(error))
    {
        auto live_snap = seed_.MakeSnapshot();
        auto live_grains = BuildGrainIndex(live_snap);
        auto next = std::make_shared<ShardState>();
        next->live_snap = std::move(live_snap);
        next->live_grains = std::move(live_grains);
        PublishState(std::move(next));
    }

    Shard::~Shard()
    {
        Stop();
    }

    void Shard::Start()
    {
        try
        {
            WalReplayStats stats = wal_.ReplayToSeed(seed_);
            if (log_info_)
                log_info_("[" + name_ + "] WAL Replay: records=" + std::to_string(stats.records_applied));
        }
        catch (...)
        {
        }
        auto live_snap = seed_.MakeSnapshot();
        auto live_grains = BuildGrainIndex(live_snap);
        auto next = std::make_shared<ShardState>();
        next->live_snap = std::move(live_snap);
        next->live_grains = std::move(live_grains);
        PublishState(std::move(next));
        if (seed_.Count() > 0)
            MaybeFreezeSegment();
        wal_.Start();
        owner_ = std::thread(&Shard::RunLoop, this);
    }

    void Shard::Stop()
    {
        ingest_q_.Close();
        if (owner_.joinable())
            owner_.join();
        wal_.Stop();
    }

    std::size_t Shard::ApproxCountUnsafe() const
    {
        return seed_.Count();
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
        const std::size_t k = TargetCentroidCount(n);
        if (k == 0)
            return nullptr;
        const std::size_t sample_cap = std::min<std::size_t>(n, 20000);
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
        const std::size_t dim = snap->dim;
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
        return grains;
    }

    SearchResponse Shard::SearchGrains(const Seed::Snapshot &snap, const GrainIndex &grains, const SearchRequest &req, const pomai::ai::Budget &budget) const
    {
        SearchResponse resp;
        const std::size_t dim = snap->dim;
        const std::size_t topk = std::min<std::size_t>(req.topk, 128);
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
        FixedTopK candidates(128);
        const auto *c_data = centroid_topk.Data();
        for (std::size_t i = 0; i < centroid_topk.Size(); ++i)
        {
            std::size_t c = static_cast<std::size_t>(c_data[i].id);
            for (std::size_t j = grains.offsets[c]; j < grains.offsets[c + 1]; ++j)
            {
                std::size_t row = grains.postings[j];
                float d = kernels::L2Sqr_SQ8_AVX2(snap->qdata.data() + row * dim, qquant.data(), dim);
                candidates.Push(-d, static_cast<Id>(row));
            }
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
        return resp;
    }

    SearchResponse Shard::Search(const SearchRequest &req, const pomai::ai::Budget &budget) const
    {
        const auto start = std::chrono::steady_clock::now();
        auto snapshot = state_.load(std::memory_order_acquire);
        SearchResponse out;
        if (!snapshot)
            return out;
        for (const auto &s : snapshot->segments)
        {
            SearchResponse r;
            if (s.grains && s.snap)
                r = SearchGrains(s.snap, *s.grains, req, budget);
            else if (s.index)
                r = s.index->Search(req.query, budget);
            else if (s.snap)
                r = Seed::SearchSnapshot(s.snap, req);
            MergeTopK(out, r, req.topk);
        }
        SearchResponse lr;
        if (snapshot->live_snap && snapshot->live_grains)
            lr = SearchGrains(snapshot->live_snap, *snapshot->live_grains, req, budget);
        else if (snapshot->live_snap)
            lr = Seed::SearchSnapshot(snapshot->live_snap, req);
        MergeTopK(out, lr, req.topk);
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
                try
                {
                    wal_.WaitDurable(wal_.WrittenLsn());
                    auto snap = seed_.MakeSnapshot();
                    std::string path = wal_dir_ + "/cp-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".bin";
                    int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
                    if (fd >= 0)
                    {
                        uint16_t d = static_cast<uint16_t>(snap->dim);
                        uint64_t c = static_cast<uint64_t>(snap->ids.size());
                        CheckedWrite(fd, &d, 2);
                        CheckedWrite(fd, &c, 8);
                        CheckedWrite(fd, snap->qmins.data(), snap->qmins.size() * 4);
                        CheckedWrite(fd, snap->qscales.data(), snap->qscales.size() * 4);
                        CheckedWrite(fd, snap->ids.data(), snap->ids.size() * 8);
                        CheckedWrite(fd, snap->qdata.data(), snap->qdata.size());
                        ::fdatasync(fd);
                        ::close(fd);
                        wal_.TruncateToZero();
                        if (task.checkpoint_done)
                            task.checkpoint_done->set_value(true);
                    }
                }
                catch (...)
                {
                    if (task.checkpoint_done)
                        task.checkpoint_done->set_value(false);
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
                    auto g = BuildGrainIndex(s);
                    std::lock_guard<std::mutex> lk(writer_mu_);
                    auto prev = state_.load(std::memory_order_acquire);
                    auto next = std::make_shared<ShardState>(prev ? *prev : ShardState{});
                    next->live_snap = std::move(s);
                    next->live_grains = std::move(g);
                    PublishState(std::move(next));
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
        std::size_t pos;
        {
            std::lock_guard<std::mutex> lk(writer_mu_);
            auto prev = state_.load(std::memory_order_acquire);
            auto next = std::make_shared<ShardState>(prev ? *prev : ShardState{});
            next->segments.push_back({snap, nullptr, nullptr});
            pos = next->segments.size() - 1;
            PublishState(std::move(next));
        }
        if (build_pool_)
        {
            IndexBuildPool::Job job{pos, snap, 48, 200, [this](std::size_t p, Seed::Snapshot s, std::shared_ptr<pomai::core::OrbitIndex> i)
                                    {
                                        auto g = this->BuildGrainIndex(s);
                                        this->AttachIndex(p, std::move(s), std::move(i), std::move(g));
                                    }};
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
            next->live_snap = std::move(live_snap);
            next->live_grains.reset();
            PublishState(std::move(next));
        }
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
        next->segments[pos].snap.reset();
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
}
