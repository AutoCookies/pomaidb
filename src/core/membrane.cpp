#include "membrane.h"
#include "spatial_router.h"
#include "search_fanout.h"
#include "memory_manager.h"
#include "fixed_topk.h"

#include <stdexcept>
#include <algorithm>
#include <future>
#include <thread>
#include <chrono>
#include <array>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <random>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <cerrno>
#include <string_view>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace pomai
{
    namespace
    {
        constexpr std::size_t kCentroidsHeaderSize = 8 + 4 + 2 + 8;
        constexpr std::uint32_t kCentroidsVersion = 1;
        constexpr char kCentroidsMagic[8] = {'P', 'O', 'M', 'C', 'E', 'N', '0', '7'};
        constexpr std::size_t kShardCandidateMultiplier = 3;

        static std::size_t ChooseSearchPoolWorkers(std::size_t requested, std::size_t shard_count)
        {
            const std::size_t min_workers = 1;
            const std::size_t max_workers = 8;
            unsigned hc = std::thread::hardware_concurrency();
            if (hc == 0)
                hc = 1;
            std::size_t w = 0;
            if (requested == 0)
                w = std::min<std::size_t>(static_cast<std::size_t>(hc), shard_count);
            else
                w = requested;
            if (w < min_workers)
                w = min_workers;
            if (w > max_workers)
                w = max_workers;
            return w;
        }

        bool WriteFull(int fd, const void *buf, std::size_t len)
        {
            const char *p = static_cast<const char *>(buf);
            std::size_t remaining = len;
            while (remaining > 0)
            {
                ssize_t w = ::write(fd, p, remaining);
                if (w < 0)
                {
                    if (errno == EINTR)
                        continue;
                    return false;
                }
                if (w == 0)
                    return false;
                p += static_cast<std::size_t>(w);
                remaining -= static_cast<std::size_t>(w);
            }
            return true;
        }

        bool ReadFull(int fd, void *buf, std::size_t len)
        {
            char *p = static_cast<char *>(buf);
            std::size_t remaining = len;
            while (remaining > 0)
            {
                ssize_t r = ::read(fd, p, remaining);
                if (r < 0)
                {
                    if (errno == EINTR)
                        continue;
                    return false;
                }
                if (r == 0)
                    return false;
                p += static_cast<std::size_t>(r);
                remaining -= static_cast<std::size_t>(r);
            }
            return true;
        }

        std::string ParentDir(const std::string &path)
        {
            std::filesystem::path p(path);
            if (p.has_parent_path())
                return p.parent_path().string();
            return ".";
        }

        bool FsyncDir(const std::string &path)
        {
            std::string dir = ParentDir(path);
            int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            if (dfd < 0)
                return false;
            int rc = ::fsync(dfd);
            ::close(dfd);
            return rc == 0;
        }

        std::uint16_t Le16ToHost(std::uint16_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap16(v);
#else
            return v;
#endif
        }

        std::uint32_t Le32ToHost(std::uint32_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap32(v);
#else
            return v;
#endif
        }

        std::uint64_t Le64ToHost(std::uint64_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap64(v);
#else
            return v;
#endif
        }

        std::uint16_t HostToLe16(std::uint16_t v) { return Le16ToHost(v); }
        std::uint32_t HostToLe32(std::uint32_t v) { return Le32ToHost(v); }
        std::uint64_t HostToLe64(std::uint64_t v) { return Le64ToHost(v); }

        float LeFloatToHost(float v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            std::uint32_t u;
            std::memcpy(&u, &v, sizeof(u));
            u = __builtin_bswap32(u);
            std::memcpy(&v, &u, sizeof(u));
#endif
            return v;
        }

        float HostToLeFloat(float v) { return LeFloatToHost(v); }

        bool FileExists(const std::string &path)
        {
            struct stat st;
            return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
        }
    }

    MembraneRouter::MembraneRouter(std::vector<std::unique_ptr<Shard>> shards,
                                   pomai::server::WhisperConfig w_cfg,
                                   std::size_t dim,
                                   std::size_t search_pool_workers,
                                   std::size_t search_timeout_ms,
                                   std::function<void()> on_rejected_upsert)
        : shards_(std::move(shards)),
          brain_(w_cfg),
          probe_P_(2),
          dim_(dim),
          search_timeout_ms_(search_timeout_ms),
          search_pool_(ChooseSearchPoolWorkers(search_pool_workers, shards_.size())),
          on_rejected_upsert_(std::move(on_rejected_upsert))
    {
        if (shards_.empty())
            throw std::runtime_error("must have at least 1 shard");
    }

    void MembraneRouter::Start()
    {
        completion_.Start();
        if (centroids_load_mode_ != CentroidsLoadMode::None && centroids_load_mode_ != CentroidsLoadMode::Async)
        {
            if (!centroids_path_.empty() && FileExists(centroids_path_))
            {
                LoadCentroidsFromFile(centroids_path_);
            }
        }

        std::vector<std::future<void>> futures;
        futures.reserve(shards_.size());
        for (auto &s : shards_)
        {
            futures.push_back(std::async(std::launch::async, [&s]()
                                         { s->Start(); }));
        }
        for (auto &f : futures)
            f.get();
    }

    void MembraneRouter::Stop()
    {
        for (auto &s : shards_)
            s->Stop();
        search_pool_.Stop();
        completion_.Stop();
    }

    std::size_t MembraneRouter::PickShardById(Id id) const
    {
        return static_cast<std::size_t>(id % shards_.size());
    }

    std::size_t MembraneRouter::PickShard(Id id, const Vector *vec_opt) const
    {
        if (vec_opt)
        {
            try
            {
                std::size_t c_idx = router_.PickShardForInsert(*vec_opt);
                if (!centroid_to_shard_.empty())
                    return centroid_to_shard_[c_idx % centroid_to_shard_.size()];
                return c_idx % shards_.size();
            }
            catch (...)
            {
            }
        }
        return PickShardById(id);
    }

    std::future<Lsn> MembraneRouter::Upsert(Id id, Vector vec, bool wait_durable)
    {
        UpsertRequest r;
        r.id = id;
        r.vec = std::move(vec);
        std::vector<UpsertRequest> batch;
        batch.push_back(std::move(r));
        return UpsertBatch(std::move(batch), wait_durable);
    }

    std::future<Lsn> MembraneRouter::UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        if (batch.empty())
        {
            std::promise<Lsn> p;
            p.set_value(0);
            return p.get_future();
        }

        std::size_t est_bytes = 0;
        for (const auto &r : batch)
        {
            if (r.vec.data.size() == dim_)
                est_bytes += sizeof(Id) + dim_ * sizeof(float);
        }

        if (!MemoryManager::Instance().CanAllocate(est_bytes))
        {
            if (on_rejected_upsert_)
                on_rejected_upsert_();
            std::promise<Lsn> p;
            p.set_exception(std::make_exception_ptr(std::runtime_error("UpsertBatch rejected: memory pressure")));
            return p.get_future();
        }

        std::vector<std::vector<UpsertRequest>> parts(shards_.size());
        for (auto &r : batch)
        {
            parts[PickShard(r.id, &r.vec)].push_back(std::move(r));
        }

        std::vector<std::future<Lsn>> futs;
        for (std::size_t i = 0; i < parts.size(); ++i)
        {
            if (!parts[i].empty())
                futs.push_back(shards_[i]->EnqueueUpserts(std::move(parts[i]), wait_durable));
        }

        std::promise<Lsn> done;
        auto out = done.get_future();
        auto task = [futs = std::move(futs), done = std::move(done)]() mutable
        {
            Lsn max_lsn = 0;
            try
            {
                for (auto &f : futs)
                {
                    Lsn l = f.get();
                    if (l > max_lsn)
                        max_lsn = l;
                }
                done.set_value(max_lsn);
            }
            catch (...)
            {
                done.set_exception(std::current_exception());
            }
        };
        if (!completion_.Enqueue(std::move(task)))
            task();

        return out;
    }

    std::size_t MembraneRouter::TotalApproxCountUnsafe() const
    {
        std::size_t sum = 0;
        for (const auto &s : shards_)
            sum += s->ApproxCountUnsafe();
        return sum;
    }

    SearchResponse MembraneRouter::Search(const SearchRequest &req) const
    {
        auto start = std::chrono::steady_clock::now();
        auto deadline = start + std::chrono::milliseconds(search_timeout_ms_);
        auto budget = brain_.compute_budget(false);
        auto health = brain_.health();
        std::size_t adaptive_probe = probe_P_;

        if (health == pomai::ai::WhisperGrain::BudgetHealth::Tight)
            adaptive_probe = std::max<std::size_t>(1, probe_P_ / 2);
        else
            adaptive_probe = std::min<std::size_t>(probe_P_ + 1, shards_.size());

        std::vector<std::size_t> target_ids;
        try
        {
            auto c_idxs = router_.CandidateShardsForQuery(req.query, adaptive_probe);
            if (c_idxs.empty())
            {
                target_ids.resize(shards_.size());
                std::iota(target_ids.begin(), target_ids.end(), 0);
            }
            else
            {
                for (auto cidx : c_idxs)
                {
                    target_ids.push_back((!centroid_to_shard_.empty()) ? centroid_to_shard_[cidx % centroid_to_shard_.size()] : (cidx % shards_.size()));
                }
                std::vector<std::size_t> uniq;
                std::unordered_set<std::size_t> seen;
                for (auto s : target_ids)
                    if (seen.insert(s).second)
                        uniq.push_back(s);
                target_ids.swap(uniq);
            }
        }
        catch (...)
        {
            target_ids.resize(shards_.size());
            std::iota(target_ids.begin(), target_ids.end(), 0);
        }

        const std::size_t shard_topk = std::max<std::size_t>(req.topk, req.topk * kShardCandidateMultiplier);
        SearchRequest shard_req = req;
        shard_req.topk = shard_topk;
        auto req_ptr = std::make_shared<SearchRequest>(std::move(shard_req));

        std::vector<std::future<SearchResponse>> futs;
        std::vector<SearchResponse> inline_responses;
        for (auto sid : target_ids)
        {
            Shard *sh_ptr = shards_[sid].get();
            try
            {
                futs.emplace_back(search_pool_.Submit([req_ptr, budget, sh_ptr]()
                                                      { return sh_ptr->Search(*req_ptr, budget); }));
            }
            catch (...)
            {
                search_overload_.fetch_add(1, std::memory_order_relaxed);
                if (std::chrono::steady_clock::now() < deadline)
                {
                    inline_responses.push_back(sh_ptr->Search(*req_ptr, budget));
                    search_inline_.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        thread_local std::unique_ptr<FixedTopK> merge_topk;
        if (!merge_topk)
            merge_topk = std::make_unique<FixedTopK>(req.topk);
        merge_topk->Reset(req.topk);

        for (auto &f : futs)
        {
            if (f.wait_until(deadline) == std::future_status::ready)
            {
                try
                {
                    auto r = f.get();
                    for (const auto &item : r.items)
                        merge_topk->Push(item.score, item.id);
                }
                catch (...)
                {
                }
            }
        }
        for (const auto &r : inline_responses)
        {
            for (const auto &item : r.items)
                merge_topk->Push(item.score, item.id);
        }

        SearchResponse out;
        merge_topk->FillSorted(out.items);
        float lat = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
        brain_.observe_latency(lat);

        if (lat > 50.0f || brain_.latency_ema() > 50.0f)
        {
            for (auto &s : shards_)
                s->RequestEmergencyFreeze();
        }

        if ((search_count_.fetch_add(1, std::memory_order_relaxed) % 128) == 0)
        {
            auto hs = router_.DetectHotspot();
            std::lock_guard<std::mutex> lk(hotspot_mu_);
            if (hs)
                hotspot_ = HotspotInfo{(!centroid_to_shard_.empty()) ? centroid_to_shard_[hs->centroid_idx % centroid_to_shard_.size()] : (hs->centroid_idx % shards_.size()), hs->centroid_idx, hs->ratio};
            else
                hotspot_.reset();
        }

        return out;
    }

    std::future<bool> MembraneRouter::RequestCheckpoint()
    {
        std::vector<std::future<bool>> futs;
        for (auto &s : shards_)
            futs.push_back(s->RequestCheckpoint());
        std::promise<bool> done;
        auto out = done.get_future();
        auto task = [futs = std::move(futs), done = std::move(done)]() mutable
        {
            try
            {
                for (auto &f : futs)
                {
                    if (!f.get())
                    {
                        done.set_value(false);
                        return;
                    }
                }
                done.set_value(true);
            }
            catch (...)
            {
                done.set_exception(std::current_exception());
            }
        };
        if (!completion_.Enqueue(std::move(task)))
            task();
        return out;
    }

    void MembraneRouter::ConfigureCentroids(const std::vector<Vector> &centroids)
    {
        router_.ReplaceCentroids(centroids);
        centroid_to_shard_.clear();
        for (std::size_t i = 0; i < centroids.size(); ++i)
            centroid_to_shard_.push_back(i % shards_.size());
    }

    void MembraneRouter::SetProbeCount(std::size_t p) { probe_P_ = (p == 0 ? 1 : p); }
    double MembraneRouter::SearchQueueAvgLatencyMs() const { return search_pool_.QueueWaitEmaMs(); }
    std::optional<MembraneRouter::HotspotInfo> MembraneRouter::CurrentHotspot() const
    {
        std::lock_guard<std::mutex> lk(hotspot_mu_);
        return hotspot_;
    }
    std::vector<Vector> MembraneRouter::SnapshotCentroids() const { return router_.SnapshotCentroids(); }
    bool MembraneRouter::HasCentroids() const { return !router_.SnapshotCentroids().empty(); }
    bool MembraneRouter::ScheduleCompletion(std::function<void()> fn, std::chrono::steady_clock::duration delay)
    {
        return completion_.Enqueue(std::move(fn), delay);
    }

    bool MembraneRouter::ComputeAndConfigureCentroids(std::size_t k, std::size_t total_samples)
    {
        if (shards_.empty())
            return false;
        const std::size_t S = shards_.size();
        std::size_t sample_size = std::clamp<std::size_t>(total_samples, 50000, 200000);
        auto &mm = MemoryManager::Instance();
        const std::size_t total = mm.TotalUsage();
        const std::size_t hard = mm.HardWatermarkBytes();
        const std::size_t avail = (hard > total) ? (hard - total) : 0;
        const std::size_t max_samples_by_mem = (dim_ > 0) ? (avail / (dim_ * sizeof(float))) : 0;
        if (max_samples_by_mem == 0)
            return false;
        sample_size = std::min(sample_size, max_samples_by_mem);
        std::vector<std::future<std::vector<Vector>>> futs;
        for (std::size_t i = 0; i < S; ++i)
        {
            Shard *sh = shards_[i].get();
            futs.push_back(std::async(std::launch::async, [sh, sample_size, S]()
                                      { return sh->SampleVectors(sample_size / S); }));
        }
        std::vector<Vector> aggregate;
        for (auto &f : futs)
        {
            try
            {
                auto p = f.get();
                aggregate.insert(aggregate.end(), std::make_move_iterator(p.begin()), std::make_move_iterator(p.end()));
            }
            catch (...)
            {
            }
        }
        if (aggregate.empty())
            return false;
        try
        {
            auto c = SpatialRouter::BuildKMeans(aggregate, k == 0 ? S * 32 : k, 10);
            ConfigureCentroids(c);
            if (!centroids_path_.empty())
                SaveCentroidsToFile(centroids_path_);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    bool MembraneRouter::LoadCentroidsFromFile(const std::string &path)
    {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            return false;
        struct stat st;
        if (::fstat(fd, &st) != 0)
        {
            ::close(fd);
            return false;
        }
        std::array<char, 8> magic;
        std::uint32_t ver_le;
        std::uint16_t dim_le;
        std::uint64_t count_le;
        if (!ReadFull(fd, magic.data(), 8) || !ReadFull(fd, &ver_le, 4) || !ReadFull(fd, &dim_le, 2) || !ReadFull(fd, &count_le, 8))
        {
            ::close(fd);
            return false;
        }
        std::size_t count = Le64ToHost(count_le);
        std::size_t dim = Le16ToHost(dim_le);
        std::vector<float> flat(count * dim);
        if (!ReadFull(fd, flat.data(), count * dim * 4))
        {
            ::close(fd);
            return false;
        }
        std::uint64_t m_count_le;
        ReadFull(fd, &m_count_le, 8);
        std::vector<std::uint32_t> m(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            std::uint32_t v;
            ReadFull(fd, &v, 4);
            m[i] = Le32ToHost(v);
        }
        ::close(fd);
        std::vector<Vector> c(count);
        for (std::size_t i = 0; i < count; ++i)
            c[i].data.assign(flat.begin() + i * dim, flat.begin() + (i + 1) * dim);
        ConfigureCentroids(c);
        centroid_to_shard_.clear();
        for (auto val : m)
            centroid_to_shard_.push_back(val % shards_.size());
        return true;
    }

    bool MembraneRouter::SaveCentroidsToFile(const std::string &path) const
    {
        auto c = router_.SnapshotCentroids();
        if (c.empty())
            return false;
        std::string tmp = path + ".tmp";
        int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0)
            return false;
        WriteFull(fd, kCentroidsMagic, 8);
        std::uint32_t v_le = HostToLe32(kCentroidsVersion);
        std::uint16_t d_le = HostToLe16(static_cast<std::uint16_t>(dim_));
        std::uint64_t c_le = HostToLe64(c.size());
        WriteFull(fd, &v_le, 4);
        WriteFull(fd, &d_le, 2);
        WriteFull(fd, &c_le, 8);
        for (const auto &v : c)
            WriteFull(fd, v.data.data(), dim_ * 4);
        std::uint64_t mc_le = HostToLe64(centroid_to_shard_.size());
        WriteFull(fd, &mc_le, 8);
        for (auto sidx : centroid_to_shard_)
        {
            std::uint32_t m_le = HostToLe32(static_cast<std::uint32_t>(sidx));
            WriteFull(fd, &m_le, 4);
        }
        ::fdatasync(fd);
        ::close(fd);
        ::rename(tmp.c_str(), path.c_str());
        FsyncDir(path);
        return true;
    }

    void MembraneRouter::SetCentroidsFilePath(const std::string &path) { centroids_path_ = path; }
    void MembraneRouter::SetCentroidsLoadMode(CentroidsLoadMode mode) { centroids_load_mode_ = mode; }
}
