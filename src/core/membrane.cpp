#include "membrane.h"
#include "spatial_router.h"
#include "search_fanout.h"
#include "memory_manager.h"

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

        // Choose worker count for SearchThreadPool based on requested knob and shard count.
        // - requested == 0 means auto choose based on hardware_concurrency and shard_count.
        // - clamp result to [1, max_cap].
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

        std::uint16_t HostToLe16(std::uint16_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap16(v);
#else
            return v;
#endif
        }

        std::uint32_t HostToLe32(std::uint32_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap32(v);
#else
            return v;
#endif
        }

        std::uint64_t HostToLe64(std::uint64_t v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            return __builtin_bswap64(v);
#else
            return v;
#endif
        }

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

        float HostToLeFloat(float v)
        {
#if __BYTE_ORDER == __BIG_ENDIAN
            std::uint32_t u;
            std::memcpy(&u, &v, sizeof(u));
            u = __builtin_bswap32(u);
            std::memcpy(&v, &u, sizeof(u));
#endif
            return v;
        }

        bool FileExists(const std::string &path)
        {
            struct stat st;
            return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
        }
    } // namespace

    // Constructor updated to accept search_pool_workers knob.
    MembraneRouter::MembraneRouter(std::vector<std::unique_ptr<Shard>> shards,
                                   pomai::server::WhisperConfig w_cfg,
                                   std::size_t dim,
                                   std::size_t search_pool_workers,
                                   std::size_t search_timeout_ms,
                                   std::function<void()> on_rejected_upsert)
        : shards_(std::move(shards)),
          brain_(w_cfg),
          probe_P_(2),
          centroids_path_(),
          centroids_load_mode_(CentroidsLoadMode::Auto),
          dim_(dim),
          search_timeout_ms_(search_timeout_ms),
          // initialize search_pool_ using helper and constructor parameter 'shards' size (use param shards size for decision)
          search_pool_(ChooseSearchPoolWorkers(search_pool_workers, shards_.size())),
          on_rejected_upsert_(std::move(on_rejected_upsert))
    {
        // Log chosen worker count for observability
        std::cout << "[Router] search_pool_workers=" << search_pool_.WorkerCount() << "\n";

        if (shards_.empty())
            throw std::runtime_error("must have at least 1 shard");
    }

    void MembraneRouter::Start()
    {
        if (centroids_load_mode_ != CentroidsLoadMode::None && centroids_load_mode_ != CentroidsLoadMode::Async)
        {
            if (!centroids_path_.empty() && FileExists(centroids_path_))
            {
                if (LoadCentroidsFromFile(centroids_path_))
                {
                    std::cout << "[Router] Loaded centroids from " << centroids_path_ << "\n";
                }
                else
                {
                    std::cerr << "[Router] Failed to load centroids from " << centroids_path_
                              << " (will recompute in background)\n";
                }
            }
            else if (!centroids_path_.empty())
            {
                std::cout << "[Router] No centroids file at " << centroids_path_
                          << " (will recompute in background)\n";
            }
        }

        // PARALLEL BOOT: Khởi động tất cả Shard cùng lúc
        std::vector<std::future<void>> futures;
        futures.reserve(shards_.size());

        for (auto &s : shards_)
        {
            // Launch async: Mỗi shard start trên một thread riêng biệt
            futures.push_back(std::async(std::launch::async, [&s]()
                                         { s->Start(); }));
        }

        // Chờ tất cả Shard khởi động xong trước khi cho Server nhận request
        for (auto &f : futures)
        {
            f.get();
        }
    }

    void MembraneRouter::Stop()
    {
        // Stop cũng nên song song để tắt nhanh, nhưng tuần tự cho an toàn cũng được
        for (auto &s : shards_)
            s->Stop();

        // Stop search pool after shards stopped
        search_pool_.Stop();
    }

    // Legacy id-based fallback
    std::size_t MembraneRouter::PickShardById(Id id) const
    {
        return static_cast<std::size_t>(id % shards_.size());
    }

    // New: pick shard using spatial router if centroids configured, otherwise fallback to id-mod
    std::size_t MembraneRouter::PickShard(Id id, const Vector *vec_opt) const
    {
        // Prefer routing by vector if provided and centroids exist
        if (vec_opt)
        {
            try
            {
                std::size_t centroid_idx = router_.PickShardForInsert(*vec_opt);
                if (!centroid_to_shard_.empty())
                    return centroid_to_shard_[centroid_idx % centroid_to_shard_.size()];
                // fallback mapping if centroid->shard mapping missing
                return centroid_idx % shards_.size();
            }
            catch (...)
            {
                // router not configured; fallthrough to id
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
            auto f = p.get_future();
            p.set_value(0);
            return f;
        }

        std::size_t estimated_bytes = 0;
        for (const auto &r : batch)
        {
            if (r.vec.data.size() != dim_)
                continue;
            estimated_bytes += sizeof(Id) + r.vec.data.size() * sizeof(float);
        }

        if (!MemoryManager::Instance().CanAllocate(estimated_bytes))
        {
            if (on_rejected_upsert_)
                on_rejected_upsert_();
            std::promise<Lsn> p;
            auto f = p.get_future();
            p.set_exception(std::make_exception_ptr(
                std::runtime_error("UpsertBatch rejected: memory pressure")));
            return f;
        }

        // Route each upsert by its vector (if possible), otherwise fallback to id hashing.
        std::vector<std::vector<UpsertRequest>> parts(shards_.size());
        for (auto &r : batch)
        {
            std::size_t shard_id = PickShard(r.id, &r.vec);
            parts[shard_id].push_back(std::move(r));
        }

        std::vector<std::future<Lsn>> futs;
        futs.reserve(shards_.size());

        for (std::size_t i = 0; i < parts.size(); ++i)
        {
            if (!parts[i].empty())
            {
                futs.push_back(shards_[i]->EnqueueUpserts(std::move(parts[i]), wait_durable));
            }
        }

        std::promise<Lsn> done;
        auto out = done.get_future();

        std::thread([futs = std::move(futs), done = std::move(done)]() mutable
                    {
            Lsn max_lsn = 0;
            try {
                for (auto& f : futs) {
                    Lsn l = f.get();
                    if (l > max_lsn) max_lsn = l;
                }
                done.set_value(max_lsn);
            } catch (...) {
                done.set_exception(std::current_exception());
            } })
            .detach();

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

        auto budget = brain_.compute_budget(false);
        auto health = brain_.health();
        std::size_t adaptive_probe = probe_P_;
        if (health == pomai::ai::WhisperGrain::BudgetHealth::Tight)
            adaptive_probe = std::max<std::size_t>(1, probe_P_ / 2);
        else
            adaptive_probe = std::min<std::size_t>(probe_P_ + 1, shards_.size());

        // If router has no centroids configured, fallback to broadcasting all shards (legacy behavior)
        std::vector<std::size_t> target_shard_ids;
        try
        {
            auto centroid_idxs = router_.CandidateShardsForQuery(req.query, adaptive_probe);
            if (centroid_idxs.empty())
            {
                // fallback: probe all shards
                target_shard_ids.resize(shards_.size());
                std::iota(target_shard_ids.begin(), target_shard_ids.end(), 0);
            }
            else
            {
                target_shard_ids.reserve(centroid_idxs.size());
                for (auto cidx : centroid_idxs)
                {
                    std::size_t sid = (!centroid_to_shard_.empty())
                                          ? centroid_to_shard_[cidx % centroid_to_shard_.size()]
                                          : (cidx % shards_.size());
                    target_shard_ids.push_back(sid);
                }
                // deduplicate while preserving order
                std::vector<std::size_t> uniq;
                std::unordered_set<std::size_t> seen;
                uniq.reserve(target_shard_ids.size());
                for (auto s : target_shard_ids)
                {
                    if (seen.insert(s).second)
                        uniq.push_back(s);
                }
                target_shard_ids.swap(uniq);
            }
        }
        catch (...)
        {
            // Router not configured or failed -> broadcast to all shards
            target_shard_ids.resize(shards_.size());
            std::iota(target_shard_ids.begin(), target_shard_ids.end(), 0);
        }

        // Use thread-pool based bounded fanout instead of std::async to avoid thread explosion.
        std::vector<std::function<SearchResponse()>> jobs;
        jobs.reserve(target_shard_ids.size());

        // Capture request via shared_ptr to avoid copying heavy vectors per job.
        auto req_ptr = std::make_shared<SearchRequest>(req);

        for (auto sid : target_shard_ids)
        {
            pomai::Shard *shard_ptr = shards_[sid].get();
            jobs.emplace_back([req_ptr, budget, shard_ptr]()
                              { return shard_ptr->Search(*req_ptr, budget); });
        }

        auto futs = ParallelSubmit(search_pool_, std::move(jobs));

        std::vector<SearchResultItem> all;
        const auto timeout = std::chrono::milliseconds(search_timeout_ms_);
        for (auto &f : futs)
        {
            if (f.wait_for(timeout) != std::future_status::ready)
            {
                std::cerr << "[Router] Search shard timeout after "
                          << search_timeout_ms_ << "ms; returning partial results\n";
                continue;
            }
            auto r = f.get();
            all.insert(all.end(), r.items.begin(), r.items.end());
        }

        std::sort(all.begin(), all.end(), [](const auto &a, const auto &b)
                  { return a.score > b.score; });
        if (all.size() > req.topk)
            all.resize(req.topk);

        SearchResponse out;
        out.items = std::move(all);

        auto end = std::chrono::steady_clock::now();
        float latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
        brain_.observe_latency(latency_ms);

        const std::uint64_t searches = search_count_.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((searches % 128) == 0)
        {
            auto hotspot = router_.DetectHotspot();
            std::lock_guard<std::mutex> lk(hotspot_mu_);
            if (hotspot)
            {
                const std::size_t shard_id = (!centroid_to_shard_.empty())
                                                 ? centroid_to_shard_[hotspot->centroid_idx % centroid_to_shard_.size()]
                                                 : (hotspot->centroid_idx % shards_.size());
                hotspot_ = HotspotInfo{shard_id, hotspot->centroid_idx, hotspot->ratio};
                if (shard_id != last_hotspot_shard_)
                {
                    last_hotspot_shard_ = shard_id;
                    std::cout << "[Router] Hotspot detected on Shard " << shard_id
                              << ". Consider RecomputeCentroids with higher K.\n";
                }
            }
            else
            {
                hotspot_.reset();
            }
        }

        return out;
    }

    std::future<bool> MembraneRouter::RequestCheckpoint()
    {
        std::vector<std::future<bool>> futs;
        futs.reserve(shards_.size());

        for (auto &s : shards_)
        {
            futs.push_back(s->RequestCheckpoint());
        }

        std::promise<bool> done;
        auto out = done.get_future();

        // Aggregate asynchronously so caller gets a future immediately.
        std::thread([futs = std::move(futs), done = std::move(done)]() mutable
                    {
            try
            {
                for (auto &f : futs)
                {
                    bool ok = f.get();
                    if (!ok)
                    {
                        done.set_value(false);
                        return;
                    }
                }
                done.set_value(true);
            }
            catch (...)
            {
                try
                {
                    done.set_exception(std::current_exception());
                }
                catch (...)
                {
                }
            } })
            .detach();

        return out;
    }

    // Admin / management helpers

    // Configure centroids (replace atomically). Also build a simple centroid->shard mapping
    // using round-robin assignment so centroids are distributed across shards.
    void MembraneRouter::ConfigureCentroids(const std::vector<Vector> &centroids)
    {
        // Replace router centroids
        router_.ReplaceCentroids(centroids);

        // Build centroid->shard mapping: round-robin assign centroids to shards for initial balance.
        centroid_to_shard_.clear();
        centroid_to_shard_.reserve(centroids.size());
        for (std::size_t i = 0; i < centroids.size(); ++i)
        {
            centroid_to_shard_.push_back(i % shards_.size());
        }
    }

    void MembraneRouter::SetProbeCount(std::size_t p)
    {
        probe_P_ = (p == 0 ? 1 : p);
    }

    double MembraneRouter::SearchQueueAvgLatencyMs() const
    {
        return search_pool_.QueueWaitEmaMs();
    }

    std::optional<MembraneRouter::HotspotInfo> MembraneRouter::CurrentHotspot() const
    {
        std::lock_guard<std::mutex> lk(hotspot_mu_);
        return hotspot_;
    }

    std::vector<Vector> MembraneRouter::SnapshotCentroids() const
    {
        return router_.SnapshotCentroids();
    }

    bool MembraneRouter::HasCentroids() const
    {
        return !router_.SnapshotCentroids().empty();
    }

    // Compute centroids from samples across shards and install them atomically.
    bool MembraneRouter::ComputeAndConfigureCentroids(std::size_t k, std::size_t total_samples)
    {
        if (shards_.empty())
            return false;

        // Autoscaler parameters (tunable)
        const std::size_t target_bucket_size = 50'000; // B: desired vectors per centroid
        const std::size_t min_centroids_per_shard = 16;
        const std::size_t max_centroids_per_shard = 8192;
        const std::size_t max_sample = 200'000;
        const std::size_t k_index_threshold = 512; // if K > threshold, consider building centroid index

        // 1) Measure per-shard sizes
        const std::size_t S = shards_.size();
        std::vector<std::size_t> shard_counts(S);
        std::size_t total_vectors = 0;
        for (std::size_t i = 0; i < S; ++i)
        {
            shard_counts[i] = shards_[i]->ApproxCountUnsafe();
            total_vectors += shard_counts[i];
        }

        // If user provided k==0, run autoscaler to pick K
        std::vector<std::size_t> per_shard_C(S, 0);
        if (k == 0)
        {
            // Compute per-shard centroids C_s = clamp(ceil(V_s / B), min, max)
            std::size_t K = 0;
            for (std::size_t i = 0; i < S; ++i)
            {
                std::size_t cs = 0;
                if (shard_counts[i] > 0)
                {
                    cs = (shard_counts[i] + target_bucket_size - 1) / target_bucket_size;
                    if (cs < min_centroids_per_shard)
                        cs = min_centroids_per_shard;
                    if (cs > max_centroids_per_shard)
                        cs = max_centroids_per_shard;
                }
                else
                {
                    cs = min_centroids_per_shard;
                }
                per_shard_C[i] = cs;
                K += cs;
            }
            k = K;
            if (k == 0)
                k = S * min_centroids_per_shard;
        }
        else
        {
            // If explicit k given, distribute roughly evenly (can be improved later)
            std::size_t base = k / S;
            std::size_t rem = k % S;
            for (std::size_t i = 0; i < S; ++i)
            {
                per_shard_C[i] = base + (i < rem ? 1 : 0);
                if (per_shard_C[i] < min_centroids_per_shard)
                    per_shard_C[i] = min_centroids_per_shard;
                if (per_shard_C[i] > max_centroids_per_shard)
                    per_shard_C[i] = max_centroids_per_shard;
            }
            // recompute k from distribution
            std::size_t K = 0;
            for (auto c : per_shard_C)
                K += c;
            k = K;
        }

        if (k == 0)
            return false;

        // 2) Choose sample size S_kmeans (relative to k)
        std::size_t sample_size = std::max<std::size_t>(10 * k, 50'000);
        if (sample_size > max_sample)
            sample_size = max_sample;
        // If user provided total_samples and it's larger, honor it (but cap by max_sample)
        if (total_samples > sample_size)
            sample_size = std::min<std::size_t>(total_samples, max_sample);

        // 3) Allocate per-shard sample budgets proportional to per_shard_C (so shards with more centroids get more samples)
        std::vector<std::size_t> per_shard_budget(S, 1);
        {
            std::size_t sumC = 0;
            for (auto c : per_shard_C)
                sumC += c;
            if (sumC == 0)
            {
                // uniform allocation
                for (std::size_t i = 0; i < S; ++i)
                    per_shard_budget[i] = std::max<std::size_t>(1, sample_size / S);
            }
            else
            {
                std::size_t assigned = 0;
                for (std::size_t i = 0; i < S; ++i)
                {
                    per_shard_budget[i] = (per_shard_C[i] * sample_size) / sumC;
                    if (per_shard_budget[i] == 0)
                        per_shard_budget[i] = 1;
                    assigned += per_shard_budget[i];
                }
                // distribute remainder
                std::size_t idx = 0;
                while (assigned < sample_size)
                {
                    per_shard_budget[idx % S]++;
                    assigned++;
                    idx++;
                }
            }
        }

        // 4) Gather samples concurrently (collect origin shard id alongside samples)
        std::vector<std::future<std::vector<Vector>>> futs;
        futs.reserve(S);
        for (std::size_t i = 0; i < S; ++i)
        {
            std::size_t bud = per_shard_budget[i];
            // capture raw shard pointer
            Shard *sh = shards_[i].get();
            futs.push_back(std::async(std::launch::async, [sh, bud]()
                                      { return sh->SampleVectors(bud); }));
        }

        std::vector<Vector> aggregate;
        std::vector<std::size_t> sample_origin; // parallel vector storing shard index for each sample
        aggregate.reserve(sample_size);
        sample_origin.reserve(sample_size);

        for (std::size_t i = 0; i < futs.size(); ++i)
        {
            try
            {
                auto part = futs[i].get();
                for (auto &v : part)
                {
                    aggregate.push_back(std::move(v));
                    sample_origin.push_back(i);
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[Router] sample gather failed for shard " << i << ": " << e.what() << "\n";
            }
            catch (...)
            {
                std::cerr << "[Router] sample gather failed for shard " << i << ": unknown error\n";
            }
        }

        if (aggregate.empty())
        {
            std::cerr << "[Router] no samples collected for centroid build\n";
            return false;
        }

        // If we have more samples than allowed, downsample randomly
        if (aggregate.size() > sample_size)
        {
            std::mt19937_64 rng(std::random_device{}());
            std::vector<std::size_t> idx(aggregate.size());
            for (std::size_t i = 0; i < idx.size(); ++i)
                idx[i] = i;
            std::shuffle(idx.begin(), idx.end(), rng);

            std::vector<Vector> agg2;
            std::vector<std::size_t> origin2;
            agg2.reserve(sample_size);
            origin2.reserve(sample_size);
            for (std::size_t t = 0; t < sample_size; ++t)
            {
                agg2.push_back(std::move(aggregate[idx[t]]));
                origin2.push_back(sample_origin[idx[t]]);
            }
            aggregate.swap(agg2);
            sample_origin.swap(origin2);
        }

        // 5) Build centroids (k-means)
        std::vector<Vector> centroids;
        try
        {
            centroids = SpatialRouter::BuildKMeans(aggregate, k, /*iterations=*/10);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[Router] BuildKMeans failed: " << e.what() << "\n";
            return false;
        }
        catch (...)
        {
            std::cerr << "[Router] BuildKMeans failed: unknown exception\n";
            return false;
        }

        if (centroids.empty())
        {
            std::cerr << "[Router] BuildKMeans returned zero centroids\n";
            return false;
        }

        // 6) Map centroid -> shard by majority voting from the sampled vectors
        // votes[c][s] = number of samples from shard s assigned to centroid c
        const std::size_t K = centroids.size();
        std::vector<std::vector<uint32_t>> votes(K, std::vector<uint32_t>(S, 0u));

        // For each sample, find nearest centroid (linear search on centroids; centroids count K should be manageable)
        for (std::size_t i = 0; i < aggregate.size(); ++i)
        {
            const Vector &v = aggregate[i];
            std::size_t best_c = 0;
            float best_d = std::numeric_limits<float>::infinity();
            for (std::size_t c = 0; c < K; ++c)
            {
                float d = pomai::kernels::L2Sqr(v.data.data(), centroids[c].data.data(), v.data.size());
                if (d < best_d)
                {
                    best_d = d;
                    best_c = c;
                }
            }
            std::size_t shard_origin = sample_origin[i];
            votes[best_c][shard_origin]++;
        }

        // Build mapping
        centroid_to_shard_.clear();
        centroid_to_shard_.resize(K);
        for (std::size_t c = 0; c < K; ++c)
        {
            uint32_t best_count = 0;
            std::size_t best_shard = 0;
            for (std::size_t s = 0; s < S; ++s)
            {
                if (votes[c][s] > best_count)
                {
                    best_count = votes[c][s];
                    best_shard = s;
                }
            }
            centroid_to_shard_[c] = best_shard;
        }

        // 7) Install centroids atomically and persist mapping
        ConfigureCentroids(centroids);
        std::cout << "[Router] ConfigureCentroids: built " << centroids.size() << " centroids\n";

        if (!centroids_path_.empty() && centroids_load_mode_ != CentroidsLoadMode::None)
        {
            if (SaveCentroidsToFile(centroids_path_))
                std::cout << "[Router] Saved centroids to " << centroids_path_ << "\n";
            else
                std::cerr << "[Router] Failed to save centroids to " << centroids_path_ << "\n";
        }

        // Optionally: if K is large, build a small routing index (HNSW/OrbitIndex) over centroids for faster routing.
        // TODO: implement centroid index build when needed.

        return true;
    }

    bool MembraneRouter::LoadCentroidsFromFile(const std::string &path)
    {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
        {
            std::cerr << "[Router] open failed for centroids file " << path
                      << ": " << std::strerror(errno) << "\n";
            return false;
        }

        struct stat st;
        if (::fstat(fd, &st) != 0)
        {
            std::cerr << "[Router] fstat failed for centroids file " << path
                      << ": " << std::strerror(errno) << "\n";
            ::close(fd);
            return false;
        }

        if (static_cast<std::size_t>(st.st_size) < kCentroidsHeaderSize)
        {
            std::cerr << "[Router] centroids file too small: " << path << "\n";
            ::close(fd);
            return false;
        }

        std::array<char, 8> magic{};
        std::uint32_t version_le = 0;
        std::uint16_t dim_le = 0;
        std::uint64_t count_le = 0;

        if (!ReadFull(fd, magic.data(), magic.size()) ||
            !ReadFull(fd, &version_le, sizeof(version_le)) ||
            !ReadFull(fd, &dim_le, sizeof(dim_le)) ||
            !ReadFull(fd, &count_le, sizeof(count_le)))
        {
            std::cerr << "[Router] failed to read centroids header: " << path << "\n";
            ::close(fd);
            return false;
        }

        if (std::memcmp(magic.data(), kCentroidsMagic, sizeof(kCentroidsMagic)) != 0)
        {
            std::cerr << "[Router] invalid centroids magic: " << path << "\n";
            ::close(fd);
            return false;
        }

        std::uint32_t version = Le32ToHost(version_le);
        if (version != kCentroidsVersion)
        {
            std::cerr << "[Router] unsupported centroids version " << version << ": " << path << "\n";
            ::close(fd);
            return false;
        }

        std::uint16_t dim = Le16ToHost(dim_le);
        if (dim_ != 0 && dim != dim_)
        {
            std::cerr << "[Router] centroids dim mismatch: file=" << dim
                      << " expected=" << dim_ << "\n";
            ::close(fd);
            return false;
        }

        std::uint64_t count = Le64ToHost(count_le);
        if (count == 0)
        {
            std::cerr << "[Router] centroids file has zero count: " << path << "\n";
            ::close(fd);
            return false;
        }

        if (dim == 0)
        {
            std::cerr << "[Router] centroids file has zero dim: " << path << "\n";
            ::close(fd);
            return false;
        }

        if (count > (std::numeric_limits<std::size_t>::max() / dim))
        {
            std::cerr << "[Router] centroids file size overflow: " << path << "\n";
            ::close(fd);
            return false;
        }

        std::size_t total_floats = static_cast<std::size_t>(count * dim);
        std::size_t floats_bytes = total_floats * sizeof(float);

        // After header + centroid floats, file must contain mapping_count (u64) and mapping entries (u32 each)
        if (static_cast<std::size_t>(st.st_size) < kCentroidsHeaderSize + floats_bytes + sizeof(std::uint64_t))
        {
            std::cerr << "[Router] centroids file too small for mapping: " << path << "\n";
            ::close(fd);
            return false;
        }

        std::vector<float> flat(total_floats);
        if (!ReadFull(fd, flat.data(), floats_bytes))
        {
            std::cerr << "[Router] failed to read centroids payload: " << path << "\n";
            ::close(fd);
            return false;
        }

        // Read mapping_count
        std::uint64_t mapping_count_le = 0;
        if (!ReadFull(fd, &mapping_count_le, sizeof(mapping_count_le)))
        {
            std::cerr << "[Router] failed to read mapping_count: " << path << "\n";
            ::close(fd);
            return false;
        }
        std::uint64_t mapping_count = Le64ToHost(mapping_count_le);
        if (mapping_count != count)
        {
            std::cerr << "[Router] mapping_count != centroid count: " << path << "\n";
            ::close(fd);
            return false;
        }

        // Ensure enough bytes for mapping entries
        std::size_t mapping_bytes = static_cast<std::size_t>(mapping_count) * sizeof(std::uint32_t);
        off_t cur_off = lseek(fd, 0, SEEK_CUR);
        if (cur_off == -1)
        {
            std::cerr << "[Router] lseek failed reading mapping: " << std::strerror(errno) << "\n";
            ::close(fd);
            return false;
        }
        if (static_cast<std::size_t>(st.st_size) != (kCentroidsHeaderSize + floats_bytes + sizeof(std::uint64_t) + mapping_bytes))
        {
            std::cerr << "[Router] centroids file size mismatch (mapping): expected "
                      << (kCentroidsHeaderSize + floats_bytes + sizeof(std::uint64_t) + mapping_bytes)
                      << " got " << st.st_size << "\n";
            ::close(fd);
            return false;
        }

        std::vector<std::uint32_t> mapping(mapping_count);
        for (std::size_t i = 0; i < mapping_count; ++i)
        {
            std::uint32_t v_le = 0;
            if (!ReadFull(fd, &v_le, sizeof(v_le)))
            {
                std::cerr << "[Router] failed to read mapping entry: " << path << "\n";
                ::close(fd);
                return false;
            }
            mapping[i] = Le32ToHost(v_le);
        }

        ::close(fd);

#if __BYTE_ORDER == __BIG_ENDIAN
        for (auto &v : flat)
            v = LeFloatToHost(v);
#endif

        std::vector<Vector> centroids;
        centroids.reserve(static_cast<std::size_t>(count));
        auto it = flat.begin();
        for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
        {
            Vector v;
            v.data.assign(it, it + dim);
            centroids.push_back(std::move(v));
            it += dim;
        }

        // Install centroids
        ConfigureCentroids(centroids);

        // Fix: define shard count here for mapping validation
        const std::size_t S = shards_.size();

        // Install mapping (ensure mapping values are in shard range)
        centroid_to_shard_.clear();
        centroid_to_shard_.reserve(mapping.size());
        for (auto m : mapping)
        {
            if (m >= S)
            {
                std::cerr << "[Router] mapping value out of range in file: " << path << "\n";
                // fallback: assign by round-robin for out-of-range entries
                centroid_to_shard_.push_back(0);
            }
            else
                centroid_to_shard_.push_back(static_cast<std::size_t>(m));
        }

        return true;
    }

    bool MembraneRouter::SaveCentroidsToFile(const std::string &path) const
    {
        auto centroids = router_.SnapshotCentroids();
        if (centroids.empty())
        {
            std::cerr << "[Router] no centroids to save\n";
            return false;
        }

        std::size_t dim = centroids.front().data.size();
        if (dim == 0)
        {
            std::cerr << "[Router] centroids have zero dim\n";
            return false;
        }
        for (const auto &c : centroids)
        {
            if (c.data.size() != dim)
            {
                std::cerr << "[Router] inconsistent centroid dims\n";
                return false;
            }
        }
        if (dim_ != 0 && dim != dim_)
        {
            std::cerr << "[Router] centroids dim mismatch: " << dim << " expected " << dim_ << "\n";
            return false;
        }

        // mapping must match centroids size
        if (centroid_to_shard_.size() != centroids.size())
        {
            std::cerr << "[Router] centroid_to_shard mapping size mismatch\n";
            return false;
        }

        std::string tmp_path = path + ".tmp";
        int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0)
        {
            std::cerr << "[Router] open failed for " << tmp_path << ": " << std::strerror(errno) << "\n";
            return false;
        }

        if (!WriteFull(fd, kCentroidsMagic, sizeof(kCentroidsMagic)))
        {
            std::cerr << "[Router] write magic failed\n";
            ::close(fd);
            return false;
        }

        std::uint32_t version_le = HostToLe32(kCentroidsVersion);
        std::uint16_t dim_le = HostToLe16(static_cast<std::uint16_t>(dim));
        std::uint64_t count_le = HostToLe64(static_cast<std::uint64_t>(centroids.size()));

        if (!WriteFull(fd, &version_le, sizeof(version_le)) ||
            !WriteFull(fd, &dim_le, sizeof(dim_le)) ||
            !WriteFull(fd, &count_le, sizeof(count_le)))
        {
            std::cerr << "[Router] write header failed\n";
            ::close(fd);
            return false;
        }

        for (const auto &c : centroids)
        {
            for (float f : c.data)
            {
                float out = HostToLeFloat(f);
                if (!WriteFull(fd, &out, sizeof(out)))
                {
                    std::cerr << "[Router] write centroid payload failed\n";
                    ::close(fd);
                    return false;
                }
            }
        }

        // Write mapping: mapping_count (u64) + mapping values (u32 each)
        std::uint64_t mapping_count_le = HostToLe64(static_cast<std::uint64_t>(centroid_to_shard_.size()));
        if (!WriteFull(fd, &mapping_count_le, sizeof(mapping_count_le)))
        {
            std::cerr << "[Router] write mapping_count failed\n";
            ::close(fd);
            return false;
        }
        for (auto sidx : centroid_to_shard_)
        {
            // store as u32
            std::uint32_t m = static_cast<std::uint32_t>(sidx);
            std::uint32_t m_le = HostToLe32(m);
            if (!WriteFull(fd, &m_le, sizeof(m_le)))
            {
                std::cerr << "[Router] write mapping payload failed\n";
                ::close(fd);
                return false;
            }
        }

        if (::fdatasync(fd) != 0)
        {
            std::cerr << "[Router] fdatasync failed for " << tmp_path << ": " << std::strerror(errno) << "\n";
            ::close(fd);
            return false;
        }
        if (::close(fd) != 0)
        {
            std::cerr << "[Router] close failed for " << tmp_path << ": " << std::strerror(errno) << "\n";
            return false;
        }

        if (::rename(tmp_path.c_str(), path.c_str()) != 0)
        {
            std::cerr << "[Router] rename failed from " << tmp_path << " to " << path
                      << ": " << std::strerror(errno) << "\n";
            return false;
        }

        if (!FsyncDir(path))
        {
            std::cerr << "[Router] fsync dir failed for " << path << ": " << std::strerror(errno) << "\n";
            return false;
        }

        return true;
    }

    void MembraneRouter::SetCentroidsFilePath(const std::string &path)
    {
        centroids_path_ = path;
    }

    void MembraneRouter::SetCentroidsLoadMode(CentroidsLoadMode mode)
    {
        centroids_load_mode_ = mode;
    }

} // namespace pomai
