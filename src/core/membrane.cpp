#include "membrane.h"
#include "spatial_router.h"

#include <stdexcept>
#include <algorithm>
#include <future>
#include <thread>
#include <chrono>
#include <vector>
#include <unordered_set>
#include <numeric>
#include <random>
#include <iostream>

namespace pomai
{

    MembraneRouter::MembraneRouter(std::vector<std::unique_ptr<Shard>> shards, pomai::server::WhisperConfig w_cfg)
        : shards_(std::move(shards)), brain_(w_cfg), probe_P_(2)
    {
        if (shards_.empty())
            throw std::runtime_error("must have at least 1 shard");
    }

    void MembraneRouter::Start()
    {
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

        // If router has no centroids configured, fallback to broadcasting all shards (legacy behavior)
        std::vector<std::size_t> target_shard_ids;
        try
        {
            auto centroid_idxs = router_.CandidateShardsForQuery(req.query, probe_P_);
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

        std::vector<std::future<SearchResponse>> futs;
        futs.reserve(target_shard_ids.size());

        for (auto sid : target_shard_ids)
        {
            // avoid capturing member expressions in lambda capture list; take raw pointer first
            pomai::Shard *shard_ptr = shards_[sid].get();
            futs.push_back(std::async(std::launch::async, [&req, &budget, shard_ptr]()
                                      { return shard_ptr->Search(req, budget); }));
        }

        std::vector<SearchResultItem> all;
        for (auto &f : futs)
        {
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

    // Compute centroids from samples across shards and install them atomically.
    bool MembraneRouter::ComputeAndConfigureCentroids(std::size_t k, std::size_t total_samples)
    {
        if (k == 0 || shards_.empty())
            return false;

        // per-shard sample budget (at least 1)
        std::size_t per_shard = std::max<std::size_t>(1, total_samples / shards_.size());

        // Collect samples concurrently from shards
        std::vector<std::future<std::vector<Vector>>> futs;
        futs.reserve(shards_.size());
        for (auto &s : shards_)
        {
            futs.push_back(std::async(std::launch::async, [s = s.get(), per_shard]()
                                      { return s->SampleVectors(per_shard); }));
        }

        std::vector<Vector> aggregate;
        for (auto &f : futs)
        {
            try
            {
                auto part = f.get();
                if (!part.empty())
                {
                    aggregate.insert(aggregate.end(), std::make_move_iterator(part.begin()), std::make_move_iterator(part.end()));
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[Router] sample gather failed: " << e.what() << "\n";
            }
            catch (...)
            {
                std::cerr << "[Router] sample gather failed: unknown error\n";
            }
        }

        if (aggregate.empty())
        {
            std::cerr << "[Router] no samples collected for centroid build\n";
            return false;
        }

        // Downsample to reasonable kmeans input size if necessary
        const std::size_t MAX_KMEANS_INPUT = std::max<std::size_t>(4096, k * 64);
        if (aggregate.size() > MAX_KMEANS_INPUT)
        {
            std::mt19937_64 rng(std::random_device{}());
            std::shuffle(aggregate.begin(), aggregate.end(), rng);
            aggregate.resize(MAX_KMEANS_INPUT);
        }

        // Build centroids (this may be moderately CPU-heavy)
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

        // Install new centroids atomically and build simple centroid->shard mapping
        ConfigureCentroids(centroids);
        std::cout << "[Router] ConfigureCentroids: built " << centroids.size() << " centroids\n";

        return true;
    }

} // namespace pomai