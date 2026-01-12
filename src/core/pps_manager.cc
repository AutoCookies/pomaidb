/*
 * src/core/pps_manager.cc
 *
 * PPSM Implementation - Powered by Pomai Orbit.
 *
 * Features:
 * - Distributed Ingestion: Routes keys to shards.
 * - Auto-Training: Initializes Orbit centroids on startup with random samples.
 * - High-Performance Search: Scatter-Gather to Orbit instances.
 */

#include "src/core/pps_manager.h"
#include "src/core/seed.h"
#include "src/core/config.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <queue>
#include <sstream>
#include <cstring>
#include <vector>
#include <random>
#include <filesystem>

namespace pomai::core
{

    PPSM::PPSM(ShardManager *shard_mgr,
               size_t dim,
               size_t max_elements_total,
               bool async_insert_ack)
        : shard_mgr_(shard_mgr),
          dim_(dim),
          max_elements_total_(max_elements_total),
          async_insert_ack_(async_insert_ack)
    {
        if (!shard_mgr_)
            throw std::invalid_argument("PPSM: shard_mgr is null");

        uint32_t discovered = 0;
        for (uint32_t i = 0; i < 1024; ++i)
        {
            if (shard_mgr_->get_shard_by_id(i) == nullptr)
                break;
            ++discovered;
        }

        if (discovered == 0)
            throw std::runtime_error("PPSM: shard_mgr has no shards available");

        shards_.reserve(discovered);
        size_t per_shard_max = std::max<size_t>(1, max_elements_total_ / discovered);

        // Initialize state for each shard
        for (uint32_t i = 0; i < discovered; ++i)
        {
            Shard *sh = shard_mgr_->get_shard_by_id(i);
            if (!sh)
                break;

            auto s = std::make_unique<ShardState>();
            s->id = i;
            s->arena = sh->get_arena();
            s->map = sh->get_map();

            if (!initPerShard(*s, dim_, per_shard_max))
            {
                std::cerr << "PPSM: Failed to init Orbit for shard " << i << "\n";
            }

            shards_.push_back(std::move(s));
        }

        startWorkers();
    }

    PPSM::~PPSM()
    {
        stopWorkers();
        shards_.clear();
    }

    // Initialize PomaiOrbit for a shard.
    // Performs "Cold Start" training using random data if the index is new.
    bool PPSM::initPerShard(ShardState &s, size_t dim, size_t per_shard_max)
    {
        try
        {
            ai::orbit::PomaiOrbit::Config cfg;
            cfg.dim = dim;

            // Adaptive configuration based on scale
            // Rough rule of thumb: ~sqrt(N) centroids
            size_t target_centroids = static_cast<size_t>(std::sqrt(per_shard_max));
            target_centroids = std::max<size_t>(256, std::min<size_t>(target_centroids, 4096));
            cfg.num_centroids = target_centroids;

            cfg.m_neighbors = 16; // Robust connectivity
            cfg.use_pq = true;
            cfg.use_fingerprint = true;

            s.orbit = std::make_unique<ai::orbit::PomaiOrbit>(cfg, s.arena);

            // --- Auto-Train Strategy ---
            // Orbit requires trained centroids to route vectors.
            // We generate synthetic data to initialize the routing graph structure.
            // In a production persistent system, we would load existing centroids from disk.
            // Since this is an in-memory/arena hybrid without explicit model persistence yet,
            // we train on init.

            std::cout << "[PPSM] Shard " << s.id << ": Auto-training Orbit with " << cfg.num_centroids << " centroids...\n";

            // Generate training samples (enough to populate centroids comfortably)
            size_t n_train = std::max<size_t>(cfg.num_centroids * 20, 5000);
            std::vector<float> samples(n_train * dim);

            std::mt19937 rng(1337 + s.id); // Deterministic seed per shard
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (float &v : samples)
                v = dist(rng);

            if (!s.orbit->train(samples.data(), n_train))
            {
                std::cerr << "[PPSM] Orbit training failed for shard " << s.id << "\n";
                return false;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "PPSM initPerShard exception: " << e.what() << "\n";
            return false;
        }
        return true;
    }

    uint32_t PPSM::computeShard(const char *key, size_t klen) const noexcept
    {
        if (shards_.empty())
            return 0;
        std::hash<std::string_view> h;
        uint64_t hv = h(std::string_view(key, klen));
        uint32_t cnt = static_cast<uint32_t>(shards_.size());
        if ((cnt & (cnt - 1)) == 0)
            return static_cast<uint32_t>(hv & (cnt - 1));
        return static_cast<uint32_t>(hv % cnt);
    }

    void PPSM::startWorkers()
    {
        for (auto &s : shards_)
        {
            s->running = true;
            ShardState *state = s.get();
            s->worker = std::thread([this, state]()
                                    { this->workerLoop(*state); });

            // Pin thread to core if on Linux
#ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            unsigned int core = state->id % std::max<unsigned>(1, std::thread::hardware_concurrency());
            CPU_SET(core, &cpuset);
            pthread_setaffinity_np(s->worker.native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
        }
    }

    void PPSM::stopWorkers()
    {
        for (auto &s : shards_)
        {
            if (!s)
                continue;
            {
                std::lock_guard<std::mutex> lg(s->q_mu);
                s->running = false;
            }
            s->q_cv.notify_all();
        }
        for (auto &s : shards_)
        {
            if (s && s->worker.joinable())
                s->worker.join();
        }
    }

    void PPSM::workerLoop(ShardState &sh)
    {
        while (true)
        {
            std::unique_ptr<Task> task;
            {
                std::unique_lock<std::mutex> lk(sh.q_mu);
                sh.q_cv.wait(lk, [&]()
                             { return !sh.q.empty() || !sh.running; });
                if (!sh.running && sh.q.empty())
                    break;
                if (!sh.q.empty())
                {
                    task = std::move(sh.q.front());
                    sh.q.pop_front();
                }
            }

            if (!task)
                continue;

            bool ok = false;
            try
            {
                // 1. Insert into Orbit (The Engine)
                if (sh.orbit)
                {
                    ok = sh.orbit->insert(task->vec.data(), task->label);
                }

                // 2. Update Map (Key-Value)
                if (ok && sh.map)
                {
                    uint64_t le = task->label;
                    const char *vptr = reinterpret_cast<const char *>(&le);
                    bool putok = sh.map->put(task->key.data(), static_cast<uint32_t>(task->key.size()), vptr, static_cast<uint32_t>(sizeof(le)));
                    if (putok)
                    {
                        Seed *sseed = sh.map->find_seed(task->key.data(), static_cast<uint32_t>(task->key.size()));
                        if (sseed)
                            sseed->type = Seed::OBJ_VECTOR;
                    }
                }

                // 3. Update Reverse Lookup
                if (ok)
                {
                    std::lock_guard<std::mutex> gl(sh.label_map_mu);
                    sh.label_to_key[task->label] = task->key;
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "PPSM worker shard " << sh.id << " exception: " << e.what() << "\n";
                ok = false;
            }

            try
            {
                task->done.set_value(ok);
            }
            catch (...)
            {
            }
        }
    }

    bool PPSM::addVec(const char *key, size_t klen, const float *vec)
    {
        if (!key || klen == 0 || !vec)
            return false;

        uint32_t sid = computeShard(key, klen);
        if (sid >= shards_.size())
            sid = 0;
        ShardState &sh = *shards_[sid];

        // Check existing label (Upsert logic)
        uint64_t label = 0;
        bool replace = false;
        if (sh.map)
        {
            uint32_t outlen = 0;
            const char *val = sh.map->get(key, static_cast<uint32_t>(klen), &outlen);
            if (val && outlen == sizeof(uint64_t))
            {
                std::memcpy(&label, val, sizeof(label));
                replace = true;
            }
        }

        auto t = std::make_unique<Task>();
        t->key.assign(key, key + klen);
        t->vec.assign(vec, vec + dim_);
        t->replace = replace;

        if (replace)
            t->label = label; // Re-use label (Orbit appends new version, old version effectively garbage collected later)
        else
            t->label = next_label_global_.fetch_add(1, std::memory_order_relaxed);

        std::future<bool> fut = t->done.get_future();

        {
            std::lock_guard<std::mutex> lg(sh.q_mu);
            sh.q.push_back(std::move(t));
        }
        sh.q_cv.notify_one();

        if (!async_insert_ack_)
        {
            try
            {
                return fut.get();
            }
            catch (...)
            {
                return false;
            }
        }
        return true;
    }

    bool PPSM::removeKey(const char *key, size_t klen)
    {
        if (!key || klen == 0)
            return false;
        uint32_t sid = computeShard(key, klen);
        if (sid >= shards_.size())
            sid = 0;
        ShardState &sh = *shards_[sid];

        bool map_erased = false;
        if (sh.map)
            map_erased = sh.map->erase(std::string(key, key + klen).c_str());

        if (map_erased)
        {
            std::lock_guard<std::mutex> gl(sh.label_map_mu);
            // Linear scan clean up from label map (slow, but removal is rare/background)
            // Ideally should look up label first then erase.
            // For now, accept eventual consistency or implement proper 2-way map.
        }
        // Orbit is append-only; actual data removal from buckets requires compaction (future feature).
        return map_erased;
    }

    std::vector<std::pair<std::string, float>> PPSM::search(const float *query, size_t dim, size_t topk)
    {
        if (!query || dim != dim_ || topk == 0)
            return {};

        // SCATTER: Query all Orbit instances
        std::vector<std::future<std::vector<std::pair<uint64_t, float>>>> futs;
        futs.reserve(shards_.size());

        for (auto &s : shards_)
        {
            ShardState *shptr = s.get();
            futs.push_back(std::async(std::launch::async, [shptr, query, topk]()
                                      {
                if (!shptr->orbit) return std::vector<std::pair<uint64_t, float>>{};
                // nprobe=3 is a good balance for Orbit
                return shptr->orbit->search(query, topk, 3); }));
        }

        // GATHER: Merge and resolve Keys
        using Item = std::pair<float, std::string>;
        struct Cmp
        {
            bool operator()(const Item &a, const Item &b) const { return a.first < b.first; }
        };
        std::priority_queue<Item, std::vector<Item>, Cmp> heap;

        for (size_t i = 0; i < futs.size(); ++i)
        {
            try
            {
                auto shard_res = futs[i].get();
                ShardState *shptr = shards_[i].get();

                for (const auto &p : shard_res)
                {
                    uint64_t label = p.first;
                    float dist = p.second;
                    std::string key;
                    {
                        std::lock_guard<std::mutex> gl(shptr->label_map_mu);
                        auto it = shptr->label_to_key.find(label);
                        if (it != shptr->label_to_key.end())
                            key = it->second;
                        else
                            key = std::to_string(label);
                    }

                    if (heap.size() < topk)
                        heap.emplace(dist, key);
                    else if (dist < heap.top().first)
                    {
                        heap.pop();
                        heap.emplace(dist, key);
                    }
                }
            }
            catch (...)
            {
            }
        }

        std::vector<std::pair<std::string, float>> out;
        out.reserve(heap.size());
        while (!heap.empty())
        {
            auto it = heap.top();
            heap.pop();
            out.emplace_back(it.second, it.first);
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

    PPSM::MemoryUsage PPSM::memoryUsage() const noexcept
    {
        MemoryUsage out{0, 0, 0};
        // Orbit memory accounting is tricky (Bucket overheads).
        // For now return dummy or implement detailed stats in PomaiOrbit.
        // Returning 0 to avoid breaking API contracts.
        return out;
    }

} // namespace pomai::core