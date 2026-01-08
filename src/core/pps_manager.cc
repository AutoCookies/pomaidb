#include "src/core/pps_manager.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <queue>
#include <sstream>
#include <cstring>
#include <vector>

#include "src/ai/quantize.h"
#include "src/core/seed.h"

namespace pomai::core
{

    static inline uint64_t now_ms()
    {
        using namespace std::chrono;
        return static_cast<uint64_t>(duration_cast<std::chrono::milliseconds>(steady_clock::now().time_since_epoch()).count());
    }

    PPSM::PPSM(ShardManager *shard_mgr,
               size_t dim,
               size_t max_elements_total,
               size_t M,
               size_t ef_construction,
               bool async_insert_ack)
        : shard_mgr_(shard_mgr),
          dim_(dim),
          max_elements_total_(max_elements_total),
          M_(M),
          ef_construction_(ef_construction),
          async_insert_ack_(async_insert_ack)
    {
        if (!shard_mgr_)
            throw std::invalid_argument("PPSM: shard_mgr null");

        uint32_t discovered = 0;
        for (uint32_t i = 0; i < 1024; ++i)
        {
            if (shard_mgr_->get_shard_by_id(i) == nullptr)
                break;
            ++discovered;
        }
        if (discovered == 0)
            throw std::runtime_error("PPSM: shard_mgr has no shards");

        shards_.reserve(discovered);
        per_shard_max_ = std::max<size_t>(1, max_elements_total_ / discovered);

        for (uint32_t i = 0; i < discovered; ++i)
        {
            Shard *sh = shard_mgr_->get_shard_by_id(i);
            if (!sh)
                break;
            auto s = std::make_unique<ShardState>();
            s->id = i;
            s->arena = sh->get_arena();
            s->map = sh->get_map();
            bool ok = initPerShard(*s, dim_, per_shard_max_, M_, ef_construction_);
            if (!ok)
            {
                std::ostringstream ss;
                ss << "PPSM: failed to init per-shard HNSW for shard " << i;
                throw std::runtime_error(ss.str());
            }
            shards_.push_back(std::move(s));
        }

        startWorkers();
    }

    PPSM::~PPSM()
    {
        stopWorkers();

        for (auto &s : shards_)
        {
            if (s)
            {
                s->pphnsw.reset();
                s->l2space.reset();
            }
        }
    }

    bool PPSM::initPerShard(ShardState &s, size_t dim, size_t per_shard_max, size_t M, size_t ef_construction)
    {
        try
        {
            s.l2space.reset(new hnswlib::L2Space(static_cast<int>(dim)));
            s.pphnsw.reset(new ai::PPHNSW<float>(s.l2space.get(), per_shard_max, M, ef_construction));
            size_t ef_for_search = std::max<size_t>(ef_construction, 64);
            s.pphnsw->setEf(ef_for_search);
            if (s.arena)
                s.pphnsw->setPomaiArena(s.arena);
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

#ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            unsigned int core = state->id % std::max<unsigned>(1, std::thread::hardware_concurrency());
            CPU_SET(core, &cpuset);
            int rc = pthread_setaffinity_np(s->worker.native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0)
                std::cerr << "PPSM: failed to pin shard " << state->id << " worker to core " << core << "\n";
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
                std::unique_lock<std::shared_mutex> idx_lk(sh.index_mu);

                int quant_bits = 8;
                // compute vector length expected by space (underlying payload bytes / sizeof(float))
                size_t underlying_bytes = sh.l2space->get_data_size();
                size_t vec_len = (underlying_bytes >= sizeof(float)) ? (underlying_bytes / sizeof(float)) : 0;

                sh.pphnsw->addQuantizedPoint(task->vec.data(), vec_len, quant_bits, static_cast<hnswlib::labeltype>(task->label), /*replace_deleted=*/task->replace);

                if (sh.map)
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

                {
                    std::lock_guard<std::mutex> gl(sh.label_map_mu);
                    sh.label_to_key[static_cast<uint64_t>(task->label)] = task->key;
                }

                sh.ppe.touch();
                ok = true;
            }
            catch (const std::exception &e)
            {
                std::cerr << "PPSM worker shard " << sh.id << " addVec exception: " << e.what() << "\n";
                ok = false;
            }
            catch (...)
            {
                std::cerr << "PPSM worker shard " << sh.id << " unknown exception\n";
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
            sid = static_cast<uint32_t>(shards_.size()) - 1;
        ShardState &sh = *shards_[sid];

        uint64_t existing_label = 0;
        if (sh.map)
        {
            uint32_t outlen = 0;
            const char *val = sh.map->get(key, static_cast<uint32_t>(klen), &outlen);
            if (val && outlen == sizeof(uint64_t))
                std::memcpy(&existing_label, val, sizeof(existing_label));
        }

        auto t = std::make_unique<Task>();
        t->key.assign(key, key + klen);
        t->vec.assign(vec, vec + dim_);
        t->replace = (existing_label != 0);
        if (t->replace)
            t->label = static_cast<labeltype>(existing_label);
        else
            t->label = static_cast<labeltype>(next_label_global_.fetch_add(1, std::memory_order_relaxed));

        std::future<bool> fut = t->done.get_future();

        {
            std::lock_guard<std::mutex> lg(sh.q_mu);
            sh.q.push_back(std::move(t));
        }
        sh.q_cv.notify_one();

        if (!async_insert_ack_)
        {
            bool ok = false;
            try
            {
                ok = fut.get();
            }
            catch (...)
            {
                ok = false;
            }
            return ok;
        }

        return true;
    }

    bool PPSM::removeKey(const char *key, size_t klen)
    {
        if (!key || klen == 0)
            return false;
        uint32_t sid = computeShard(key, klen);
        if (sid >= shards_.size())
            sid = static_cast<uint32_t>(shards_.size()) - 1;
        ShardState &sh = *shards_[sid];

        bool map_erased = false;
        if (sh.map)
        {
            map_erased = sh.map->erase(std::string(key, key + klen).c_str());
        }

        if (map_erased)
        {
            std::lock_guard<std::mutex> gl(sh.label_map_mu);
            for (auto it = sh.label_to_key.begin(); it != sh.label_to_key.end(); ++it)
            {
                if (it->second.size() == klen && memcmp(it->second.data(), key, klen) == 0)
                {
                    sh.label_to_key.erase(it);
                    break;
                }
            }
        }

        return map_erased;
    }

    std::vector<std::pair<std::string, float>> PPSM::search(const float *query, size_t dim, size_t topk)
    {
        if (!query || dim != dim_ || topk == 0)
            return {};

        std::vector<std::future<std::vector<std::pair<std::string, float>>>> futs;
        futs.reserve(shards_.size());

        for (auto &s : shards_)
        {
            ShardState *shptr = s.get();
            size_t qdim = dim;
            const float *qptr = query;
            futs.push_back(std::async(std::launch::async, [shptr, qptr, qdim, topk]() -> std::vector<std::pair<std::string, float>>
                                      {
            std::vector<std::pair<std::string, float>> out;
            if (!shptr || !shptr->pphnsw)
                return out;
            std::shared_lock<std::shared_mutex> idx_lk(shptr->index_mu);
            try
            {
                std::vector<char> payload(qdim * sizeof(float));
                std::memcpy(payload.data(), reinterpret_cast<const char *>(qptr), qdim * sizeof(float));
                auto pq = shptr->pphnsw->searchKnnAdaptive(payload.data(), topk, 0.0f);
                std::vector<std::pair<std::string, float>> tmp;
                while (!pq.empty())
                {
                    auto pr = pq.top();
                    pq.pop();
                    float dist = pr.first;
                    uint64_t label = static_cast<uint64_t>(pr.second);
                    std::string key;
                    {
                        std::lock_guard<std::mutex> gl(shptr->label_map_mu);
                        auto it = shptr->label_to_key.find(label);
                        if (it != shptr->label_to_key.end())
                            key = it->second;
                        else
                            key = std::to_string(label);
                    }
                    tmp.emplace_back(std::move(key), dist);
                }
                return tmp;
            }
            catch (...)
            {
                return out;
            } }));
        }

        std::vector<std::vector<std::pair<std::string, float>>> parts;
        parts.reserve(futs.size());
        for (auto &f : futs)
        {
            try
            {
                parts.push_back(f.get());
            }
            catch (...)
            {
                parts.emplace_back();
            }
        }

        return merge_results(parts, topk);
    }

    std::vector<std::pair<std::string, float>> PPSM::merge_results(const std::vector<std::vector<std::pair<std::string, float>>> &parts, size_t topk)
    {
        using Item = std::pair<float, std::string>;
        struct Cmp
        {
            bool operator()(Item const &a, Item const &b) const { return a.first < b.first; }
        };

        std::priority_queue<Item, std::vector<Item>, Cmp> heap;

        for (const auto &lst : parts)
        {
            for (const auto &pr : lst)
            {
                float score = pr.second;
                const std::string &key = pr.first;
                if (heap.size() < topk)
                    heap.emplace(score, key);
                else if (score < heap.top().first)
                {
                    heap.pop();
                    heap.emplace(score, key);
                }
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

    size_t PPSM::size() const noexcept
    {
        size_t sum = 0;
        for (const auto &s : shards_)
        {
            if (!s)
                continue;
            // Not exposing PPHNSW size; could be tracked separately.
            if (s->pphnsw)
            {
                sum += s->pphnsw->elementCount();
            }
        }
        return sum;
    }

    // ---------------- Memory usage reporting ----------------
    PPSM::MemoryUsage PPSM::memoryUsage() const noexcept
    {
        MemoryUsage out{};
        try
        {
            uint64_t total_payload = 0;
            uint64_t total_index_overhead = 0;

            for (const auto &s : shards_)
            {
                if (!s)
                    continue;
                if (!s->pphnsw)
                    continue;

                try
                {
                    // Use PPHNSW helper to get element count and seed size
                    size_t cnt = s->pphnsw->elementCount();
                    size_t seed_size = s->pphnsw->getSeedSize();

                    uint64_t payload_bytes = static_cast<uint64_t>(seed_size) * static_cast<uint64_t>(cnt);

                    // estimated total bytes (payload + graph overhead + misc) from PPHNSW
                    size_t estimated_total = s->pphnsw->estimatedMemoryUsageBytes(/*avg_degree_multiplier=*/2);

                    uint64_t index_over = 0;
                    if (estimated_total > payload_bytes)
                        index_over = static_cast<uint64_t>(estimated_total) - payload_bytes;
                    else
                        index_over = 0;

                    total_payload += payload_bytes;
                    total_index_overhead += index_over;
                }
                catch (...)
                {
                    // ignore shard errors and continue
                }
            }

            out.payload_bytes = total_payload;
            out.index_overhead_bytes = total_index_overhead;
            out.total_bytes = total_payload + total_index_overhead;
        }
        catch (...)
        {
            // on unexpected errors return zeros
        }
        return out;
    }

} // namespace pomai::core