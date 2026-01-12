/*
 * src/core/pps_manager.cc
 *
 * Pomai Pomegranate Shard Manager (PPSM) Implementation.
 *
 * Major Features:
 * 1. Distributed/Sharded Ingestion.
 * 2. Hybrid Storage (HNSW + SoA).
 * 3. Holographic Search (Scatter-Gather).
 */

#include "src/core/pps_manager.h"
#include "src/ai/holographic_scanner.h"
#include "src/ai/quantize.h"
#include "src/core/seed.h"
#include "src/core/config.h"
#include "src/ai/ids_block.h" // <-- Added: Required for IdEntry usage

#include <algorithm>
#include <future>
#include <iostream>
#include <queue>
#include <sstream>
#include <cstring>
#include <vector>
#include <filesystem>

namespace pomai::core
{

    // Helper to get current time in milliseconds
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
            throw std::invalid_argument("PPSM: shard_mgr is null");

        // Discover shards from the manager
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
        per_shard_max_ = std::max<size_t>(1, max_elements_total_ / discovered);

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

            // Initialize PPHNSW index
            if (!initPerShard(*s, dim_, per_shard_max_, M_, ef_construction_))
            {
                std::ostringstream ss;
                ss << "PPSM: failed to init per-shard HNSW for shard " << i;
                throw std::runtime_error(ss.str());
            }

            // Initialize SoA (Structure of Arrays) storage for Holographic Search
            if (!initSoAPerShard(*s, dim_))
            {
                std::cerr << "PPSM: Warning: Failed to init SoA for shard " << i << ". Holographic search will be disabled for this shard.\n";
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
                s->soa.reset();
            }
        }
    }

    // Initialize PPHNSW (Graph Index) for a specific shard
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

    // Initialize SoA (VectorStoreSoA) for a specific shard
    bool PPSM::initSoAPerShard(ShardState &s, size_t dim)
    {
        // Define path for SoA file: ./data/shard_{id}.soa
        // Ensure directory exists
        std::string data_dir = "./data";
        if (!std::filesystem::exists(data_dir))
        {
            std::filesystem::create_directory(data_dir);
        }
        std::string soa_path = data_dir + "/shard_" + std::to_string(s.id) + ".soa";

        // Configuration for SoA headers
        uint16_t pq_m = 8;    // Default sub-quantizers
        uint16_t pq_k = 256;  // Default centroids per sub
        uint32_t fp_bits = pomai::config::runtime.fingerprint_bits;

        ai::soa::SoaMmapHeader hdr_template{};
        hdr_template.num_vectors = per_shard_max_; // Align with PPHNSW capacity
        hdr_template.dim = static_cast<uint32_t>(dim);
        hdr_template.pq_m = pq_m;
        hdr_template.pq_k = pq_k;
        hdr_template.fingerprint_bits = static_cast<uint16_t>(fp_bits);

        s.soa = std::make_unique<ai::soa::VectorStoreSoA>();
        if (!s.soa->open_or_create(soa_path, hdr_template))
        {
            std::cerr << "PPSM: Failed to open/create SoA file: " << soa_path << "\n";
            return false;
        }

        // Initialize Encoders
        try
        {
            // Fingerprint Encoder (SimHash)
            if (fp_bits > 0)
            {
                s.fp_enc = ai::FingerprintEncoder::createSimHash(dim, fp_bits);
            }

            // Product Quantizer
            s.pq = std::make_unique<ai::ProductQuantizer>(dim, pq_m, pq_k);
            s.pq_packed_bytes = ai::ProductQuantizer::packed4BytesPerVec(pq_m);
        }
        catch (const std::exception &e)
        {
            std::cerr << "PPSM: Error initializing encoders for shard " << s.id << ": " << e.what() << "\n";
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

                // --- 1. Insert into PPHNSW (Graph Index) ---
                size_t expected_bytes = 0;
                if (sh.l2space)
                    expected_bytes = sh.l2space->get_data_size();

                if (expected_bytes == (static_cast<size_t>(dim_) * sizeof(float)))
                {
                    sh.pphnsw->addPoint(task->vec.data(), static_cast<hnswlib::labeltype>(task->label), task->replace);
                }
                else if (expected_bytes == static_cast<size_t>(dim_))
                {
                    sh.pphnsw->addQuantizedPoint(task->vec.data(), dim_, 8, static_cast<hnswlib::labeltype>(task->label), task->replace);
                }
                else if (expected_bytes == static_cast<size_t>((dim_ + 1) / 2))
                {
                    sh.pphnsw->addQuantizedPoint(task->vec.data(), dim_, 4, static_cast<hnswlib::labeltype>(task->label), task->replace);
                }
                else
                {
                    if (sh.pphnsw)
                        sh.pphnsw->addPoint(task->vec.data(), static_cast<hnswlib::labeltype>(task->label), task->replace);
                    else
                        throw std::runtime_error("PPSM worker: unknown index payload layout");
                }

                // --- 2. Update Map (Key-Value Store) ---
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

                // --- 3. Insert into SoA (Structure of Arrays) for Holographic Search ---
                if (sh.soa)
                {
                    std::vector<uint8_t> fp_buf;
                    if (sh.fp_enc)
                    {
                        fp_buf.resize(sh.fp_enc->bytes());
                        sh.fp_enc->compute(task->vec.data(), fp_buf.data());
                    }

                    std::vector<uint8_t> pq_codes;
                    std::vector<uint8_t> pq_packed;
                    const uint8_t *pq_ptr = nullptr;
                    uint32_t pq_len = 0;

                    if (sh.pq)
                    {
                        pq_codes.resize(sh.pq->m());
                        sh.pq->encode(task->vec.data(), pq_codes.data());

                        size_t packed_sz = ai::ProductQuantizer::packed4BytesPerVec(sh.pq->m());
                        pq_packed.resize(packed_sz);
                        ai::ProductQuantizer::pack4From8(pq_codes.data(), pq_packed.data(), sh.pq->m());

                        pq_ptr = pq_packed.data();
                        pq_len = static_cast<uint32_t>(packed_sz);
                    }

                    // Use IdEntry pack helpers (Safe now that ids_block.h is included)
                    uint64_t id_entry = ai::soa::IdEntry::pack_label(task->label);

                    sh.soa->append_vector(
                        fp_buf.empty() ? nullptr : fp_buf.data(),
                        static_cast<uint32_t>(fp_buf.size()),
                        pq_ptr, pq_len,
                        id_entry);
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

    // Standard Search: Uses PPHNSW (Graph Index)
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

    // Holographic Search: Uses SoA Scan (SimHash + PQ)
    std::vector<std::pair<std::string, float>> PPSM::searchHolographic(const float *query, size_t dim, size_t topk)
    {
        if (!query || dim != dim_ || topk == 0)
            return {};

        std::vector<std::future<std::vector<ai::HolographicScanner::ScanResult>>> futs;
        futs.reserve(shards_.size());

        for (auto &s : shards_)
        {
            ShardState *shptr = s.get();
            futs.push_back(std::async(std::launch::async, [shptr, query, topk]()
                                      {
                if (!shptr->soa) return std::vector<ai::HolographicScanner::ScanResult>{};

                std::shared_lock<std::shared_mutex> idx_lk(shptr->index_mu);
                
                return ai::HolographicScanner::scan_shard(
                    shptr->soa.get(),
                    shptr->fp_enc.get(),
                    shptr->pq.get(),
                    query,
                    topk
                ); }));
        }

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
                auto shard_results = futs[i].get();
                ShardState *shptr = shards_[i].get();

                for (const auto &res : shard_results)
                {
                    std::string key;
                    if (ai::soa::IdEntry::is_label(res.id_entry))
                    {
                        uint64_t lbl = ai::soa::IdEntry::unpack_label(res.id_entry);
                        {
                            std::lock_guard<std::mutex> gl(shptr->label_map_mu);
                            auto it = shptr->label_to_key.find(lbl);
                            if (it != shptr->label_to_key.end())
                                key = it->second;
                            else
                                key = std::to_string(lbl);
                        }
                    }
                    else
                    {
                        key = std::to_string(res.id_entry);
                    }

                    float score = res.score;
                    if (heap.size() < topk)
                    {
                        heap.emplace(score, key);
                    }
                    else if (score < heap.top().first)
                    {
                        heap.pop();
                        heap.emplace(score, key);
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
            if (s->pphnsw)
            {
                sum += s->pphnsw->elementCount();
            }
        }
        return sum;
    }

    PPSM::MemoryUsage PPSM::memoryUsage() const noexcept
    {
        MemoryUsage out{};
        try
        {
            uint64_t total_payload = 0;
            uint64_t total_index_overhead = 0;

            for (const auto &s : shards_)
            {
                if (!s || !s->pphnsw)
                    continue;

                try
                {
                    size_t cnt = s->pphnsw->elementCount();
                    size_t seed_size = s->pphnsw->getSeedSize();

                    uint64_t payload_bytes = static_cast<uint64_t>(seed_size) * static_cast<uint64_t>(cnt);
                    size_t estimated_total = s->pphnsw->estimatedMemoryUsageBytes(/*avg_degree_multiplier=*/2);

                    uint64_t index_over = 0;
                    if (estimated_total > payload_bytes)
                        index_over = static_cast<uint64_t>(estimated_total) - payload_bytes;

                    total_payload += payload_bytes;
                    total_index_overhead += index_over;
                }
                catch (...)
                {
                }
            }

            out.payload_bytes = total_payload;
            out.index_overhead_bytes = total_index_overhead;
            out.total_bytes = total_payload + total_index_overhead;
        }
        catch (...)
        {
        }
        return out;
    }

} // namespace pomai::core