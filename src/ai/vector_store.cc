// updated src/ai/vector_store.cc
//
// Added automatic capacity estimation (estimate_max_elements) and conservative
// defaults (M <= 8, ef_construction <= 50) when initializing VectorStore.
// If an arena is available we account for storing payloads indirectly which
// drastically reduces per-element memory footprint.
//
// Also added memoryUsage() implementation to report estimated memory usage.
// Wired PPHNSW background demoter to use config.promote_lookahead_ms.

#include "src/ai/vector_store.h"

#include <cstring>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <random>
#include <future>
#include <numeric>
#include <sys/sysinfo.h> // get_free_ram_bytes
#include <unistd.h>
#include <sstream>

#include "src/core/config.h" // <- added to read promote_lookahead_ms when starting demoter
#include "src/ai/ppe.h"

using namespace pomai::memory;

namespace pomai::ai
{

    // Return approximate free RAM available on the system in bytes.
    // Uses sysinfo.freeram * mem_unit which is the conventional Linux approach.
    static inline uint64_t get_free_ram_bytes()
    {
        struct sysinfo s;
        if (sysinfo(&s) != 0)
            return 0;
        return static_cast<uint64_t>(s.freeram) * static_cast<uint64_t>(s.mem_unit);
    }

    // Estimate a safe maximum number of elements that can be held in memory given
    // the available bytes. The estimate is conservative and intended for a single
    // process instance. If payload_in_arena is true we assume the arena stores
    // raw float payloads and the index stores only an 8-byte offset.
    static inline size_t estimate_max_elements(uint64_t available_bytes,
                                               bool payload_in_arena,
                                               size_t dim,
                                               size_t M_param,
                                               double safety_frac = 0.60)
    {
        if (available_bytes == 0 || safety_frac <= 0.0 || safety_frac >= 1.0 || dim == 0)
            return 0;

        // payload stored in index (if not stored in arena) or offset if in arena
        size_t payload_in_index = payload_in_arena ? sizeof(uint64_t) : (dim * sizeof(float));

        // Approximate PPE/in-memory header overhead (atomics/flags/alignment)
        size_t ppe_overhead = 64; // conservative

        // approximate neighbors/storage overhead: average degree ~ 2*M
        size_t avg_degree = std::max<size_t>(1, M_param * 2);
        size_t neighbor_bytes = avg_degree * sizeof(int); // store neighbors as ints/tableint

        // extra miscellaneous overhead per element
        size_t misc = 32;

        size_t per_elem = payload_in_index + ppe_overhead + neighbor_bytes + misc;
        if (per_elem == 0)
            return 0;

        uint64_t budget = static_cast<uint64_t>(available_bytes * safety_frac);
        size_t max_elems = static_cast<size_t>(budget / per_elem);
        return max_elems;
    }

    // ---------------- VectorStore public ----------------

    VectorStore::VectorStore() = default;

    VectorStore::~VectorStore()
    {
        // If we are single-mode and have PPPQ demoter, stop it.
        ppq_demote_running_.store(false, std::memory_order_release);
        if (ppq_demote_thread_.joinable())
            ppq_demote_thread_.join();
    }

    bool VectorStore::init(size_t dim, size_t max_elements, size_t M, size_t ef_construction, PomaiArena *arena)
    {
        // normalize parameters: prefer conservative defaults for memory-constrained machines
        if (M == 0)
            M = 8;
        if (ef_construction == 0)
            ef_construction = 50;

        // Enforce upper bounds to keep memory small by default.
        if (M > 8)
            M = 8;
        if (ef_construction > 200)
            ef_construction = std::min<size_t>(ef_construction, 200);
        // But prefer ef_construction <= 50 for construction budget
        if (ef_construction > 50)
            ef_construction = 50;

        dim_ = dim;
        M_ = M;
        ef_construction_ = ef_construction;

        // If caller passed max_elements==0 or a small value, compute a conservative
        // estimate using free RAM and (if available) arena capacity.
        size_t chosen_max_elements = max_elements;
        uint64_t free_ram = get_free_ram_bytes();
        bool has_arena = (arena != nullptr);

        // Compute estimate based on RAM
        size_t est_ram = estimate_max_elements(free_ram, has_arena, dim_, M_, 0.60);

        // If arena present, compute how many vectors it can hold (based on packed blob size)
        size_t est_arena = 0;
        if (arena)
        {
            uint64_t cap = arena->get_capacity_bytes();
            if (cap > 0)
            {
                size_t per_blob = sizeof(uint32_t) + dim_ * sizeof(float); // header + payload
                if (per_blob > 0)
                    est_arena = static_cast<size_t>(cap / per_blob);
            }
        }

        // Choose a conservative maximum:
        // - If caller provided a value > 0, allow it but warn and possibly bump if it's too small.
        // - If caller provided 0, pick from estimates; prefer est_ram but cap by est_arena if smaller.
        if (chosen_max_elements == 0)
        {
            size_t pick = 16384; // baseline floor
            if (est_ram > pick)
                pick = est_ram;
            if (est_arena > 0 && est_arena < pick)
                pick = est_arena;
            // clamp to reasonable upper bound
            pick = std::min<size_t>(pick, 1000000);
            chosen_max_elements = pick;
        }
        else
        {
            // if caller value is smaller than RAM estimate, we may bump it up slightly to better utilize memory
            if (est_ram > chosen_max_elements && est_ram <= chosen_max_elements * 4)
                chosen_max_elements = est_ram;
            // but never exceed 1M by default
            chosen_max_elements = std::min<size_t>(chosen_max_elements, 1000000);
        }

        max_elements_total_ = chosen_max_elements;

        // If in sharded mode, treat `max_elements` as total across shards and call sharded init.
        if (sharded_mode_)
        {
            if (!shard_mgr_)
                return false;

            // create PPSM instance if not already present. PPSM expects total capacity.
            try
            {
                if (!ppsm_)
                    ppsm_.reset(new pomai::core::PPSM(shard_mgr_, dim_, max_elements_total_, M_, ef_construction_, /*async_insert_ack=*/true));
            }
            catch (const std::exception &e)
            {
                std::cerr << "VectorStore::init: failed to create PPSM: " << e.what() << "\n";
                return false;
            }
            return true;
        }

        // single-mode initialization
        return init_single(dim_, max_elements_total_, M_, ef_construction, arena);
    }

    void VectorStore::attach_map(PomaiMap *map)
    {
        // single-map attach
        map_ = map;
        if (pphnsw_)
            pphnsw_->setPomaiArena(map ? map->get_arena() : nullptr);
    }

    void VectorStore::attach_shard_manager(ShardManager *mgr, uint32_t shard_count)
    {
        if (!mgr || shard_count == 0)
            throw std::invalid_argument("attach_shard_manager: invalid args");

        shard_mgr_ = mgr;
        shard_count_ = shard_count;
        sharded_mode_ = true;

        // create PPSM immediately if init() already ran
        if (dim_ != 0 && max_elements_total_ != 0 && !ppsm_)
        {
            ppsm_.reset(new pomai::core::PPSM(shard_mgr_, dim_, max_elements_total_, M_, ef_construction_, /*async=*/true));
        }
    }

    bool VectorStore::enable_ivf(size_t num_clusters, size_t m_sub, size_t nbits, uint64_t seed)
    {
        if (sharded_mode_)
        {
            // Not supporting IVFPQ across PPSM in this simple wrapper.
            return false;
        }
        if (dim_ == 0)
            return false;
        ivf_.reset(new PPIVF(dim_, num_clusters, m_sub, nbits));
        if (!ivf_->init_random_seed(seed))
        {
            ivf_.reset();
            return false;
        }
        ivf_enabled_ = true;
        return true;
    }

    bool VectorStore::upsert(const char *key, size_t klen, const float *vec)
    {
        if (sharded_mode_)
        {
            if (!ppsm_ || !shard_mgr_)
                return false;
            // route to PPSM. If PPSM rejects (full/backpressure) caller gets false.
            return ppsm_->addVec(key, klen, vec);
        }

        // Single-mode path
        std::unique_lock<std::shared_mutex> write_lock(rw_mu_);

        if (!pphnsw_)
            return false;
        if (!key || klen == 0 || !vec)
            return false;

        uint64_t label = 0;
        {
            std::lock_guard<std::mutex> lk(label_map_mu_);
            std::string skey(key, key + klen);
            auto it = key_to_label_.find(skey);
            if (it != key_to_label_.end())
                label = it->second;
        }
        if (label == 0 && map_)
        {
            label = read_label_from_map(key, klen);
            if (label != 0)
            {
                std::lock_guard<std::mutex> lk(label_map_mu_);
                std::string skey(key, key + klen);
                key_to_label_[skey] = label;
                label_to_key_[label] = skey;
            }
        }

        try
        {
            // Determine what payload layout the index expects:
            // seed_size = sizeof(PPEHeader) + underlying_payload_bytes
            size_t seed_size = pphnsw_->getSeedSize();
            if (seed_size < sizeof(PPEHeader))
                throw std::runtime_error("VectorStore::upsert: invalid seed size from index");
            size_t payload_bytes = seed_size - sizeof(PPEHeader);

            // Decide how to insert:
            // - If index expects full float payload: payload_bytes == dim * sizeof(float) => call addPoint
            // - If index expects 8-bit quantized payload: payload_bytes == dim => call addQuantizedPoint(..., bits=8)
            // - If index expects packed-4 payload: payload_bytes == (dim+1)/2 => call addQuantizedPoint(..., bits=4)
            // Otherwise error.

            // Helper lambda to store label mapping & optional WAL on successful insert
            auto finalize_label_after_insert = [&](uint64_t new_label) -> bool
            {
                if (map_)
                {
                    bool ok = store_label_in_map(new_label, key, klen);
                    if (!ok)
                    {
                        try
                        {
                            pphnsw_->markDelete(static_cast<hnswlib::labeltype>(new_label));
                        }
                        catch (...)
                        {
                        }
                        return false;
                    }
                }
                {
                    std::lock_guard<std::mutex> lk(label_map_mu_);
                    std::string skey(key, key + klen);
                    key_to_label_[skey] = new_label;
                    label_to_key_[new_label] = skey;
                }
                return true;
            };

            // Insert path
            if (payload_bytes == dim_ * sizeof(float))
            {
                // index expects full float payloads
                if (label != 0)
                {
                    pphnsw_->addPoint(vec, static_cast<hnswlib::labeltype>(label), /*replace_deleted=*/true);
                }
                else
                {
                    uint64_t new_label = next_label_.fetch_add(1, std::memory_order_relaxed);
                    pphnsw_->addPoint(vec, static_cast<hnswlib::labeltype>(new_label), /*replace_deleted=*/false);
                    if (!finalize_label_after_insert(new_label))
                        return false;
                    label = new_label;
                }
            }
            else if (payload_bytes == dim_)
            {
                // index expects 8-bit quantized payload (one byte per dim)
                const int bits = 8;
                if (label != 0)
                {
                    pphnsw_->addQuantizedPoint(vec, dim_, bits, static_cast<hnswlib::labeltype>(label), /*replace_deleted=*/true);
                }
                else
                {
                    uint64_t new_label = next_label_.fetch_add(1, std::memory_order_relaxed);
                    pphnsw_->addQuantizedPoint(vec, dim_, bits, static_cast<hnswlib::labeltype>(new_label), /*replace_deleted=*/false);
                    if (!finalize_label_after_insert(new_label))
                        return false;
                    label = new_label;
                }
            }
            else if (payload_bytes == ((dim_ + 1) / 2))
            {
                // index expects packed 4-bit payload
                const int bits = 4;
                if (label != 0)
                {
                    pphnsw_->addQuantizedPoint(vec, dim_, bits, static_cast<hnswlib::labeltype>(label), /*replace_deleted=*/true);
                }
                else
                {
                    uint64_t new_label = next_label_.fetch_add(1, std::memory_order_relaxed);
                    pphnsw_->addQuantizedPoint(vec, dim_, bits, static_cast<hnswlib::labeltype>(new_label), /*replace_deleted=*/false);
                    if (!finalize_label_after_insert(new_label))
                        return false;
                    label = new_label;
                }
            }
            else
            {
                // Unsupported payload layout
                std::ostringstream ss;
                ss << "addPoint/addQuantizedPoint mismatch: index payload_bytes=" << payload_bytes
                   << " dim=" << dim_ << " (expected float bytes=" << (dim_ * sizeof(float))
                   << " or quant8 bytes=" << dim_ << " or packed4 bytes=" << ((dim_ + 1) / 2) << ")";
                throw std::runtime_error(ss.str());
            }

            // optional IVFPQ registration (unchanged)
            if (ivf_enabled_ && ivf_)
            {
                int cl = ivf_->assign_cluster(vec);
                const uint8_t *code = ivf_->encode_pq(vec);
                ivf_->add_label(label, cl, code);
            }

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore::upsert exception: " << e.what() << "\n";
            return false;
        }
    }

    bool VectorStore::remove(const char *key, size_t klen)
    {
        if (sharded_mode_)
        {
            if (!ppsm_)
                return false;
            // Best-effort: remove from shared map and let PPSM worker observe it (or provide API to remove by label)
            if (!map_)
                return false;
            return map_->erase(std::string(key, key + klen).c_str());
        }

        std::unique_lock<std::shared_mutex> write_lock(rw_mu_);

        if (!map_)
            return false;
        if (!key || klen == 0)
            return false;

        uint64_t label = 0;
        {
            std::lock_guard<std::mutex> lk(label_map_mu_);
            std::string skey(key, key + klen);
            auto it = key_to_label_.find(skey);
            if (it != key_to_label_.end())
            {
                label = it->second;
                key_to_label_.erase(it);
                label_to_key_.erase(label);
            }
        }
        if (label == 0)
        {
            label = read_label_from_map(key, klen);
        }

        bool map_erased = map_->erase(std::string(key, key + klen).c_str());

        if (label != 0 && pphnsw_)
        {
            try
            {
                pphnsw_->markDelete(static_cast<hnswlib::labeltype>(label));
            }
            catch (...)
            {
            }
        }

        return map_erased;
    }

    std::vector<std::pair<std::string, float>> VectorStore::search(const float *query, size_t dim, size_t topk)
    {
        if (sharded_mode_)
        {
            if (!ppsm_)
                return {};
            return ppsm_->search(query, dim, topk);
        }

        // single-mode search (unchanged)
        std::vector<std::pair<std::string, float>> out;
        if (!pphnsw_ || !query || dim != dim_ || topk == 0)
            return out;

        try
        {
            auto buf = build_seed_buffer(query);
            auto pq = pphnsw_->searchKnnAdaptive(buf.get(), topk, 0.0f);
            std::vector<std::pair<std::string, float>> tmp;
            tmp.reserve(pq.size());
            while (!pq.empty())
            {
                auto pr = pq.top();
                pq.pop();
                float dist = pr.first;
                uint64_t label = static_cast<uint64_t>(pr.second);

                std::string key;
                {
                    std::lock_guard<std::mutex> lk(label_map_mu_);
                    auto it = label_to_key_.find(label);
                    if (it != label_to_key_.end())
                        key = it->second;
                }
                if (key.empty() && map_)
                    key = std::to_string(label);
                tmp.emplace_back(std::move(key), dist);
            }
            std::reverse(tmp.begin(), tmp.end());
            out = std::move(tmp);
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore::search exception: " << e.what() << "\n";
        }

        return out;
    }

    size_t VectorStore::size() const
    {
        if (sharded_mode_)
        {
            if (ppsm_)
                return ppsm_->size();
            return 0;
        }
        std::lock_guard<std::mutex> lk(label_map_mu_);
        return label_to_key_.size();
    }

    // ---------------- Memory usage reporting ----------------
    VectorStore::MemoryUsage VectorStore::memoryUsage() const noexcept
    {
        MemoryUsage out{};
        try
        {
            if (sharded_mode_)
            {
                // delegate to PPSM if available
                if (ppsm_)
                {
                    auto mu = ppsm_->memoryUsage();
                    out.payload_bytes = mu.payload_bytes;
                    out.index_overhead_bytes = mu.index_overhead_bytes;
                    out.total_bytes = mu.total_bytes;
                    return out;
                }

                // If sharded_mode_ but no PPSM (unlikely), aggregate per-shard stores if present
                uint64_t total_payload = 0;
                uint64_t total_over = 0;
                for (const auto &s : per_shard_stores_)
                {
                    if (!s)
                        continue;
                    auto mu = s->memoryUsage();
                    total_payload += mu.payload_bytes;
                    total_over += mu.index_overhead_bytes;
                }
                out.payload_bytes = total_payload;
                out.index_overhead_bytes = total_over;
                out.total_bytes = total_payload + total_over;
                return out;
            }

            // single-mode: use pphnsw_ estimates
            if (pphnsw_)
            {
                size_t cnt = pphnsw_->elementCount();
                size_t seed_size = pphnsw_->getSeedSize();
                uint64_t payload_bytes = static_cast<uint64_t>(seed_size) * static_cast<uint64_t>(cnt);

                size_t estimated_total = pphnsw_->estimatedMemoryUsageBytes(/*avg_degree_multiplier=*/2);
                uint64_t index_over = 0;
                if (estimated_total > payload_bytes)
                    index_over = static_cast<uint64_t>(estimated_total) - payload_bytes;
                else
                    index_over = 0;

                out.payload_bytes = payload_bytes;
                out.index_overhead_bytes = index_over;
                out.total_bytes = payload_bytes + index_over;
                return out;
            }

            // nothing initialized
            return out;
        }
        catch (...)
        {
            return MemoryUsage{};
        }
    }

    // ---------------- internal helpers ----------------

    bool VectorStore::init_single(size_t dim, size_t max_elements, size_t M, size_t ef_construction, PomaiArena *arena)
    {
        if (dim == 0 || max_elements == 0)
            return false;

        arena_ = arena;

        // create underlying L2 space for HNSW
        l2space_.reset(new hnswlib::L2Space(static_cast<int>(dim)));

        try
        {
            pphnsw_.reset(new PPHNSW<float>(l2space_.get(), max_elements, M, ef_construction));
            // set a reasonable search ef
            size_t ef_for_search = std::max<size_t>(ef_construction, 64);
            pphnsw_->setEf(ef_for_search);
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore: failed to create PPHNSW: " << e.what() << "\n";
            return false;
        }

        if (arena_)
            pphnsw_->setPomaiArena(arena_);

        //
        // PPPQ: create, train quickly on random samples and attach to PPHNSW.
        //
        try
        {
            size_t pq_m = 8;                 // number of subquantizers
            size_t pq_k = 256;               // codebook size per sub
            size_t max_elems = max_elements; // must be >= max labels used

            while (pq_m > 1 && (dim % pq_m) != 0)
            {
                pq_m /= 2; // Try 8→4→2→1 until it divides dim
            }

            if (pq_m < 1 || (dim % pq_m) != 0)
            {
                std::clog << "VectorStore: Skipping PPPQ (dim=" << dim << ")\n";
            }

            auto ppq = std::make_unique<pomai::ai::PPPQ>(dim, pq_m, pq_k, max_elems, "pppq_codes.mmap");

            // Quick synthetic training (replace with dataset samples for production)
            size_t n_train = std::min<size_t>(20000, max_elems);
            std::vector<float> samples;
            samples.resize(n_train * dim);
            std::mt19937_64 rng(123456);
            std::uniform_real_distribution<float> ud(0.0f, 1.0f);
            for (size_t i = 0; i < n_train * dim; ++i)
                samples[i] = ud(rng);

            ppq->train(samples.data(), n_train, 10);

            // Attach PPPQ into PPHNSW
            pphnsw_->setPPPQ(std::move(ppq));

            // Start background demoter that periodically calls PPPQ::purgeCold
            if (!ppq_demote_running_.load(std::memory_order_acquire))
            {
                ppq_demote_running_.store(true, std::memory_order_release);
                ppq_demote_thread_ = std::thread([this]()
                                                 {
                    auto *ppq = pphnsw_->getPPPQ();
                    if (!ppq) return;
                    while (ppq_demote_running_.load(std::memory_order_acquire))
                    {
                        try
                        {
                            ppq->purgeCold(ppq_demote_cold_thresh_ms_);
                        }
                        catch (...)
                        {
                            // suppressed PPPQ demoter exception output as requested
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(ppq_demote_interval_ms_));
                    } });
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore: PPPQ init/train failed: " << e.what() << "\n";
            // non-fatal: continue without PPPQ
        }

        // Start PPHNSW background demoter with lookahead converted to ns from config.milliseconds
        {
            uint64_t lookahead_ms = pomai::config::runtime.promote_lookahead_ms;
            uint64_t lookahead_ns = lookahead_ms * 1000000ULL;
            // reasonable interval (1s)
            uint64_t interval_ms = 1000;

            //------------------------TEST SAU KHI DEBUG XONG THÌ BỎ COMMENT VÌ ĐÂY LÀ BASE------------------------------------
            pphnsw_->startBackgroundDemoter(interval_ms, lookahead_ns);
            //------------------------TEST SAU KHI DEBUG XONG THÌ BỎ COMMENT VÌ ĐÂY LÀ BASE------------------------------------
        }

        return true;
    }

    bool VectorStore::init_sharded(size_t dim, size_t max_elements_total, size_t M, size_t ef_construction, ShardManager *mgr)
    {
        if (!mgr || shard_count_ == 0)
            return false;
        // create per-shard VectorStore instances and initialize each with shard-specific arena/map
        per_shard_stores_.clear();
        per_shard_stores_.resize(shard_count_);

        // split max_elements evenly (at least 1)
        size_t base = std::max<size_t>(1, max_elements_total / shard_count_);
        for (uint32_t i = 0; i < shard_count_; ++i)
        {
            Shard *s = mgr->get_shard_by_id(i);
            if (!s)
                return false;

            auto store = std::make_unique<VectorStore>();
            // initialize per-shard as single-mode instances: pass per-shard arena and per-shard max_elements
            bool ok = store->init(dim, base, M, ef_construction, s->get_arena());
            if (!ok)
                return false;
            store->attach_map(s->get_map());
            per_shard_stores_[i] = std::move(store);
        }
        return true;
    }

    std::unique_ptr<char[]> VectorStore::build_seed_buffer(const float *vec) const
    {
        size_t payload_size = dim_ * sizeof(float);
        std::unique_ptr<char[]> buf(new char[payload_size]);
        std::memcpy(buf.get(), reinterpret_cast<const char *>(vec), payload_size);
        return buf;
    }

    bool VectorStore::store_label_in_map(uint64_t label, const char *key, size_t klen)
    {
        if (!map_)
            return false;
        uint64_t le = label;
        const char *vptr = reinterpret_cast<const char *>(&le);
        bool ok = map_->put(key, static_cast<uint32_t>(klen), vptr, static_cast<uint32_t>(sizeof(le)));
        if (!ok)
            return false;
        Seed *s = map_->find_seed(key, static_cast<uint32_t>(klen));
        if (s)
            s->type = Seed::OBJ_VECTOR;
        return true;
    }

    uint64_t VectorStore::read_label_from_map(const char *key, size_t klen) const
    {
        if (!map_)
            return 0;
        uint32_t outlen = 0;
        const char *val = map_->get(key, static_cast<uint32_t>(klen), &outlen);
        if (!val || outlen != sizeof(uint64_t))
            return 0;
        uint64_t label = 0;
        std::memcpy(&label, val, sizeof(label));
        return label;
    }

    // Merge k-sorted-ish lists (not strictly sorted) by score and pick topk smallest distances.
    std::vector<std::pair<std::string, float>> VectorStore::merge_topk(const std::vector<std::vector<std::pair<std::string, float>>> &lists, size_t topk)
    {
        // Use a max-heap of current best topk to keep final topk smallest
        using Item = std::pair<float, std::pair<std::string, size_t>>; // (score, (key, src_index))
        struct Cmp
        {
            bool operator()(Item const &a, Item const &b) const { return a.first < b.first; }
        }; // max-heap
        std::priority_queue<Item, std::vector<Item>, Cmp> heap;

        for (size_t i = 0; i < lists.size(); ++i)
        {
            const auto &lst = lists[i];
            for (const auto &pr : lst)
            {
                float score = pr.second;
                if (heap.size() < topk)
                {
                    heap.push({score, {pr.first, i}});
                }
                else if (score < heap.top().first)
                {
                    heap.pop();
                    heap.push({score, {pr.first, i}});
                }
            }
        }

        std::vector<std::pair<std::string, float>> out;
        out.reserve(heap.size());
        while (!heap.empty())
        {
            auto it = heap.top();
            heap.pop();
            out.emplace_back(it.second.first, it.first);
        }
        std::reverse(out.begin(), out.end());
        return out;
    }
} // namespace pomai::ai