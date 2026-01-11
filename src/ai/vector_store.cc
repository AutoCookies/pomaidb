// updated src/ai/vector_store.cc
//
// Phase 2 additions: integrate fingerprint (SimHash) + SoA prefilter path (single-mode).
// - attach_soa() to attach SoA mapping and create fingerprint encoder.
// - on upsert() when a new label is created, write fingerprint into SoA.
// - on search() if SoA+fingerprint available: run prefilter -> refine_topk_l2 -> map labels -> return topk.
//
// This file fixes previous compile errors by including the full VectorStoreSoA
// definition and restoring the chosen_max_elements computation used to set
// max_elements_total_. It also ensures fingerprint encoder is created when SoA is
// attached or when init() runs and SoA already present.

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
#include <thread>
#include <queue>

#include "src/core/config.h" // <- added to read promote_lookahead_ms when starting demoter
#include "src/ai/ppe.h"
#include "src/ai/fingerprint.h"
#include "src/ai/prefilter.h"
#include "src/ai/refine.h"
#include "src/ai/vector_store_soa.h" // need full definition here
#include "src/ai/ids_block.h"
#include "src/ai/atomic_utils.h"

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

        // If soa_ already attached and fingerprint not created yet, create encoder now.
        if (soa_ && !fingerprint_)
        {
            uint32_t bits = (soa_->fingerprint_bits() != 0) ? soa_->fingerprint_bits() : static_cast<uint32_t>(pomai::config::runtime.fingerprint_bits);
            try
            {
                fingerprint_ = FingerprintEncoder::createSimHash(dim_, static_cast<size_t>(bits));
                if (fingerprint_)
                    fingerprint_bytes_ = fingerprint_->bytes();
                else
                    fingerprint_bytes_ = 0;
            }
            catch (...)
            {
                fingerprint_.reset();
                fingerprint_bytes_ = 0;
            }
        }

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
            if (est_ram > chosen_max_elements && est_ram <= chosen_max_elements * 4)
                chosen_max_elements = est_ram;
            chosen_max_elements = std::min<size_t>(chosen_max_elements, 1000000);
        }

        max_elements_total_ = chosen_max_elements;

        // If in sharded mode, treat `max_elements` as total across shards and call sharded init.
        if (sharded_mode_)
        {
            if (!shard_mgr_)
                return false;

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

    void VectorStore::attach_soa(std::unique_ptr<pomai::ai::soa::VectorStoreSoA> soa)
    {
        if (!soa)
            return;
        soa_ = std::move(soa);

        // prefer SoA-specified fingerprint bits; fall back to runtime config default
        uint32_t bits = (soa_->fingerprint_bits() != 0) ? soa_->fingerprint_bits() : static_cast<uint32_t>(pomai::config::runtime.fingerprint_bits);
        if (dim_ != 0 && bits > 0 && !fingerprint_)
        {
            try
            {
                fingerprint_ = FingerprintEncoder::createSimHash(dim_, static_cast<size_t>(bits));
                if (fingerprint_)
                    fingerprint_bytes_ = fingerprint_->bytes();
                else
                    fingerprint_bytes_ = 0;
            }
            catch (...)
            {
                fingerprint_.reset();
                fingerprint_bytes_ = 0;
            }
        }
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
            size_t seed_size = pphnsw_->getSeedSize();
            if (seed_size < sizeof(PPEHeader))
                throw std::runtime_error("VectorStore::upsert: invalid seed size from index");
            size_t payload_bytes = seed_size - sizeof(PPEHeader);

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

                // Phase2: if SoA + fingerprint present, write fingerprint into SoA
                if (soa_ && fingerprint_)
                {
                    try
                    {
                        std::vector<uint8_t> fp(fingerprint_bytes_);
                        fingerprint_->compute(vec, fp.data());
                        // append into SoA (pq_packed not used here)
                        soa_->append_vector(fp.data(), static_cast<uint32_t>(fp.size()), nullptr, 0, new_label);
                    }
                    catch (...)
                    {
                        // non-fatal; continue
                    }
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
            // Delegate remove to PPSM which routes to the correct shard worker.
            // PPSM::removeKey returns whether the operation was accepted.
            return ppsm_->removeKey(key, klen);
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

        // Note: We currently do not remove/update SoA fingerprint entry (left as-is).
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

        // single-mode: if SoA + fingerprint available, prefer prefilter + refine path
        std::vector<std::pair<std::string, float>> out;
        if (!pphnsw_ || !query || dim != dim_ || topk == 0)
            return out;

        // Phase2 prefilter path
        if (soa_ && fingerprint_ && soa_->fingerprint_bits() > 0 && soa_->ids_ptr() != nullptr)
        {
            try
            {
                // compute query fingerprint bytes
                std::vector<uint8_t> qfp(fingerprint_bytes_);
                fingerprint_->compute(query, qfp.data());

                // Build a compacted DB containing only published fingerprints to avoid
                // torn reads during concurrent appends. We use soa_->fingerprint_ptr(i)
                // which checks the per-slot publish flag and returns nullptr for unpublished.
                size_t nv = static_cast<size_t>(soa_->num_vectors());
                std::vector<size_t> published_indices;
                published_indices.reserve(nv);
                std::vector<uint8_t> db_compact;
                db_compact.reserve(nv * fingerprint_bytes_);

                for (size_t i = 0; i < nv; ++i)
                {
                    const uint8_t *p = soa_->fingerprint_ptr(i);
                    if (p)
                    {
                        published_indices.push_back(i);
                        db_compact.insert(db_compact.end(), p, p + fingerprint_bytes_);
                    }
                }

                // If no published fingerprints present, fall back to HNSW search
                if (!published_indices.empty())
                {
                    // choose threshold from config (fall back to sensible default)
                    uint32_t hamming_thresh = static_cast<uint32_t>(pomai::config::runtime.prefilter_hamming_threshold);
                    if (hamming_thresh == 0)
                        hamming_thresh = 128;

                    // collect candidate indices in compacted space
                    std::vector<size_t> compact_candidates;
                    pomai::ai::prefilter::collect_candidates_threshold(qfp.data(), fingerprint_bytes_, db_compact.data(), published_indices.size(), hamming_thresh, compact_candidates);

                    if (!compact_candidates.empty())
                    {
                        // map compacted candidate indices back to original SoA indices
                        std::vector<size_t> candidates;
                        candidates.reserve(compact_candidates.size());
                        for (size_t ci : compact_candidates)
                            candidates.push_back(published_indices[ci]);

                        // refine topk using exact vectors referenced by ids block
                        const uint64_t *ids_block = soa_->ids_ptr();
                        auto refined = pomai::ai::refine::refine_topk_l2(query, dim_, candidates, ids_block, arena_, topk);

                        // refined returns pairs (soaid_idx, distance)
                        out.reserve(refined.size());
                        for (auto &p : refined)
                        {
                            size_t soaid = p.first;
                            float dist = p.second;
                            uint64_t identry = soa_->id_entry_at(soaid);
                            std::string key;
                            {
                                std::lock_guard<std::mutex> lk(label_map_mu_);
                                auto it = label_to_key_.find(identry);
                                if (it != label_to_key_.end())
                                    key = it->second;
                            }
                            if (key.empty() && map_)
                            {
                                // fallback: if identry is numeric label, stringify
                                key = std::to_string(identry);
                            }
                            out.emplace_back(std::move(key), dist);
                            if (out.size() >= topk)
                                break;
                        }
                        return out;
                    }
                }
                // else fallthrough to HNSW search
            }
            catch (const std::exception &e)
            {
                std::cerr << "VectorStore::search prefilter/refine exception: " << e.what() << "\n";
                // fall back to HNSW search
            }
        }

        // Fallback: HNSW search (existing)
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

            pphnsw_->startBackgroundDemoter(interval_ms, lookahead_ns);
        }

        return true;
    }

} // namespace pomai::ai