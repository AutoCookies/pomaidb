/*
 * src/ai/vector_store.cc
 *
 * VectorStore with fingerprint + PQ integration (Phase 2/3).
 * - fingerprints: SimHash-based bitpacked stored in SoA
 * - PQ: ProductQuantizer trained at init_single, encodes on upsert and stores packed4 codes in SoA
 * - Search: prefilter (Hamming) -> PQ approximate eval on packed4/raw8 -> refine exact top-k
 *
 * Notes:
 * - This file keeps previous VectorStore semantics for fallback HNSW search.
 * - Atomic helpers (atomic_utils) are used when reading IDs / flags from SoA.
 */

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

#include "src/core/config.h" // config runtime parameters
#include "src/ai/ppe.h"
#include "src/ai/fingerprint.h"
#include "src/ai/prefilter.h"
#include "src/ai/pq.h"
#include "src/ai/pq_eval.h"
#include "src/ai/refine.h"
#include "src/ai/vector_store_soa.h" // need full definition here
#include "src/ai/ids_block.h"
#include "src/ai/atomic_utils.h"
#include "src/ai/pppq.h" // PPPQ integration
#include <functional>

using namespace pomai::memory;

namespace pomai::ai
{

    // Return approximate free RAM available on the system in bytes.
    static inline uint64_t get_free_ram_bytes()
    {
        struct sysinfo s;
        if (sysinfo(&s) != 0)
            return 0;
        return static_cast<uint64_t>(s.freeram) * static_cast<uint64_t>(s.mem_unit);
    }

    // Conservative capacity estimate helper (unchanged)
    static inline size_t estimate_max_elements(uint64_t available_bytes,
                                               bool payload_in_arena,
                                               size_t dim,
                                               size_t M_param,
                                               double safety_frac = 0.60)
    {
        if (available_bytes == 0 || safety_frac <= 0.0 || safety_frac >= 1.0 || dim == 0)
            return 0;
        size_t payload_in_index = payload_in_arena ? sizeof(uint64_t) : (dim * sizeof(float));
        size_t ppe_overhead = 64;
        size_t avg_degree = std::max<size_t>(1, M_param * 2);
        size_t neighbor_bytes = avg_degree * sizeof(int);
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
        ppq_demote_running_.store(false, std::memory_order_release);
        if (ppq_demote_thread_.joinable())
            ppq_demote_thread_.join();
    }

    bool VectorStore::init(size_t dim, size_t max_elements, size_t M, size_t ef_construction, PomaiArena *arena)
    {
        if (M == 0)
            M = 8;
        if (ef_construction == 0)
            ef_construction = 50;
        if (M > 8)
            M = 8;
        if (ef_construction > 200)
            ef_construction = std::min<size_t>(ef_construction, 200);
        if (ef_construction > 50)
            ef_construction = 50;

        dim_ = dim;
        M_ = M;
        ef_construction_ = ef_construction;

        // If SoA already attached create fingerprint encoder now (will be used when upserting)
        if (soa_ && !fingerprint_)
        {
            uint32_t bits = (soa_->fingerprint_bits() != 0) ? soa_->fingerprint_bits() : static_cast<uint32_t>(pomai::config::runtime.fingerprint_bits);
            try
            {
                fingerprint_ = FingerprintEncoder::createSimHash(dim_, static_cast<size_t>(bits));
                fingerprint_bytes_ = fingerprint_ ? fingerprint_->bytes() : 0;
            }
            catch (...)
            {
                fingerprint_.reset();
                fingerprint_bytes_ = 0;
            }
        }

        size_t chosen_max_elements = max_elements;
        uint64_t free_ram = get_free_ram_bytes();
        bool has_arena = (arena != nullptr);
        size_t est_ram = estimate_max_elements(free_ram, has_arena, dim_, M_, 0.60);
        size_t est_arena = 0;
        if (arena)
        {
            uint64_t cap = arena->get_capacity_bytes();
            if (cap > 0)
            {
                size_t per_blob = sizeof(uint32_t) + dim_ * sizeof(float);
                if (per_blob > 0)
                    est_arena = static_cast<size_t>(cap / per_blob);
            }
        }

        if (chosen_max_elements == 0)
        {
            size_t pick = 16384;
            if (est_ram > pick)
                pick = est_ram;
            if (est_arena > 0 && est_arena < pick)
                pick = est_arena;
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

        // PQ compatibility / load codebooks from SoA if present
        if (pq_ && soa_->pq_m() != 0)
        {
            if (soa_->pq_m() == static_cast<uint16_t>(pq_->m()))
            {
                pq_packed_bytes_ = ProductQuantizer::packed4BytesPerVec(pq_->m());
            }
            else
            {
                std::cerr << "[VectorStore] SoA PQ layout mismatch (soa.pq_m=" << soa_->pq_m()
                          << " trained.m=" << pq_->m() << ") -> disabling PQ\n";
                pq_.reset();
                pq_packed_bytes_ = 0;
            }
        }
        else
        {
            // If PQ not present but SoA contains embedded codebooks, load them into pq_
            if (!pq_ && soa_->pq_m() != 0 && soa_->pq_k() != 0 && soa_->codebooks_size_bytes() > 0)
            {
                if (dim_ != 0 && dim_ == soa_->dim())
                {
                    uint16_t m = soa_->pq_m();
                    uint16_t k = soa_->pq_k();
                    try
                    {
                        pq_.reset(new ProductQuantizer(dim_, static_cast<size_t>(m), static_cast<size_t>(k)));
                        const float *cb = soa_->codebooks_ptr();
                        size_t floats = soa_->codebooks_size_bytes() / sizeof(float);
                        if (!pq_->load_codebooks_from_buffer(cb, floats))
                        {
                            std::cerr << "[VectorStore] failed to load codebooks from SoA into PQ\n";
                            pq_.reset();
                            pq_packed_bytes_ = 0;
                        }
                        else
                        {
                            pq_packed_bytes_ = ProductQuantizer::packed4BytesPerVec(pq_->m());
                            std::cerr << "[VectorStore] loaded PQ codebooks from SoA m=" << pq_->m() << " k=" << pq_->k() << "\n";
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "[VectorStore] attach_soa: failed to create PQ: " << e.what() << "\n";
                        pq_.reset();
                        pq_packed_bytes_ = 0;
                    }
                }
                else
                {
                    // dim mismatch or init not run yet; PQ will be loaded later from init when dim_ is known.
                }
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
            size_t seed_size = pphnsw_->getSeedSize();
            if (seed_size < sizeof(PPEHeader))
                throw std::runtime_error("VectorStore::upsert: invalid seed size from index");
            size_t payload_bytes = seed_size - sizeof(PPEHeader);

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

                // Prepare fingerprint (if configured)
                const uint8_t *fp_ptr = nullptr;
                uint32_t fp_len = 0;
                std::vector<uint8_t> fpbuf;
                if (fingerprint_)
                {
                    fpbuf.resize(fingerprint_bytes_);
                    fingerprint_->compute(vec, fpbuf.data());
                    fp_ptr = fpbuf.data();
                    fp_len = static_cast<uint32_t>(fpbuf.size());
                }

                // Prepare PQ codes / packed representation (if configured)
                const uint8_t *pq_ptr = nullptr;
                uint32_t pq_len = 0;
                std::vector<uint8_t> pq_packed;
                std::vector<uint8_t> codes;
                if (pq_)
                {
                    // encode into raw 8-bit codes
                    codes.resize(pq_->m());
                    pq_->encode(vec, codes.data());

                    // Decide storage form:
                    bool need_raw = (pq_->k() > 16);
                    bool soa_has_raw = (soa_ && soa_->pq_codes_ptr(0) != nullptr);
                    bool soa_has_packed = (soa_ && soa_->pq_packed_ptr(0) != nullptr);

                    if (need_raw && soa_has_raw)
                    {
                        // SoA has raw 8-bit codes storage; store raw codes
                        pq_ptr = codes.data();
                        pq_len = static_cast<uint32_t>(codes.size());
                    }
                    else if (!need_raw && soa_has_packed)
                    {
                        // k <= 16: it's safe to pack to 4-bit and store
                        pq_packed.resize(ProductQuantizer::packed4BytesPerVec(pq_->m()));
                        ProductQuantizer::pack4From8(codes.data(), pq_packed.data(), pq_->m());
                        pq_ptr = pq_packed.data();
                        pq_len = static_cast<uint32_t>(pq_packed.size());
                    }
                    else
                    {
                        // Either: need_raw but SoA has no raw-space (cannot store raw codes),
                        // or soa lacks any PQ codes storage. In either case, do not store
                        // truncated packed4 codes which would be misleading for PQ approximate eval.
                        pq_ptr = nullptr;
                        pq_len = 0;
                        if (need_raw && !soa_has_raw)
                        {
                            std::cerr << "[VectorStore] Warning: PQ k=" << pq_->k()
                                      << " requires raw 8-bit storage but SoA has no pq_codes block; skipping storing PQ codes to avoid truncation\n";
                        }
                        else if (!soa_has_packed && !soa_has_raw)
                        {
                            // SoA has no PQ storage at all
                        }
                    }
                }

                // Default id_entry to label encoding
                uint64_t id_entry_to_store = pomai::ai::soa::IdEntry::pack_label(new_label);

                // If we have an arena, prefer publishing a local arena offset into the SoA ids block.
                // payload_bytes was computed earlier from pphnsw_->getSeedSize() (outer scope).
                if (arena_)
                {
                    size_t payload_bytes = 0;
                    try
                    {
                        size_t seed_size = pphnsw_->getSeedSize();
                        if (seed_size >= sizeof(PPEHeader))
                            payload_bytes = seed_size - sizeof(PPEHeader);
                    }
                    catch (...)
                    {
                        payload_bytes = 0;
                    }

                    if (payload_bytes > 0)
                    {
                        char *blob_hdr = arena_->alloc_blob(static_cast<uint32_t>(payload_bytes));
                        if (blob_hdr)
                        {
                            // copy payload bytes (vec points to the original vector payload)
                            std::memcpy(blob_hdr + sizeof(uint32_t), reinterpret_cast<const char *>(vec), payload_bytes);
                            uint64_t off = arena_->offset_from_blob_ptr(blob_hdr);
                            if (off != UINT64_MAX)
                            {
                                id_entry_to_store = pomai::ai::soa::IdEntry::pack_local_offset(off);
                                // record reverse mapping identry->label so searches can map local id_entry back to the original label/key
                                {
                                    std::lock_guard<std::mutex> lk(label_map_mu_);
                                    identry_to_label_[id_entry_to_store] = new_label;
                                }
                            }
                        }
                        else
                        {
                            // alloc failed -> fallback: keep label-based id_entry (or optionally demote to remote)
                            // (existing behavior is to store label so refine() can still work)
                        }
                    }
                }

                // Append into SoA (either or both may be present)
                if (soa_)
                {
                    // pass pq_ptr/pq_len as determined above (may be null/0 if we chose to skip PQ storage)
                    soa_->append_vector(fp_ptr, fp_len, pq_ptr, pq_len, id_entry_to_store);
                }

                return true;
            };

            // Insert into HNSW / PPHNSW depending on payload layout
            if (payload_bytes == dim_ * sizeof(float))
            {
                if (label != 0)
                    pphnsw_->addPoint(vec, static_cast<hnswlib::labeltype>(label), /*replace_deleted=*/true);
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
                const int bits = 8;
                if (label != 0)
                    pphnsw_->addQuantizedPoint(vec, dim_, bits, static_cast<hnswlib::labeltype>(label), /*replace_deleted=*/true);
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
                const int bits = 4;
                if (label != 0)
                    pphnsw_->addQuantizedPoint(vec, dim_, bits, static_cast<hnswlib::labeltype>(label), /*replace_deleted=*/true);
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
                std::ostringstream ss;
                ss << "addPoint/addQuantizedPoint mismatch: index payload_bytes=" << payload_bytes
                   << " dim=" << dim_ << " (expected float bytes=" << (dim_ * sizeof(float))
                   << " or quant8 bytes=" << dim_ << " or packed4 bytes=" << ((dim_ + 1) / 2) << ")";
                throw std::runtime_error(ss.str());
            }

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
            label = read_label_from_map(key, klen);

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

        // We don't unpublish SoA entries here.
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

        std::vector<std::pair<std::string, float>> out;
        if (!pphnsw_ || !query || dim != dim_ || topk == 0)
            return out;

        // Phase2: prefilter path using fingerprints stored in SoA
        if (soa_ && fingerprint_ && soa_->fingerprint_bits() > 0 && soa_->ids_ptr() != nullptr)
        {
            try
            {
                std::vector<uint8_t> qfp(fingerprint_bytes_);
                fingerprint_->compute(query, qfp.data());

                size_t nv = static_cast<size_t>(soa_->num_vectors());

                std::vector<uint8_t> db_compact;
                db_compact.reserve(nv * fingerprint_bytes_);

                std::vector<size_t> published_indices;
                published_indices.reserve(nv);

                for (size_t i = 0; i < nv; ++i)
                {
                    const uint8_t *p = soa_->fingerprint_ptr(i);
                    if (p)
                    {
                        published_indices.push_back(i);
                        db_compact.insert(db_compact.end(), p, p + fingerprint_bytes_);
                    }
                }

                if (!published_indices.empty())
                {
                    uint32_t base_thresh = static_cast<uint32_t>(pomai::config::runtime.prefilter_hamming_threshold);
                    if (base_thresh == 0)
                        base_thresh = 128;

                    size_t bits = pomai::config::runtime.fingerprint_bits; // runtime-configured fingerprint bits
                    uint32_t thresh = base_thresh;

                    // Adaptive for small dim/N
                    if (dim_ < 128 || nv < 5000)
                    {
                        thresh = static_cast<uint32_t>(std::max<uint32_t>(base_thresh, static_cast<uint32_t>(bits / 2)));
                        std::cerr << "[VectorStore] adaptive prefilter: dim=" << dim_ << " nv=" << nv << " bits=" << bits << " -> thresh=" << thresh << "\n";
                    }

                    std::vector<size_t> compact_candidates;
                    pomai::ai::prefilter::collect_candidates_threshold(qfp.data(), fingerprint_bytes_, db_compact.data(), published_indices.size(), thresh, compact_candidates);

                    std::vector<size_t> candidates;
                    candidates.reserve(compact_candidates.size());
                    for (size_t ci : compact_candidates)
                        candidates.push_back(published_indices[ci]);

                    std::cerr << "[VectorStore] prefilter candidates=" << candidates.size() << " (thresh=" << thresh << ")\n";

                    // Fallback: if too few candidates for robust PQ/refine
                    size_t min_needed = std::max<size_t>(topk * 5, 64);
                    if (candidates.empty() || candidates.size() < min_needed)
                    {
                        std::cerr << "[VectorStore] prefilter produced too few candidates (" << candidates.size() << "), falling back to published_indices (" << published_indices.size() << ")\n";
                        candidates = std::move(published_indices);
                        if (candidates.empty())
                        {
                            candidates.resize(nv);
                            std::iota(candidates.begin(), candidates.end(), 0);
                        }
                    }

                    // PQ approximate if available AND safe to use
                    bool use_pq_approx = false;
                    bool soa_has_raw_codes = (soa_ && soa_->pq_codes_ptr(0) != nullptr);
                    bool soa_has_packed = (pq_packed_bytes_ > 0 && soa_ && soa_->pq_packed_ptr(0) != nullptr);

                    if (pq_ && soa_->pq_m() != 0 && soa_->pq_k() != 0)
                    {
                        if (pq_->k() > 16 && soa_has_raw_codes)
                        {
                            // can use raw8 approximate evaluation
                            use_pq_approx = true;
                        }
                        else if (pq_->k() <= 16 && soa_has_packed)
                        {
                            // can use packed4 approximate evaluation
                            use_pq_approx = true;
                        }
                    }

                    if (use_pq_approx)
                    {
                        // compute PQ distance tables (ProductQuantizer)
                        std::vector<float> tables(pq_->m() * pq_->k());
                        pq_->compute_distance_tables(query, tables.data());

                        // compute approximate distances for all candidates (choose packed4 or raw8 as appropriate)
                        std::vector<float> approx_dists(candidates.size());

                        if (pq_->k() > 16 && soa_has_raw_codes)
                        {
                            // use raw 8-bit codes stored in SoA
                            std::vector<uint8_t> raw_compact;
                            raw_compact.reserve(candidates.size() * pq_->m());
                            for (size_t idx : candidates)
                            {
                                const uint8_t *pc = soa_->pq_codes_ptr(idx); // returns pointer to m bytes of raw codes
                                if (pc)
                                    raw_compact.insert(raw_compact.end(), pc, pc + pq_->m());
                                else
                                    raw_compact.insert(raw_compact.end(), pq_->m(), 0);
                            }
                            pq_approx_dist_batch_raw8(tables.data(), pq_->m(), pq_->k(),
                                                      raw_compact.data(), candidates.size(), approx_dists.data());
                        }
                        else
                        {
                            // fallback: use packed4 path (existing behaviour)
                            std::vector<uint8_t> packed_compact;
                            packed_compact.reserve(candidates.size() * pq_packed_bytes_);
                            for (size_t idx : candidates)
                            {
                                const uint8_t *pc = soa_->pq_packed_ptr(idx);
                                if (pc)
                                    packed_compact.insert(packed_compact.end(), pc, pc + pq_packed_bytes_);
                                else
                                    packed_compact.insert(packed_compact.end(), pq_packed_bytes_, 0);
                            }
                            pq_approx_dist_batch_packed4(tables.data(), pq_->m(), pq_->k(),
                                                         packed_compact.data(), candidates.size(), approx_dists.data());
                        }

                        // pick top Napprox
                        size_t Napprox = std::min<size_t>(std::max<size_t>(topk, 100), static_cast<size_t>(256));
                        Napprox = std::min<size_t>(Napprox, candidates.size());

                        CandidateCollector collector(Napprox);
                        for (size_t i = 0; i < candidates.size(); ++i)
                        {
                            // CandidateCollector::add(id, score)
                            collector.add(candidates[i], approx_dists[i]);
                        }
                        auto top_pairs = collector.topk(); // vector<pair<id,score>> best-first

                        std::vector<size_t> approx_top;
                        approx_top.reserve(top_pairs.size());
                        for (const auto &pp : top_pairs)
                            approx_top.push_back(pp.first);

                        // build label fetcher: resolves LABEL id_entries by reading index memory via pphnsw_->getDataByLabel
                        std::function<bool(uint64_t, std::vector<float> &)> label_fetcher = nullptr;
                        if (pphnsw_)
                        {
                            label_fetcher = [this, dim](uint64_t id_entry, std::vector<float> &out_buf) -> bool
                            {
                                using namespace pomai::ai::soa;
                                if (!IdEntry::is_label(id_entry))
                                    return false;
                                uint64_t label = IdEntry::unpack_label(id_entry);
                                char *data = pphnsw_->getDataByLabel(static_cast<hnswlib::labeltype>(label));
                                if (!data)
                                    return false;
                                // payload pointer is after PPEHeader
                                const char *payload = data + sizeof(PPEHeader);
                                size_t expect_bytes = dim * sizeof(float);
                                // basic sanity: ensure payload pointer not null
                                if (!payload)
                                    return false;
                                out_buf.resize(dim);
                                std::memcpy(out_buf.data(), payload, expect_bytes);
                                return true;
                            };
                        }

                        const uint64_t *ids_block = soa_->ids_ptr();
                        auto refined = pomai::ai::refine::refine_topk_l2(query, dim_, approx_top, ids_block, arena_, topk, label_fetcher);

                        // If refine returned nothing, dump diagnostics and fall back to HNSW search.
                        if (refined.empty())
                        {
                            std::cerr << "[VectorStore] WARNING: refine_topk_l2 returned 0 results. Dumping diagnostics...\n";

                            // Show a few candidate entries and their id_entry values and whether we have a label->key mapping.
                            size_t showN = std::min<size_t>(10, approx_top.size());
                            for (size_t i = 0; i < showN; ++i)
                            {
                                size_t soaid = approx_top[i];
                                uint64_t identry = 0;
                                try
                                {
                                    identry = soa_->id_entry_at(soaid);
                                }
                                catch (...)
                                {
                                    identry = UINT64_MAX;
                                }
                                std::cerr << "[VectorStore][diag] approx_top[" << i << "] soaid=" << soaid << " identry=0x" << std::hex << identry << std::dec;
                                {
                                    std::lock_guard<std::mutex> lk(label_map_mu_);
                                    // If identry encodes a label, lookup the unpacked label
                                    if (pomai::ai::soa::IdEntry::is_label(identry))
                                    {
                                        uint64_t lbl = pomai::ai::soa::IdEntry::unpack_label(identry);
                                        auto it = label_to_key_.find(lbl);
                                        if (it != label_to_key_.end())
                                            std::cerr << " label_to_key FOUND (unpacked)\n";
                                        else
                                            std::cerr << " label_to_key MISSING (unpacked)\n";
                                    }
                                    else
                                    {
                                        auto it = label_to_key_.find(identry);
                                        if (it != label_to_key_.end())
                                            std::cerr << " label_to_key FOUND\n";
                                        else
                                            std::cerr << " label_to_key MISSING\n";
                                    }
                                }
                            }

                            // Also report whether arena_ is set (refine may need arena to load payloads)
                            std::cerr << "[VectorStore][diag] arena_=" << (arena_ ? "present" : "null") << " soa_->num_vectors()=" << soa_->num_vectors() << "\n";

                            // Conservative fallback: run HNSW search and return its results (guarantees non-empty on populated index)
                            try
                            {
                                auto buf = build_seed_buffer(query);
                                auto pqres = pphnsw_->searchKnnAdaptive(buf.get(), topk, 0.0f);
                                std::vector<std::pair<std::string, float>> tmp;
                                tmp.reserve(pqres.size());
                                while (!pqres.empty())
                                {
                                    auto pr = pqres.top();
                                    pqres.pop();
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
                                std::cerr << "[VectorStore] Fallback HNSW returned " << tmp.size() << " results\n";
                                return tmp;
                            }
                            catch (const std::exception &e)
                            {
                                std::cerr << "[VectorStore] Fallback HNSW threw: " << e.what() << "\n";
                                return {}; // absolute last resort
                            }
                        }

                        // --- otherwise, continue processing 'refined' as before
                        out.reserve(refined.size());
                        for (auto &pr : refined)
                        {
                            size_t soaid = pr.first;
                            float dist = pr.second;
                            uint64_t identry = soa_->id_entry_at(soaid);
                            std::string key;
                            {
                                std::lock_guard<std::mutex> lk(label_map_mu_);
                                // If identry is a LABEL entry, unpack and lookup label_to_key_ by unpacked label
                                if (pomai::ai::soa::IdEntry::is_label(identry))
                                {
                                    uint64_t lbl = pomai::ai::soa::IdEntry::unpack_label(identry);
                                    auto it = label_to_key_.find(lbl);
                                    if (it != label_to_key_.end())
                                        key = it->second;
                                }
                                else
                                {
                                    auto it = label_to_key_.find(identry);
                                    if (it != label_to_key_.end())
                                        key = it->second;
                                    else
                                    {
                                        // try reverse mapping identry->label (for published local offsets)
                                        auto it2 = identry_to_label_.find(identry);
                                        if (it2 != identry_to_label_.end())
                                        {
                                            auto it3 = label_to_key_.find(it2->second);
                                            if (it3 != label_to_key_.end())
                                                key = it3->second;
                                        }
                                    }
                                }
                            }
                            if (key.empty() && map_)
                            {
                                // Prefer to show unpacked label for LABEL entries for readability
                                if (pomai::ai::soa::IdEntry::is_label(identry))
                                {
                                    uint64_t lbl = pomai::ai::soa::IdEntry::unpack_label(identry);
                                    key = std::to_string(lbl);
                                }
                                else
                                {
                                    key = std::to_string(identry);
                                }
                            }
                            out.emplace_back(std::move(key), dist);
                            if (out.size() >= topk)
                                break;
                        }
                        return out;
                    }
                    else
                    {
                        // No PQ-approx (either PQ not present or unsafe to use because k>16 / missing raw codes).
                        // Fall back to exact refine on candidates.
                        const uint64_t *ids_block = soa_->ids_ptr();

                        // build label fetcher: resolves LABEL id_entries by reading index memory via pphnsw_->getDataByLabel
                        std::function<bool(uint64_t, std::vector<float> &)> label_fetcher = nullptr;
                        if (pphnsw_)
                        {
                            label_fetcher = [this, dim](uint64_t id_entry, std::vector<float> &out_buf) -> bool
                            {
                                using namespace pomai::ai::soa;
                                if (!IdEntry::is_label(id_entry))
                                    return false;
                                uint64_t label = IdEntry::unpack_label(id_entry);
                                char *data = pphnsw_->getDataByLabel(static_cast<hnswlib::labeltype>(label));
                                if (!data)
                                    return false;
                                const char *payload = data + sizeof(PPEHeader);
                                size_t expect_bytes = dim * sizeof(float);
                                if (!payload)
                                    return false;
                                out_buf.resize(dim);
                                std::memcpy(out_buf.data(), payload, expect_bytes);
                                return true;
                            };
                        }

                        auto refined = pomai::ai::refine::refine_topk_l2(query, dim_, candidates, ids_block, arena_, topk, label_fetcher);
                        out.reserve(refined.size());
                        for (auto &p : refined)
                        {
                            size_t soaid = p.first;
                            float dist = p.second;
                            uint64_t identry = soa_->id_entry_at(soaid);
                            std::string key;
                            {
                                std::lock_guard<std::mutex> lk(label_map_mu_);
                                if (pomai::ai::soa::IdEntry::is_label(identry))
                                {
                                    uint64_t lbl = pomai::ai::soa::IdEntry::unpack_label(identry);
                                    auto it = label_to_key_.find(lbl);
                                    if (it != label_to_key_.end())
                                        key = it->second;
                                }
                                else
                                {
                                    auto it = label_to_key_.find(identry);
                                    if (it != label_to_key_.end())
                                        key = it->second;
                                    else
                                    {
                                        auto it2 = identry_to_label_.find(identry);
                                        if (it2 != identry_to_label_.end())
                                        {
                                            auto it3 = label_to_key_.find(it2->second);
                                            if (it3 != label_to_key_.end())
                                                key = it3->second;
                                        }
                                    }
                                }
                            }
                            if (key.empty() && map_)
                            {
                                if (pomai::ai::soa::IdEntry::is_label(identry))
                                {
                                    uint64_t lbl = pomai::ai::soa::IdEntry::unpack_label(identry);
                                    key = std::to_string(lbl);
                                }
                                else
                                {
                                    key = std::to_string(identry);
                                }
                            }
                            out.emplace_back(std::move(key), dist);
                            if (out.size() >= topk)
                                break;
                        }
                        return out;
                    }
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "VectorStore::search prefilter/refine exception: " << e.what() << "\n";
            }
        }

        // Fallback HNSW search
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

    VectorStore::MemoryUsage VectorStore::memoryUsage() const noexcept
    {
        MemoryUsage out{};
        try
        {
            if (sharded_mode_)
            {
                if (ppsm_)
                {
                    auto mu = ppsm_->memoryUsage();
                    out.payload_bytes = mu.payload_bytes;
                    out.index_overhead_bytes = mu.index_overhead_bytes;
                    out.total_bytes = mu.total_bytes;
                    return out;
                }

                uint64_t total_payload = 0, total_over = 0;
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

            // Non-sharded (single-mode)
            if (pphnsw_)
            {
                // Base index bytes (what index believes it's storing as payload per-seed)
                size_t cnt = pphnsw_->elementCount();
                size_t seed_size = pphnsw_->getSeedSize();
                uint64_t payload_bytes = static_cast<uint64_t>(seed_size) * static_cast<uint64_t>(cnt);

                // start with PPHNSW estimate for total; we'll add other contributions below
                size_t estimated_total = pphnsw_->estimatedMemoryUsageBytes(/*avg_degree_multiplier=*/2);
                uint64_t index_over = 0;
                if (estimated_total > payload_bytes)
                    index_over = static_cast<uint64_t>(estimated_total) - payload_bytes;

                // ----- SoA contributions (estimate using public accessors) -----
                uint64_t soa_bytes = 0;
                try
                {
                    if (soa_)
                    {
                        uint64_t nv = soa_->num_vectors();
                        // codebooks block (if present)
                        soa_bytes += static_cast<uint64_t>(soa_->codebooks_size_bytes());
                        // fingerprints block (approx)
                        if (soa_->fingerprint_bits() > 0)
                        {
                            uint64_t fp_bytes = static_cast<uint64_t>((soa_->fingerprint_bits() + 7) / 8);
                            soa_bytes += fp_bytes * nv;
                        }
                        // pq packed bytes block (on-disk packed4)
                        if (pq_packed_bytes_ > 0)
                            soa_bytes += static_cast<uint64_t>(pq_packed_bytes_) * nv;
                        // ids block (uint64 per vector)
                        soa_bytes += static_cast<uint64_t>(nv) * static_cast<uint64_t>(sizeof(uint64_t));
                        // PPE block: we don't have public accessor for ppe size here; skip or estimate conservatively
                    }
                }
                catch (...)
                {
                    // ignore soa accounting failures (best-effort)
                }

                // ----- PQ in-RAM codebooks (if pq_ present) -----
                uint64_t pq_bytes = 0;
                try
                {
                    if (pq_)
                    {
                        // codebooks float count * sizeof(float)
                        size_t floats = pq_->codebooks_float_count();
                        pq_bytes += static_cast<uint64_t>(floats) * static_cast<uint64_t>(sizeof(float));
                        // In-RAM per-element codes (8-bit) may exist in other components (e.g., PPPQ or PQ instances).
                        // We don't attempt to introspect PPPQ internal buffers here.
                    }
                }
                catch (...)
                {
                }

                // ----- Arena contribution (best-effort, use capacity / blob region size as upper bound) -----
                uint64_t arena_bytes = 0;
                try
                {
                    if (arena_)
                    {
                        // PomaiArena::get_capacity_bytes() returns total mapped region size allocated for arena
                        // treat it as an upper bound for "payload" stored in arena.
                        arena_bytes = static_cast<uint64_t>(arena_->get_capacity_bytes());
                    }
                }
                catch (...)
                {
                }

                // Aggregate
                out.payload_bytes = payload_bytes + soa_bytes + pq_bytes + arena_bytes;
                out.index_overhead_bytes = index_over + /* add SoA/index metadata overhead conservatively */ 0;
                out.total_bytes = out.payload_bytes + out.index_overhead_bytes;
                return out;
            }

            return out;
        }
        catch (...)
        {
            return MemoryUsage{};
        }
    }

    bool VectorStore::init_single(size_t dim, size_t max_elements, size_t M, size_t ef_construction, PomaiArena *arena)
    {
        if (dim == 0 || max_elements == 0)
            return false;

        arena_ = arena;

        l2space_.reset(new hnswlib::L2Space(static_cast<int>(dim)));

        try
        {
            pphnsw_.reset(new PPHNSW<float>(l2space_.get(), max_elements, M, ef_construction));
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

        // PPPQ: create, train quickly on random samples and attach to PPHNSW.
        try
        {
            size_t pq_m = 8;
            size_t pq_k = 256;
            size_t max_elems = max_elements;
            while (pq_m > 1 && (dim % pq_m) != 0)
                pq_m /= 2;
            if (pq_m < 1 || (dim % pq_m) != 0)
            {
                std::clog << "VectorStore: Skipping PPPQ (dim=" << dim << ")\n";
            }
            else
            {
                auto ppq = std::make_unique<pomai::ai::PPPQ>(dim, pq_m, pq_k, max_elems, "pppq_codes.mmap");
                size_t n_train_ppq = std::min<size_t>(20000, max_elems);
                std::vector<float> samples_ppq;
                samples_ppq.resize(n_train_ppq * dim);
                std::mt19937_64 rng(123456);
                std::uniform_real_distribution<float> ud(0.0f, 1.0f);
                for (size_t i = 0; i < n_train_ppq * dim; ++i)
                    samples_ppq[i] = ud(rng);
                ppq->train(samples_ppq.data(), n_train_ppq, 10);
                pphnsw_->setPPPQ(std::move(ppq));
                if (!ppq_demote_running_.load(std::memory_order_acquire))
                {
                    ppq_demote_running_.store(true, std::memory_order_release);
                    ppq_demote_thread_ = std::thread([this]()
                                                     {
                    auto *ppq = pphnsw_->getPPPQ();
                    if (!ppq) return;
                    while (ppq_demote_running_.load(std::memory_order_acquire))
                    {
                        try { ppq->purgeCold(ppq_demote_cold_thresh_ms_); } catch (...) {}
                        std::this_thread::sleep_for(std::chrono::milliseconds(ppq_demote_interval_ms_));
                    } });
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore: PPPQ init/train failed: " << e.what() << "\n";
        }

        // --- PQ training (Phase 3) ---
        try
        {
            size_t pq_m = 8;
            size_t pq_k = 256;
            while (pq_m > 1 && (dim % pq_m) != 0)
                pq_m /= 2;
            if (pq_m >= 1 && (dim % pq_m) == 0)
            {
                size_t max_elems = max_elements;
                size_t n_train = std::min<size_t>(20000, max_elems);
                std::vector<float> samples;
                samples.resize(n_train * dim);
                std::mt19937_64 rng2(654321);
                std::uniform_real_distribution<float> ud2(0.0f, 1.0f);
                for (size_t i = 0; i < n_train * dim; ++i)
                    samples[i] = ud2(rng2);
                pq_.reset(new ProductQuantizer(dim, pq_m, pq_k));
                pq_->train(samples.data(), n_train, 10);

                // populate local Codebooks object from pq_
                if (pq_)
                {
                    const float *cb = pq_->codebooks_data();
                    size_t floats = pq_->codebooks_float_count();
                    if (cb && floats > 0)
                    {
                        std::vector<float> raw(cb, cb + floats);
                        codebooks_.set_codebooks_from_raw(dim, pq_->m(), pq_->k(), raw);
                    }
                }

                if (soa_ && soa_->codebooks_size_bytes() > 0)
                {
                    size_t expected_bytes = pq_->codebooks_float_count() * sizeof(float);
                    if (expected_bytes == soa_->codebooks_size_bytes())
                    {
                        if (!soa_->write_codebooks(pq_->codebooks_data(), pq_->codebooks_float_count()))
                            std::cerr << "[VectorStore] failed to write codebooks into SoA mapping\n";
                        else
                            std::cerr << "[VectorStore] wrote trained codebooks into SoA mapping\n";
                    }
                    else
                    {
                        std::cerr << "[VectorStore] codebooks size mismatch: PQ expects " << expected_bytes
                                  << " bytes, SoA codebooks block has " << soa_->codebooks_size_bytes() << " bytes\n";
                    }
                }
                pq_packed_bytes_ = ProductQuantizer::packed4BytesPerVec(pq_->m());
                std::cerr << "[VectorStore] trained ProductQuantizer m=" << pq_->m() << " k=" << pq_->k()
                          << " packed_bytes=" << pq_packed_bytes_ << "\n";
                (void)pq_->save_codebooks(std::string("pq_codebooks.bin"));
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "VectorStore: PQ train failed: " << e.what() << "\n";
            pq_.reset();
            pq_packed_bytes_ = 0;
        }
        return true;
    }

} // namespace pomai::ai