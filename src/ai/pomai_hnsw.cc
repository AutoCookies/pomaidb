/*
 * src/ai/pomai_hnsw.cc
 *
 * Out-of-line implementations for PPHNSW members.
 *
 * Updated to use atomic helpers when updating PPEHeader flags and the 8-byte
 * payload (arena/local offset or remote_id) that lives immediately after the
 * PPEHeader in each element's storage. This avoids torn 64-bit writes and
 * races where a reader may observe a flag change before payload is fully written.
 *
 * Also implements a background demoter/promoter thread that:
 *  - periodically scans elements,
 *  - uses PPEHeader predictor (predict_next_ns) to decide promotion/demotion,
 *  - performs atomic payload updates and flag updates with safe publish ordering,
 *  - logs actions taken for observability.
 *
 * The demoter/promoter uses runtime-configurable thresholds supplied via
 * pomai::config::runtime:
 *   - runtime.promote_lookahead_ms   (ms) : used to decide promotion safety
 *   - runtime.demote_threshold_ms    (ms) : used to allow a demotion tolerance window
 *   - runtime.hot_size_limit         (advisory) : not enforced here (left for arena policies)
 *
 * Conventions used here:
 *  - When writing a 64-bit payload (local offset / remote id) use atomic_store_u64(ptr, val)
 *    provided by atomic_utils to ensure atomic 8-byte stores on supported platforms.
 *  - When changing flags use PPEHeader::atomic_set_flags / PPEHeader::atomic_clear_flags helpers.
 *  - Ordering: write payload first (atomic store), then set the corresponding flag(s).
 *    Readers should load flags atomically (use PPEHeader::atomic_load_flags when doing so)
 *    and then atomically load the payload (atomic_load_u64) if flags indicate payload presence.
 *
 * Note: file focuses on the demoter/promoter changes and related call-sites.
 */

#include "src/ai/pomai_hnsw.h"
#include "src/ai/quantize.h"
#include "src/ai/space_quantized.h"
#include "src/ai/ppe.h"
#include "src/ai/pomai_space.h"
#include "src/memory/arena.h"
#include "src/ai/pp_ivf.h"
#include "src/ai/pppq.h" // PPPQ integration
#include "src/ai/atomic_utils.h"
#include "src/core/config.h"

#include <cstring>
#include <cstdlib>
#include <cassert>
#include <new>
#include <mutex>
#include <algorithm>
#include <memory>
#include <cstdint>
#include <string>
#include <functional>
#include <thread>
#include <chrono>
#include <atomic>
#include <iostream>
#include <limits>
#include <vector>
#include <sstream>

namespace pomai::ai
{
    thread_local void *detail::tmp_pomai_space_void = nullptr;

    struct OperatorDeleteDeleter
    {
        void operator()(void *p) const noexcept { ::operator delete(p); }
    };

    // --- addPoint with PPE hint-aware wiring + optional PPIVF/PPPQ registration ---
    template <typename dist_t>
    void PPHNSW<dist_t>::addPoint(const void *datapoint, hnswlib::labeltype label, bool replace_deleted)
    {
        PomaiSpace<dist_t> *space_ptr = this->pomai_space_owner_.get();
        if (space_ptr == nullptr)
            throw std::runtime_error("pomai_space not initialized");

        // Propagate PPPQ pointer if attached
        if (pp_pq_ && pomai_space_owner_)
            pomai_space_owner_->setPPPQ(pp_pq_.get());

        size_t full_size = space_ptr->get_data_size();
        size_t payload_bytes = (full_size >= sizeof(PPEHeader)) ? (full_size - sizeof(PPEHeader)) : 0;

        bool computed_ivf = false;
        int ivf_cluster = -1;
        const uint8_t *ivf_code = nullptr;
        std::vector<uint8_t> ivf_code_local;

        const float *raw_fvec = nullptr;
        if (pp_ivf_)
        {
            size_t underlying_bytes = space_ptr->underlying_data_size();
            if (underlying_bytes > 0 && payload_bytes == underlying_bytes)
            {
                if (underlying_bytes % sizeof(float) == 0)
                {
                    size_t expected_floats = underlying_bytes / sizeof(float);
                    if (payload_bytes == expected_floats * sizeof(float))
                    {
                        const float *fvec = reinterpret_cast<const float *>(datapoint);
                        raw_fvec = fvec;
                        try
                        {
                            ivf_cluster = pp_ivf_->assign_cluster(fvec);
                            const uint8_t *codeptr = pp_ivf_->encode_pq(fvec);
                            if (codeptr)
                            {
                                ivf_code_local.assign(codeptr, codeptr + pp_ivf_->pq_bytes_per_code());
                                ivf_code = ivf_code_local.data();
                            }
                            computed_ivf = true;
                        }
                        catch (...)
                        {
                            computed_ivf = false;
                            ivf_cluster = -1;
                            ivf_code = nullptr;
                            ivf_code_local.clear();
                        }
                    }
                }
            }
        }

        std::unique_ptr<void, OperatorDeleteDeleter> buf(::operator new(full_size));
        void *seed_buffer = buf.get();
        // Zero header bytes then placement-new PPEHeader to properly initialize atomics.
        std::memset(seed_buffer, 0, sizeof(PPEHeader));
        new (seed_buffer) PPEHeader();
        // Copy payload after header object is constructed.
        std::memcpy(static_cast<char *>(seed_buffer) + sizeof(PPEHeader),
                    datapoint,
                    full_size - sizeof(PPEHeader));

        // Determine whether update or insert
        bool will_update_existing = false;
        hnswlib::tableint existing_internal = static_cast<hnswlib::tableint>(-1);
        {
            std::unique_lock<std::mutex> lock(this->label_lookup_lock);
            auto it = this->label_lookup_.find(label);
            if (it != this->label_lookup_.end())
            {
                existing_internal = it->second;
                if (!this->isMarkedDeleted(existing_internal))
                    will_update_existing = true;
            }
        }

        if (will_update_existing)
        {
            Base::addPoint(seed_buffer, label, -1);
            return;
        }

        size_t orig_M = this->M_;
        // (wiring heuristics and temporary M override could be here; omitted for brevity)

        Base::addPoint(seed_buffer, label, replace_deleted);

        // Restore M_ if changed (no-op here)
        if (static_cast<size_t>(orig_M) != orig_M)
            const_cast<PPHNSW<dist_t> *>(this)->M_ = orig_M;

        // Find assigned internal id
        hnswlib::tableint assigned_internal = static_cast<hnswlib::tableint>(-1);
        {
            std::unique_lock<std::mutex> lock(this->label_lookup_lock);
            auto it = this->label_lookup_.find(label);
            if (it == this->label_lookup_.end())
                throw std::runtime_error("PPHNSW::addPoint: label not found after insertion");
            assigned_internal = it->second;
        }

        // placement-new PPEHeader in-place and initialize hints
        char *dst = this->getDataByInternalId(assigned_internal);
        new (dst) PPEHeader();
        PPEHeader *h = reinterpret_cast<PPEHeader *>(dst);
        h->init_hints(static_cast<uint16_t>(orig_M), static_cast<uint16_t>(this->ef_));

        // store external label inside PPEHeader
        h->set_label(static_cast<uint64_t>(label));

        // Move payload into arena if configured
        if (pomai_arena_)
        {
            const char *vec_src = dst + sizeof(PPEHeader);
            size_t vec_size = full_size - sizeof(PPEHeader);

            // Try to allocate inside arena
            char *blob_hdr = pomai_arena_->alloc_blob(static_cast<uint32_t>(vec_size));
            if (blob_hdr)
            {
                std::memcpy(blob_hdr + sizeof(uint32_t), vec_src, vec_size);
                uint64_t offset = pomai_arena_->offset_from_blob_ptr(blob_hdr);

                // Atomic store of the 8-byte offset payload, then atomically set INDIRECT flag
                // 1. Cast memory location to atomic<uint64_t> (zero-copy cast)
                //    We target the memory immediately following the PPEHeader.
                std::atomic<uint64_t> *payload_atomic = reinterpret_cast<std::atomic<uint64_t> *>(dst + sizeof(PPEHeader));

                // 2. RELEASE STORE: Ghi offset payload xuống bộ nhớ.
                //    Đảm bảo dữ liệu blob đã được memcpy xong trước khi con trỏ này hiển thị.
                payload_atomic->store(offset, std::memory_order_release);

                // 3. RELEASE FLAG: Bật cờ INDIRECT.
                //    Luồng đọc (Search) sẽ thấy cờ này SAU KHI payload != 0.
                pomai::ai::atomic_utils::atomic_store_u32(&h->flags, PPE_FLAG_INDIRECT);

                std::clog << "[PPHNSW] addPoint: stored payload to arena for internal=" << assigned_internal
                          << " offset=" << offset << " label=" << label << "\n";
            }
            else
            {
                // Fallback: demote direct to file and store remote_id atomically
                uint32_t len32 = static_cast<uint32_t>(vec_size);
                std::vector<char> tmpbuf(sizeof(uint32_t) + vec_size + 1);
                std::memcpy(tmpbuf.data(), &len32, sizeof(uint32_t));
                std::memcpy(tmpbuf.data() + sizeof(uint32_t), vec_src, vec_size);
                tmpbuf[sizeof(uint32_t) + vec_size] = '\0';

                uint64_t remote_id = pomai_arena_->demote_blob_data(tmpbuf.data(), static_cast<uint32_t>(tmpbuf.size()));
                if (remote_id == 0)
                {
                    throw std::runtime_error("PomaiArena alloc_blob and demote failed in addPoint");
                }

                // atomic store remote id into payload then set flags INDIRECT+REMOTE
                // 1. Cast memory location
                std::atomic<uint64_t> *payload_atomic = reinterpret_cast<std::atomic<uint64_t> *>(dst + sizeof(PPEHeader));

                // 2. RELEASE STORE: Ghi remote_id
                payload_atomic->store(remote_id, std::memory_order_release);

                // 3. RELEASE FLAG: Bật cờ INDIRECT + REMOTE
                //    Sử dụng atomic_store để đảm bảo tính nhìn thấy (visibility) tuần tự.
                pomai::ai::atomic_utils::atomic_store_u32(&h->flags, PPE_FLAG_INDIRECT | PPE_FLAG_REMOTE);

                std::clog << "[PPHNSW] addPoint: demoted payload for internal=" << assigned_internal
                          << " remote_id=" << remote_id << " label=" << label << "\n";
            }
        }

        // IVFPQ/PPPQ registration unchanged (omitted) ...
        if (pp_ivf_ && computed_ivf && ivf_cluster >= 0)
        {
            try
            {
                pp_ivf_->add_label(static_cast<uint64_t>(label), ivf_cluster, ivf_code);
            }
            catch (...)
            {
            }
        }

        if (pp_pq_ && raw_fvec)
        {
            try
            {
                pp_pq_->addVec(raw_fvec, static_cast<size_t>(label));
            }
            catch (...)
            {
            }
        }
    }

    // --- addQuantizedPoint (similar updates for atomic payload+flags) ---
    template <typename dist_t>
    void PPHNSW<dist_t>::addQuantizedPoint(const float *vec, size_t dim, int bits, hnswlib::labeltype label, bool replace_deleted)
    {
        PomaiSpace<dist_t> *space_ptr = this->pomai_space_owner_.get();
        if (!space_ptr)
            throw std::runtime_error("pomai_space not initialized");

        // propagate PPPQ pointer if attached
        if (pp_pq_ && pomai_space_owner_)
            pomai_space_owner_->setPPPQ(pp_pq_.get());

        size_t payload_bytes = space_ptr->underlying_data_size();

        size_t expected_payload = 0;
        if (bits == 8)
            expected_payload = dim;
        else if (bits == 4)
            expected_payload = (dim + 1) / 2;
        else
            throw std::invalid_argument("addQuantizedPoint: only 8 or 4 bits supported");

        if (payload_bytes != expected_payload)
        {
            std::ostringstream ss;
            ss << "addQuantizedPoint: payload_size mismatch with space->get_data_size()"
               << " payload_bytes=" << payload_bytes << " expected=" << expected_payload << " dim=" << dim << " bits=" << bits;
            throw std::runtime_error(ss.str());
        }

        size_t full_size = sizeof(PPEHeader) + payload_bytes;
        std::unique_ptr<void, OperatorDeleteDeleter> buf(::operator new(full_size));
        void *seed_buffer = buf.get();

        std::memset(seed_buffer, 0, sizeof(PPEHeader));
        new (seed_buffer) PPEHeader();

        void *payload_ptr = static_cast<char *>(seed_buffer) + sizeof(PPEHeader);

        if (bits == 8)
            quantize::quantize_u8(vec, static_cast<uint8_t *>(payload_ptr), dim);
        else
            quantize::quantize_u4(vec, static_cast<uint8_t *>(payload_ptr), dim);

        PPEHeader *h_temp = reinterpret_cast<PPEHeader *>(seed_buffer);
        h_temp->set_precision(static_cast<uint32_t>(bits));
        h_temp->set_label(static_cast<uint64_t>(label));

        addPoint(seed_buffer, label, replace_deleted);

        hnswlib::tableint assigned_internal = static_cast<hnswlib::tableint>(-1);
        {
            std::unique_lock<std::mutex> lock(this->label_lookup_lock);
            auto it = this->label_lookup_.find(label);
            if (it == this->label_lookup_.end())
                throw std::runtime_error("PPHNSW::addQuantizedPoint: label not found after insertion");
            assigned_internal = it->second;
        }

        char *dst = this->getDataByInternalId(assigned_internal);
        PPEHeader *dst_h = reinterpret_cast<PPEHeader *>(dst);
        dst_h->set_precision(static_cast<uint32_t>(bits));
        dst_h->set_label(static_cast<uint64_t>(label));

        if (pomai_arena_)
        {
            const char *vec_src = dst + sizeof(PPEHeader);
            char *blob_hdr = pomai_arena_->alloc_blob(static_cast<uint32_t>(payload_bytes));
            if (blob_hdr)
            {
                std::memcpy(blob_hdr + sizeof(uint32_t), vec_src, payload_bytes);
                uint64_t offset = pomai_arena_->offset_from_blob_ptr(blob_hdr);
                uint64_t *payload_dest = reinterpret_cast<uint64_t *>(dst + sizeof(PPEHeader));
                pomai::ai::atomic_utils::atomic_store_u64(payload_dest, offset);
                dst_h->atomic_set_flags(PPE_FLAG_INDIRECT);

                std::clog << "[PPHNSW] addQuantizedPoint: moved quantized payload to arena internal=" << assigned_internal
                          << " offset=" << offset << " label=" << label << "\n";
            }
            else
            {
                uint32_t len32 = static_cast<uint32_t>(payload_bytes);
                std::vector<char> tmpbuf(sizeof(uint32_t) + payload_bytes + 1);
                std::memcpy(tmpbuf.data(), &len32, sizeof(uint32_t));
                std::memcpy(tmpbuf.data() + sizeof(uint32_t), vec_src, payload_bytes);
                tmpbuf[sizeof(uint32_t) + payload_bytes] = '\0';

                uint64_t remote_id = pomai_arena_->demote_blob_data(tmpbuf.data(), static_cast<uint32_t>(tmpbuf.size()));
                if (remote_id == 0)
                {
                    throw std::runtime_error("PomaiArena alloc_blob and demote failed in addQuantizedPoint");
                }

                uint64_t *payload_dest = reinterpret_cast<uint64_t *>(dst + sizeof(PPEHeader));
                pomai::ai::atomic_utils::atomic_store_u64(payload_dest, remote_id);
                dst_h->atomic_set_flags(PPE_FLAG_INDIRECT | PPE_FLAG_REMOTE);

                std::clog << "[PPHNSW] addQuantizedPoint: demoted quantized payload internal=" << assigned_internal
                          << " remote_id=" << remote_id << " label=" << label << "\n";
            }
        }
    }

    // --- restorePPEHeaders: when moving inline payload into arena / demote ensure atomic writes ---
    template <typename dist_t>
    void PPHNSW<dist_t>::restorePPEHeaders()
    {
        if (!pomai_space_owner_)
            throw std::runtime_error("pomai_space not initialized in restorePPEHeaders");

        if (pp_pq_ && pomai_space_owner_)
            pomai_space_owner_->setPPPQ(pp_pq_.get());

        size_t num = static_cast<size_t>(this->cur_element_count.load());
        size_t full_size = pomai_space_owner_->get_data_size();

        for (size_t i = 0; i < num; ++i)
        {
            char *dst = this->getDataByInternalId(static_cast<hnswlib::tableint>(i));
            new (dst) PPEHeader();
            PPEHeader *h = reinterpret_cast<PPEHeader *>(dst);
            h->init_hints(static_cast<uint16_t>(this->M_), static_cast<uint16_t>(this->ef_));

            // set label
            {
                std::unique_lock<std::mutex> lock(this->label_lookup_lock);
                for (const auto &kv : this->label_lookup_)
                {
                    if (kv.second == static_cast<hnswlib::tableint>(i))
                    {
                        h->set_label(static_cast<uint64_t>(kv.first));
                        break;
                    }
                }
            }

            if (pomai_arena_)
            {
                const char *vec_src = dst + sizeof(PPEHeader);
                size_t vec_size = full_size - sizeof(PPEHeader);

                char *blob_hdr = pomai_arena_->alloc_blob(static_cast<uint32_t>(vec_size));
                if (blob_hdr)
                {
                    std::memcpy(blob_hdr + sizeof(uint32_t), vec_src, vec_size);
                    uint64_t offset = pomai_arena_->offset_from_blob_ptr(blob_hdr);
                    uint64_t *payload_dest = reinterpret_cast<uint64_t *>(dst + sizeof(PPEHeader));
                    pomai::ai::atomic_utils::atomic_store_u64(payload_dest, offset);
                    h->atomic_set_flags(PPE_FLAG_INDIRECT);

                    std::clog << "[PPHNSW] restorePPEHeaders: moved payload to arena internal=" << i << " offset=" << offset << "\n";
                }
                else
                {
                    uint32_t len32 = static_cast<uint32_t>(vec_size);
                    std::vector<char> tmpbuf(sizeof(uint32_t) + vec_size + 1);
                    std::memcpy(tmpbuf.data(), &len32, sizeof(uint32_t));
                    std::memcpy(tmpbuf.data() + sizeof(uint32_t), vec_src, vec_size);
                    tmpbuf[sizeof(uint32_t) + vec_size] = '\0';

                    uint64_t remote_id = pomai_arena_->demote_blob_data(tmpbuf.data(), static_cast<uint32_t>(tmpbuf.size()));
                    if (remote_id == 0)
                        throw std::runtime_error("PomaiArena alloc_blob and demote failed in restorePPEHeaders");

                    uint64_t *payload_dest = reinterpret_cast<uint64_t *>(dst + sizeof(PPEHeader));
                    pomai::ai::atomic_utils::atomic_store_u64(payload_dest, remote_id);
                    h->atomic_set_flags(PPE_FLAG_INDIRECT | PPE_FLAG_REMOTE);

                    std::clog << "[PPHNSW] restorePPEHeaders: demoted payload internal=" << i << " remote_id=" << remote_id << "\n";
                }
            }
        }
    }

    // --- Background demoter: ensure atomic payload/flag ordering on promote/demote ---
    template <typename dist_t>
    void PPHNSW<dist_t>::startBackgroundDemoter(uint64_t interval_ms, uint64_t lookahead_ns)
    {
        if (demote_running_.load(std::memory_order_acquire))
            return;

        if (!pomai_arena_)
            return;

        // Allow runtime override of promote_lookahead_ms (ms -> ns)
        uint64_t cfg_lookahead_ms = pomai::config::runtime.promote_lookahead_ms;
        if (cfg_lookahead_ms != 0)
        {
            lookahead_ns = static_cast<uint64_t>(cfg_lookahead_ms) * 1000000ULL;
            std::clog << "[PPHNSW] demoter: using runtime.promote_lookahead_ms=" << cfg_lookahead_ms << "ms -> lookahead_ns=" << lookahead_ns << "\n";
        }
        else
        {
            std::clog << "[PPHNSW] demoter: using supplied lookahead_ns=" << lookahead_ns << "\n";
        }

        // Log hot_size_limit advisory
        if (pomai::config::runtime.hot_size_limit > 0)
        {
            std::clog << "[PPHNSW] demoter: runtime.hot_size_limit=" << pomai::config::runtime.hot_size_limit << " (advisory)\n";
        }

        demote_running_.store(true, std::memory_order_release);
        demote_thread_ = std::thread([this, interval_ms, lookahead_ns]()
                                     {
                                         using namespace std::chrono;
                                         std::clog << "[PPHNSW] demoter: started (interval_ms=" << interval_ms << " lookahead_ns=" << lookahead_ns << ")\n";
                                         while (demote_running_.load(std::memory_order_acquire))
                                         {
                                             try
                                             {
                                                 int64_t now_ns = PPEHeader::now_ns();
                                                 size_t num = static_cast<size_t>(this->cur_element_count.load());
                                                 size_t full_size = pomai_space_owner_ ? pomai_space_owner_->get_data_size() : 0;

                                                 // dynamic demote threshold from config (ms -> ns)
                                                 int64_t demote_threshold_ns = static_cast<int64_t>(pomai::config::runtime.demote_threshold_ms * 1000000ULL);

                                                 for (size_t i = 0; i < num; ++i)
                                                 {
                                                     if (!demote_running_.load(std::memory_order_acquire))
                                                         break;

                                                     hnswlib::tableint iid = static_cast<hnswlib::tableint>(i);

                                                     char *dst = nullptr;
                                                     try
                                                     {
                                                         dst = this->getDataByInternalId(iid);
                                                     }
                                                     catch (...)
                                                     {
                                                         dst = nullptr;
                                                     }
                                                     if (!dst)
                                                         continue;

                                                     PPEHeader *h = reinterpret_cast<PPEHeader *>(dst);

                                                     // Snapshot flags atomically
                                                     uint32_t flags_now = h->atomic_load_flags();

                                                     // Promotion: REMOTE+INDIRECT -> attempt promote
                                                     if ((flags_now & PPE_FLAG_INDIRECT) && (flags_now & PPE_FLAG_REMOTE))
                                                     {
                                                         int64_t pred = h->predict_next_ns();
                                                         // Use lookahead_ns (configured above) to require future access horizon
                                                         if (pred > static_cast<int64_t>(now_ns + static_cast<int64_t>(lookahead_ns)))
                                                         {
                                                             // Atomic read of payload (use atomic_utils)
                                                             const uint64_t *payload_ptr_const = reinterpret_cast<const uint64_t *>(dst + sizeof(PPEHeader));
                                                             uint64_t remote_id = pomai::ai::atomic_utils::atomic_load_u64(payload_ptr_const);
                                                             if (remote_id != 0)
                                                             {
                                                                 uint64_t new_local = pomai_arena_->promote_remote(remote_id);
                                                                 if (new_local != UINT64_MAX)
                                                                 {
                                                                     uint64_t *payload_dest = reinterpret_cast<uint64_t *>(dst + sizeof(PPEHeader));
                                                                     // store new local offset atomically
                                                                     pomai::ai::atomic_utils::atomic_store_u64(payload_dest, new_local);
                                                                     // clear REMOTE flag, keep INDIRECT
                                                                     h->atomic_clear_flags(PPE_FLAG_REMOTE);

                                                                     std::clog << "[PPHNSW] demoter: promoted internal=" << i << " remote_id=" << remote_id
                                                                               << " -> new_local=" << new_local << "\n";
                                                                 }
                                                                 else
                                                                 {
                                                                     std::clog << "[PPHNSW] demoter: promote failed for internal=" << i << " remote_id=" << remote_id << "\n";
                                                                 }
                                                             }
                                                         }
                                                         continue;
                                                     }

                                                     // Demote: check predicted next access
                                                     int64_t pred = h->predict_next_ns();

                                                     // Case A: INDIRECT local -> demote to remote
                                                     if ((flags_now & PPE_FLAG_INDIRECT) && !(flags_now & PPE_FLAG_REMOTE))
                                                     {
                                                         // Use demote_threshold_ns to allow small grace window; default 0 preserves old behaviour
                                                         if (pred <= static_cast<int64_t>(now_ns + demote_threshold_ns))
                                                         {
                                                             // read local offset atomically
                                                             const uint64_t *payload_ptr_const = reinterpret_cast<const uint64_t *>(dst + sizeof(PPEHeader));
                                                             uint64_t local_off = pomai::ai::atomic_utils::atomic_load_u64(payload_ptr_const);

                                                             // sanity check
                                                             if (local_off < pomai_arena_->get_capacity_bytes())
                                                             {
                                                                 uint64_t remote_id = pomai_arena_->demote_blob(local_off);
                                                                 if (remote_id != 0)
                                                                 {
                                                                     uint64_t *payload_dest = reinterpret_cast<uint64_t *>(dst + sizeof(PPEHeader));
                                                                     pomai::ai::atomic_utils::atomic_store_u64(payload_dest, remote_id);
                                                                     h->atomic_set_flags(PPE_FLAG_REMOTE);

                                                                     std::clog << "[PPHNSW] demoter: demoted internal=" << i << " local_off=" << local_off
                                                                               << " -> remote_id=" << remote_id << "\n";
                                                                 }
                                                             }
                                                         }
                                                     }
                                                     // Case B: inline payload (not INDIRECT) -> move to arena and optionally demote
                                                     else if (!(flags_now & PPE_FLAG_INDIRECT))
                                                     {
                                                         if (pred <= static_cast<int64_t>(now_ns + demote_threshold_ns))
                                                         {
                                                             if (pomai_arena_)
                                                             {
                                                                 size_t vec_size = full_size - sizeof(PPEHeader);
                                                                 // allocate in arena
                                                                 char *blob_hdr = pomai_arena_->alloc_blob(static_cast<uint32_t>(vec_size));
                                                                 if (!blob_hdr)
                                                                 {
                                                                     // unable to allocate; skip
                                                                     continue;
                                                                 }
                                                                 // copy inline payload into arena
                                                                 std::memcpy(blob_hdr + sizeof(uint32_t), dst + sizeof(PPEHeader), vec_size);
                                                                 uint64_t off = pomai_arena_->offset_from_blob_ptr(blob_hdr);
                                                                 if (off == UINT64_MAX)
                                                                     continue;
                                                                 // publish offset atomically and set INDIRECT
                                                                 uint64_t *payload_dest = reinterpret_cast<uint64_t *>(dst + sizeof(PPEHeader));
                                                                 pomai::ai::atomic_utils::atomic_store_u64(payload_dest, off);
                                                                 h->atomic_set_flags(PPE_FLAG_INDIRECT);

                                                                 std::clog << "[PPHNSW] demoter: moved inline->arena internal=" << i << " off=" << off << "\n";

                                                                 // now attempt immediate demote to remote (best-effort)
                                                                 uint64_t remote_id = pomai_arena_->demote_blob(off);
                                                                 if (remote_id != 0)
                                                                 {
                                                                     pomai::ai::atomic_utils::atomic_store_u64(payload_dest, remote_id);
                                                                     h->atomic_set_flags(PPE_FLAG_REMOTE);

                                                                     std::clog << "[PPHNSW] demoter: demoted newly-indirect internal=" << i
                                                                               << " off=" << off << " -> remote_id=" << remote_id << "\n";
                                                                 }
                                                             }
                                                         }
                                                     }
                                                 } // for each element
                                             }
                                             catch (const std::exception &e)
                                             {
                                                 std::clog << "[PPHNSW] demoter: exception: " << e.what() << "\n";
                                             }
                                             catch (...)
                                             {
                                                 std::clog << "[PPHNSW] demoter: unknown exception\n";
                                             }

                                             std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
                                         } // while running

                                         std::clog << "[PPHNSW] demoter: stopped\n"; });
    }

    template <typename dist_t>
    void PPHNSW<dist_t>::stopBackgroundDemoter()
    {
        demote_running_.store(false, std::memory_order_release);
        if (demote_thread_.joinable())
            demote_thread_.join();
    }

    template <typename dist_t>
    size_t PPHNSW<dist_t>::estimatedMemoryUsageBytes(size_t avg_degree_multiplier) const noexcept
    {
        try
        {
            size_t cnt = elementCount();
            size_t seed_size = getSeedSize(); // PPEHeader + payload per element
            uint64_t payload_bytes = static_cast<uint64_t>(seed_size) * static_cast<uint64_t>(cnt);

            // estimate average degree (neighbors stored): use M_ as base, average degree ~= 2*M
            size_t avg_degree = std::max<size_t>(1, static_cast<size_t>(this->M_) * avg_degree_multiplier);

            // neighbor list overhead: store neighbors as int/tableint
            uint64_t neighbor_bytes = static_cast<uint64_t>(cnt) * static_cast<uint64_t>(avg_degree) * static_cast<uint64_t>(sizeof(int));

            // misc overhead: small per-element (locks, annotations, label maps)
            uint64_t misc_per_elem = 64; // conservative
            uint64_t misc_bytes = static_cast<uint64_t>(cnt) * misc_per_elem;

            uint64_t total = payload_bytes + neighbor_bytes + misc_bytes;
            return static_cast<size_t>(total);
        }
        catch (...)
        {
            return 0;
        }
    }

    template <typename dist_t>
    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>
    PPHNSW<dist_t>::searchKnnAdaptive(const void *query_data, size_t k, float panic_factor)
    {
        // clamp panic_factor to >=0
        if (panic_factor < 0.0f)
            panic_factor = 0.0f;

        // save original ef_
        int original_ef = this->ef_;

        // compute increased ef; ensure at least original_ef
        int increase = static_cast<int>(std::lround(static_cast<double>(original_ef) * static_cast<double>(panic_factor)));
        int new_ef = original_ef + increase;
        if (new_ef <= 0)
            new_ef = original_ef;

        // set ef_ to new value for this search
        this->ef_ = new_ef;

        // If pomai_space_owner_ is present, Base::searchKnn expects a seed buffer that
        // matches PomaiSpace::get_data_size() (PPEHeader + payload). Most callers pass
        // a raw payload pointer (float*). To be backward-compatible we always wrap the
        // provided payload into a temporary seed buffer: placement-new a PPEHeader and
        // copy the underlying payload bytes into the buffer before calling Base::searchKnn.
        //
        // If pomai_space_owner_ is not set we simply forward the query_data pointer.
        std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> res;
        if (this->pomai_space_owner_)
        {
            size_t full_size = this->pomai_space_owner_->get_data_size();
            size_t underlying_bytes = this->pomai_space_owner_->underlying_data_size();

            // Defensive: if full_size is zero, fall back to forwarding pointer.
            if (full_size == 0 || underlying_bytes == 0 || !query_data)
            {
                res = Base::searchKnn(query_data, k);
            }
            else
            {
                // allocate temporary seed buffer on stack if small, otherwise heap
                std::unique_ptr<char[]> tmp;
                char *seed_buf = nullptr;
                try
                {
                    tmp.reset(new char[full_size]);
                    seed_buf = tmp.get();
                }
                catch (...)
                {
                    // allocation failed; fallback to forwarding pointer (best-effort)
                    res = Base::searchKnn(query_data, k);
                    // restore ef_ before return
                    this->ef_ = original_ef;
                    return res;
                }

                // placement-new PPEHeader at start (ensure atomics initialized)
                std::memset(seed_buf, 0, sizeof(PPEHeader));
                new (seed_buf) PPEHeader();

                // copy underlying payload bytes from caller-provided query_data into seed buffer
                // We assume caller passed a payload pointer (float*). This is the compatible path.
                std::memcpy(seed_buf + sizeof(PPEHeader), query_data, underlying_bytes);

                // call base search with wrapped seed buffer
                res = Base::searchKnn(static_cast<const void *>(seed_buf), k);
            }
        }
        else
        {
            // no PomaiSpace wrapper: forward as-is
            res = Base::searchKnn(query_data, k);
        }

        // restore original ef_
        this->ef_ = original_ef;

        return res;
    }

    template <typename dist_t>
    size_t PPHNSW<dist_t>::countColdSeeds(size_t limit) const
    {
        size_t count = 0;
        size_t n = this->cur_element_count.load(std::memory_order_relaxed);
        // Duyệt qua các node (lưu ý: internal_id thường từ 0..n-1)
        for (size_t i = 0; i < n; ++i)
        {
            if (limit > 0 && count >= limit)
                break;

            // HNSW base class function to get node data pointer
            // Note: const_cast is nasty but HNSW lib api is tricky with const correctness
            char *data = const_cast<PPHNSW<dist_t> *>(this)->getDataByInternalId(i);
            if (!data)
                continue;

            const PPEHeader *h = reinterpret_cast<const PPEHeader *>(data);

            // Atomic load relaxed is enough for statistics
            uint32_t flags = atomic_utils::atomic_load_u32(const_cast<uint32_t *>(&h->flags));

            if (flags & PPE_FLAG_REMOTE)
            {
                count++;
            }
        }
        return count;
    }

    // explicit instantiation for float
    template class PPHNSW<float>;

} // namespace pomai::ai