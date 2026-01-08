// ai/pomai_hnsw.cc
//
// Out-of-line implementations for PPHNSW members.
// This file was extended to optionally register inserted labels in an attached
// PPIVF instance (coarse clusters + PQ) and to integrate PPPQ for approximate
// distance computation when both stored points are quantized (4/8-bit).
//
// The actual cluster-aware search / per-cluster HNSW refinement is not added
// here (that requires per-cluster indices). VectorStore provides an IVFPQ
// filtered search path that can leverage the PPIVF registration.

#include "src/ai/pomai_hnsw.h"
#include "src/ai/quantize.h"
#include "src/ai/space_quantized.h"
#include "src/ai/ppe.h"
#include "src/ai/pomai_space.h"
#include "src/memory/arena.h"
#include "src/ai/pp_ivf.h"
#include "src/ai/pppq.h" // PPPQ integration

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

        // If PomaiSpace doesn't yet have PPPQ pointer but we have one attached here,
        // propagate it (covers setPPPQ called before/after constructor corner cases).
        if (pp_pq_ && pomai_space_owner_)
            pomai_space_owner_->setPPPQ(pp_pq_.get());

        size_t full_size = space_ptr->get_data_size();
        size_t payload_bytes = (full_size >= sizeof(PPEHeader)) ? (full_size - sizeof(PPEHeader)) : 0;

        // If a PPIVF is attached and the payload looks like raw floats of the expected
        // dimensionality, compute cluster + pq-code now so we can register the label
        // after insertion. We only compute cluster/code here; registration happens
        // after the label has been added to the index (to ensure label existence).
        bool computed_ivf = false;
        int ivf_cluster = -1;
        const uint8_t *ivf_code = nullptr;
        std::vector<uint8_t> ivf_code_local; // hold code if we need ownership

        // keep pointer to raw float vector (if detected) so we can hand it to PPPQ after insertion
        const float *raw_fvec = nullptr;
        if (pp_ivf_)
        {
            // compute expected float count from underlying payload byte size
            size_t underlying_bytes = space_ptr->underlying_data_size();
            if (underlying_bytes > 0 && payload_bytes == underlying_bytes)
            {
                if (underlying_bytes % sizeof(float) == 0)
                {
                    size_t expected_floats = underlying_bytes / sizeof(float);
                    // check payload_bytes matches those floats
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
        std::memset(seed_buffer, 0, sizeof(PPEHeader));
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

        // Heuristic: probe nearby neighbors to compute adaptive M hint for this insertion
        uint16_t computed_M = static_cast<uint16_t>(this->M_);
        {
            size_t probe_k = std::min<size_t>(this->M_, 10);
            if (probe_k > 0)
            {
                auto nearby = Base::searchKnn(seed_buffer, probe_k);
                uint64_t sum = 0;
                size_t count = 0;
                auto tmp = nearby;
                while (!tmp.empty())
                {
                    hnswlib::labeltype lbl = tmp.top().second;
                    tmp.pop();

                    // Lookup internal id under lock (allowed here since we're a member function)
                    hnswlib::tableint iid = static_cast<hnswlib::tableint>(-1);
                    {
                        std::unique_lock<std::mutex> lock(this->label_lookup_lock);
                        auto it = this->label_lookup_.find(lbl);
                        if (it != this->label_lookup_.end())
                            iid = it->second;
                    }
                    if (iid == static_cast<hnswlib::tableint>(-1))
                        continue;
                    char *nptr = this->getDataByInternalId(iid);
                    PPEHeader *nh = reinterpret_cast<PPEHeader *>(nptr);
                    uint16_t nm = nh->get_hint_M();
                    if (nm == 0)
                        nm = static_cast<uint16_t>(this->M_);
                    sum += nm;
                    ++count;
                }
                if (count > 0)
                {
                    uint16_t avg = static_cast<uint16_t>((sum + count / 2) / count);
                    uint16_t lo = static_cast<uint16_t>(std::max<size_t>(1, this->M_ / 2));
                    uint16_t hi = static_cast<uint16_t>(std::max<size_t>(1, this->M_ * 2));
                    if (avg < lo)
                        avg = lo;
                    if (avg > hi)
                        avg = hi;
                    computed_M = avg;
                }
            }
        }

        // Temporarily set M_ for wiring
        size_t orig_M = this->M_;
        if (static_cast<size_t>(computed_M) != orig_M)
            const_cast<PPHNSW<dist_t> *>(this)->M_ = static_cast<size_t>(computed_M);

        // Insert using base algorithm (uses temporary M_)
        Base::addPoint(seed_buffer, label, replace_deleted);

        // Restore M_
        if (static_cast<size_t>(computed_M) != orig_M)
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

        // store external label inside PPEHeader for PomaiSpace/PPPQ lookup (0 reserved for unset)
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
                std::memcpy(dst + sizeof(PPEHeader), &offset, sizeof(offset));
                h->flags |= PPE_FLAG_INDIRECT;
            }
            else
            {
                // Fallback: demote direct to file (atomic write + rename) and store remote_id
                uint32_t len32 = static_cast<uint32_t>(vec_size);
                std::vector<char> tmpbuf(sizeof(uint32_t) + vec_size + 1);
                std::memcpy(tmpbuf.data(), &len32, sizeof(uint32_t));
                std::memcpy(tmpbuf.data() + sizeof(uint32_t), vec_src, vec_size);
                tmpbuf[sizeof(uint32_t) + vec_size] = '\0';

                uint64_t remote_id = pomai_arena_->demote_blob_data(tmpbuf.data(), static_cast<uint32_t>(tmpbuf.size()));
                if (remote_id == 0)
                {
                    std::cerr << "PPHNSW::addPoint: alloc_blob failed and demote_blob_data failed for label " << label << "\n";
                    throw std::runtime_error("PomaiArena alloc_blob and demote failed in addPoint");
                }

                // store remote id in payload area and mark INDIRECT+REMOTE
                std::memcpy(dst + sizeof(PPEHeader), &remote_id, sizeof(remote_id));
                h->flags |= PPE_FLAG_INDIRECT;
                h->flags |= PPE_FLAG_REMOTE;
            }
        }

        // If we computed IVFPQ cluster/code before insertion, register the label now.
        if (pp_ivf_ && computed_ivf && ivf_cluster >= 0)
        {
            try
            {
                pp_ivf_->add_label(static_cast<uint64_t>(label), ivf_cluster, ivf_code);
            }
            catch (...)
            {
                // non-fatal: registration failure should not kill insertion
            }
        }

        // If PPPQ attached and we detected a raw float vector earlier, add PQ codes for this label.
        // We call addVec after insertion to ensure any external registries that rely on label existence see it.
        if (pp_pq_ && raw_fvec)
        {
            try
            {
                // Use label as the PPPQ id (caller must ensure labels map within PPPQ max_elems)
                pp_pq_->addVec(raw_fvec, static_cast<size_t>(label));
            }
            catch (const std::exception &e)
            {
                std::cerr << "PPHNSW: PPPQ addVec failed for label " << label << ": " << e.what() << "\n";
            }
            catch (...)
            {
                std::cerr << "PPHNSW: PPPQ addVec unknown error for label " << label << "\n";
            }
        }
    }

    // --- searchKnnAdaptive uses PPE hint_ef to modulate ef_ ---
    template <typename dist_t>
    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>
    PPHNSW<dist_t>::searchKnnAdaptive(const void *query_data, size_t k, float panic_factor)
    {
        size_t original_ef = this->ef_;
        if (panic_factor > 0.0f)
        {
            size_t reduced_ef = std::max(k, (size_t)(original_ef * (1.0f - panic_factor)));
            const_cast<PPHNSW<dist_t> *>(this)->ef_ = reduced_ef;
        }

        PomaiSpace<dist_t> *space_ptr = this->pomai_space_owner_.get();
        if (space_ptr == nullptr)
            throw std::runtime_error("pomai_space not initialized");
        size_t full_size = space_ptr->get_data_size();

        // Use entrypoint_node_ hint if available
        hnswlib::tableint ep = this->enterpoint_node_;
        // Change: compare against the sentinel (tableint)-1 instead of `>= 0` to avoid unsigned-vs-0 warning.
        if (ep != static_cast<hnswlib::tableint>(-1))
        {
            char *eptr = nullptr;
            try
            {
                eptr = this->getDataByInternalId(ep);
            }
            catch (...)
            {
                eptr = nullptr;
            }
            if (eptr)
            {
                PPEHeader *hep = reinterpret_cast<PPEHeader *>(eptr);
                uint16_t hint = hep->get_hint_ef();
                if (hint > 0)
                {
                    size_t newef = std::max(this->ef_, static_cast<size_t>(hint));
                    const_cast<PPHNSW<dist_t> *>(this)->ef_ = newef;
                }
            }
        }

        std::unique_ptr<void, OperatorDeleteDeleter> buf(::operator new(full_size));
        void *seed_buffer = buf.get();
        new (seed_buffer) PPEHeader();
        std::memcpy(static_cast<char *>(seed_buffer) + sizeof(PPEHeader),
                    query_data,
                    full_size - sizeof(PPEHeader));

        auto result = Base::searchKnn(seed_buffer, k);

        // After search, bump hint_ef/hint_M on visited candidates
        {
            auto tmp = result;
            while (!tmp.empty())
            {
                hnswlib::labeltype lbl = tmp.top().second;
                tmp.pop();

                // lookup internal id under lock
                hnswlib::tableint iid = static_cast<hnswlib::tableint>(-1);
                {
                    std::unique_lock<std::mutex> lock(this->label_lookup_lock);
                    auto it = this->label_lookup_.find(lbl);
                    if (it != this->label_lookup_.end())
                        iid = it->second;
                }
                if (iid == static_cast<hnswlib::tableint>(-1))
                    continue;
                char *nptr = this->getDataByInternalId(iid);
                PPEHeader *nh = reinterpret_cast<PPEHeader *>(nptr);
                nh->bump_hint_ef(static_cast<uint16_t>(this->ef_));
                nh->bump_hint_M(static_cast<uint16_t>(this->M_));
            }
        }

        // restore ef_
        if (panic_factor > 0.0f)
            const_cast<PPHNSW<dist_t> *>(this)->ef_ = original_ef;

        return result;
    }

    // --- approxDistLabels: PPPQ helper used by external search/codepaths ---
    template <typename dist_t>
    float PPHNSW<dist_t>::approxDistLabels(hnswlib::labeltype a, hnswlib::labeltype b) const
    {
        if (!pp_pq_)
            return std::numeric_limits<float>::infinity();
        try
        {
            return pp_pq_->approxDist(static_cast<size_t>(a), static_cast<size_t>(b));
        }
        catch (...)
        {
            return std::numeric_limits<float>::infinity();
        }
    }

    // --- restorePPEHeaders initializes hint fields, sets labels and propagates PPPQ pointer ---
    template <typename dist_t>
    void PPHNSW<dist_t>::restorePPEHeaders()
    {
        if (!pomai_space_owner_)
            throw std::runtime_error("pomai_space not initialized in restorePPEHeaders");

        // If we have PPPQ attached, ensure PomaiSpace knows about it after potential resets.
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

            // Try to set label field by scanning label_lookup_ (done once during restore).
            // This is O(N) over label map but only happens during restore.
            {
                std::unique_lock<std::mutex> lock(this->label_lookup_lock);
                // Find any label that maps to internal id `i`
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
                    std::memcpy(dst + sizeof(PPEHeader), &offset, sizeof(offset));
                    h->flags |= PPE_FLAG_INDIRECT;
                }
                else
                {
                    // fallback demote to file
                    uint32_t len32 = static_cast<uint32_t>(vec_size);
                    std::vector<char> tmpbuf(sizeof(uint32_t) + vec_size + 1);
                    std::memcpy(tmpbuf.data(), &len32, sizeof(uint32_t));
                    std::memcpy(tmpbuf.data() + sizeof(uint32_t), vec_src, vec_size);
                    tmpbuf[sizeof(uint32_t) + vec_size] = '\0';

                    uint64_t remote_id = pomai_arena_->demote_blob_data(tmpbuf.data(), static_cast<uint32_t>(tmpbuf.size()));
                    if (remote_id == 0)
                    {
                        throw std::runtime_error("PomaiArena alloc_blob and demote failed in restorePPEHeaders");
                    }
                    std::memcpy(dst + sizeof(PPEHeader), &remote_id, sizeof(remote_id));
                    h->flags |= PPE_FLAG_INDIRECT;
                    h->flags |= PPE_FLAG_REMOTE;
                }
            }
        }
    }

    // --- loadIndex ---
    template <typename dist_t>
    void PPHNSW<dist_t>::loadIndex(const std::string &location, hnswlib::SpaceInterface<dist_t> *raw_space, size_t max_elements)
    {
        if (!pomai_space_owner_)
            pomai_space_owner_.reset(new PomaiSpace<dist_t>(raw_space));
        if (pomai_arena_)
            pomai_space_owner_->set_arena(pomai_arena_);
        // propagate PPPQ pointer into newly created PomaiSpace if we own one
        if (pp_pq_ && pomai_space_owner_)
            pomai_space_owner_->setPPPQ(pp_pq_.get());

        Base::loadIndex(location, pomai_space_owner_.get(), max_elements);
        restorePPEHeaders();
    }

    // --- countColdSeeds ---
    template <typename dist_t>
    size_t PPHNSW<dist_t>::countColdSeeds(uint64_t threshold_ns)
    {
        size_t cold_count = 0;
        size_t num = this->cur_element_count.load();
        size_t size = this->size_data_per_element_;
        char *ptr = this->data_level0_memory_;

        for (size_t i = 0; i < num; ++i)
        {
            PPEHeader *h = reinterpret_cast<PPEHeader *>(ptr + i * size);
            if (h->is_cold_ns(static_cast<int64_t>(threshold_ns)))
                cold_count++;
        }
        return cold_count;
    }

    // --- addQuantizedPoint ---
    template <typename dist_t>
    void PPHNSW<dist_t>::addQuantizedPoint(const float *vec, size_t dim, int bits, hnswlib::labeltype label, bool replace_deleted)
    {
        PomaiSpace<dist_t> *space_ptr = this->pomai_space_owner_.get();
        if (!space_ptr)
            throw std::runtime_error("pomai_space not initialized");

        // propagate PPPQ pointer in case not set yet
        if (pp_pq_ && pomai_space_owner_)
            pomai_space_owner_->setPPPQ(pp_pq_.get());

        size_t payload_bytes = space_ptr->underlying_data_size();

        size_t expected_payload = 0;
        if (bits == 8)
            expected_payload = dim; // bytes for 8-bit
        else if (bits == 4)
            expected_payload = (dim + 1) / 2; // packed nibbles bytes
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

        // Reuse addPoint path (it will run wiring heuristic and placement-new)
        addPoint(seed_buffer, label, replace_deleted);

        // Ensure precision and (if needed) move payload into arena
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
            // Try allocate inside arena
            char *blob_hdr = pomai_arena_->alloc_blob(static_cast<uint32_t>(payload_bytes));
            if (blob_hdr)
            {
                std::memcpy(blob_hdr + sizeof(uint32_t), vec_src, payload_bytes);
                uint64_t offset = pomai_arena_->offset_from_blob_ptr(blob_hdr);
                std::memcpy(dst + sizeof(PPEHeader), &offset, sizeof(offset));
                dst_h->flags |= PPE_FLAG_INDIRECT;
            }
            else
            {
                // fallback demote direct
                uint32_t len32 = static_cast<uint32_t>(payload_bytes);
                std::vector<char> tmpbuf(sizeof(uint32_t) + payload_bytes + 1);
                std::memcpy(tmpbuf.data(), &len32, sizeof(uint32_t));
                std::memcpy(tmpbuf.data() + sizeof(uint32_t), vec_src, payload_bytes);
                tmpbuf[sizeof(uint32_t) + payload_bytes] = '\0';

                uint64_t remote_id = pomai_arena_->demote_blob_data(tmpbuf.data(), static_cast<uint32_t>(tmpbuf.size()));
                if (remote_id == 0)
                {
                    std::cerr << "PPHNSW::addQuantizedPoint: alloc_blob failed and demote_blob_data failed for label " << label << "\n";
                    throw std::runtime_error("PomaiArena alloc_blob and demote failed in addQuantizedPoint");
                }
                std::memcpy(dst + sizeof(PPEHeader), &remote_id, sizeof(remote_id));
                dst_h->flags |= PPE_FLAG_INDIRECT;
                dst_h->flags |= PPE_FLAG_REMOTE;
            }
        }
    }

    // --- Background demoter: start/stop and worker loop (unchanged) ---

    template <typename dist_t>
    void PPHNSW<dist_t>::startBackgroundDemoter(uint64_t interval_ms, uint64_t lookahead_ns)
    {
        if (demote_running_.load(std::memory_order_acquire))
            return; // already running

        // require arena to be configured; otherwise worker has nothing to do
        if (!pomai_arena_)
            return;

        demote_running_.store(true, std::memory_order_release);
        demote_thread_ = std::thread([this, interval_ms, lookahead_ns]()
                                     {
                                         using namespace std::chrono;
                                         while (demote_running_.load(std::memory_order_acquire))
                                         {
                                             try
                                             {
                                                 // snapshot now and element count
                                                 int64_t now_ns = PPEHeader::now_ns();
                                                 size_t num = static_cast<size_t>(this->cur_element_count.load());
                                                 size_t full_size = pomai_space_owner_ ? pomai_space_owner_->get_data_size() : 0;

                                                 for (size_t i = 0; i < num; ++i)
                                                 {
                                                     if (!demote_running_.load(std::memory_order_acquire))
                                                         break;

                                                     hnswlib::tableint iid = static_cast<hnswlib::tableint>(i);

                                                     // get pointer to element payload; handle exceptions conservatively
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

                                                     // If REMOTE+INDIRECT => candidate for promotion if predicted to be hot
                                                     if ((h->flags & PPE_FLAG_INDIRECT) && (h->flags & PPE_FLAG_REMOTE))
                                                     {
                                                         int64_t pred = h->predict_next_ns();
                                                         if (pred > static_cast<int64_t>(now_ns + static_cast<int64_t>(lookahead_ns)))
                                                         {
                                                             // promote remote -> local
                                                             uint64_t remote_id = 0;
                                                             std::memcpy(&remote_id, dst + sizeof(PPEHeader), sizeof(remote_id));
                                                             if (remote_id != 0)
                                                             {
                                                                 uint64_t new_local = pomai_arena_->promote_remote(remote_id);
                                                                 if (new_local != UINT64_MAX)
                                                                 {
                                                                     std::memcpy(dst + sizeof(PPEHeader), &new_local, sizeof(new_local));
                                                                     // clear REMOTE flag
                                                                     h->flags &= ~PPE_FLAG_REMOTE;
                                                                 }
                                                             }
                                                         }
                                                         continue;
                                                     }

                                                     // If INDIRECT (local) and predicted cold -> demote
                                                     int64_t pred = h->predict_next_ns();
                                                     if ((h->flags & PPE_FLAG_INDIRECT) && !(h->flags & PPE_FLAG_REMOTE))
                                                     {
                                                         if (pred <= static_cast<int64_t>(now_ns))
                                                         {
                                                             uint64_t local_off = 0;
                                                             std::memcpy(&local_off, dst + sizeof(PPEHeader), sizeof(local_off));
                                                             // simple sanity: local_off should be < blob_region_bytes
                                                             if (local_off < pomai_arena_->get_capacity_bytes())
                                                             {
                                                                 uint64_t remote_id = pomai_arena_->demote_blob(local_off);
                                                                 if (remote_id != 0)
                                                                 {
                                                                     std::memcpy(dst + sizeof(PPEHeader), &remote_id, sizeof(remote_id));
                                                                     h->flags |= PPE_FLAG_REMOTE;
                                                                 }
                                                             }
                                                         }
                                                     }
                                                     else if (!(h->flags & PPE_FLAG_INDIRECT))
                                                     {
                                                         // Inline payload: if predicted cold, move inline -> arena (alloc_blob) then demote
                                                         if (pred <= static_cast<int64_t>(now_ns))
                                                         {
                                                             if (pomai_arena_)
                                                             {
                                                                 size_t vec_size = full_size - sizeof(PPEHeader);
                                                                 char *blob_hdr = pomai_arena_->alloc_blob(static_cast<uint32_t>(vec_size));
                                                                 if (!blob_hdr)
                                                                     continue;
                                                                 std::memcpy(blob_hdr + sizeof(uint32_t), dst + sizeof(PPEHeader), vec_size);
                                                                 uint64_t off = pomai_arena_->offset_from_blob_ptr(blob_hdr);
                                                                 if (off == UINT64_MAX)
                                                                     continue;
                                                                 // overwrite payload with offset and mark INDIRECT
                                                                 std::memcpy(dst + sizeof(PPEHeader), &off, sizeof(off));
                                                                 h->flags |= PPE_FLAG_INDIRECT;

                                                                 // now demote immediately
                                                                 uint64_t remote_id = pomai_arena_->demote_blob(off);
                                                                 if (remote_id != 0)
                                                                 {
                                                                     std::memcpy(dst + sizeof(PPEHeader), &remote_id, sizeof(remote_id));
                                                                     h->flags |= PPE_FLAG_REMOTE;
                                                                 }
                                                             }
                                                         }
                                                     }
                                                 } // for each element
                                             }
                                             catch (const std::exception &e)
                                             {
                                                 std::cerr << "PPHNSW background demoter exception: " << e.what() << "\n";
                                             }
                                             catch (...)
                                             {
                                                 std::cerr << "PPHNSW background demoter unknown exception\n";
                                             }

                                             std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
                                         } // while running
                                     });
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

    // explicit instantiation for float
    template class PPHNSW<float>;

} // namespace pomai::ai