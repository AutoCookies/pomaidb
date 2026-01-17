/*
 * src/ai/pomai_space.h
 *
 * PomaiSpace: Unified distance evaluation with Atomic Membrane protection.
 * Refactored to use pomai::config::StorageLayout for explicit memory offsets.
 */

#pragma once

#include "src/ai/hnswlib/hnswalg.h"
#include "src/ai/ppe.h"
#include "src/ai/pppq.h"
#include "src/memory/arena.h"
#include "src/ai/atomic_utils.h"
#include "src/core/config.h" // [ADDED] Include config for StorageLayout

#include <memory>
#include <cstring>
#include <limits>
#include <type_traits>
#include <iostream>
#include <cmath>

namespace pomai::ai
{
    // [ADDED] Alias for clean access
    using StoreLayout = pomai::config::StorageLayout;

    template <typename dist_t>
    class PomaiSpace : public hnswlib::SpaceInterface<dist_t>
    {
    public:
        PomaiSpace(hnswlib::SpaceInterface<dist_t> *underlying)
            : underlying_(underlying), arena_(nullptr), pp_pq_(nullptr) {}

        PomaiSpace(hnswlib::SpaceInterface<dist_t> *underlying, pomai::memory::PomaiArena *arena)
            : underlying_(underlying), arena_(arena), pp_pq_(nullptr) {}

        ~PomaiSpace() override = default;

        void set_arena(pomai::memory::PomaiArena *arena) { arena_ = arena; }
        void set_pppq(PPPQ *pq) { pp_pq_ = pq; }
        void setPPPQ(PPPQ *pq) { pp_pq_ = pq; }

        size_t get_data_size() override
        {
            return sizeof(PPEHeader) + (underlying_ ? underlying_->get_data_size() : 0);
        }

        size_t underlying_data_size() const noexcept
        {
            return underlying_ ? underlying_->get_data_size() : 0;
        }

        hnswlib::DISTFUNC<dist_t> get_dist_func() override
        {
            return dist_wrapper;
        }

        void *get_dist_func_param() override
        {
            return this;
        }

        dist_t distance_impl(const void *p1, const void *p2, size_t /*flags*/) const
        {
            const PPEHeader *ha = static_cast<const PPEHeader *>(p1);
            const PPEHeader *hb = static_cast<const PPEHeader *>(p2);

            // 1. PPPQ Fast-Path
            if (pp_pq_)
            {
                uint32_t pa = ha->get_precision();
                uint32_t pb = hb->get_precision();

                if (pa != 0 && pb != 0 && pa == pb)
                {
                    uint64_t la = ha->get_label();
                    uint64_t lb = hb->get_label();
                    if (la != 0 && lb != 0)
                    {
                        try
                        {
                            float d = pp_pq_->approxDist(static_cast<size_t>(la), static_cast<size_t>(lb));
                            if (std::isfinite(d))
                                return static_cast<dist_t>(d);
                        }
                        catch (...)
                        {
                        }
                    }
                }
            }

            // 2. Resolve Payloads
            const float *v1 = resolve_payload(p1);
            const float *v2 = resolve_payload(p2);

            // 3. Safety Check: If data is physically unreachable, return Infinity.
            if (!v1 || !v2)
            {
                return std::numeric_limits<dist_t>::max();
            }

            return underlying_->get_dist_func()(v1, v2, underlying_->get_dist_func_param());
        }

    private:
        const float *resolve_payload(const void *ptr) const
        {
            const PPEHeader *header = static_cast<const PPEHeader *>(ptr);
            const char *next_ptr = reinterpret_cast<const char *>(header) + sizeof(PPEHeader);

            // Take an atomic snapshot of (flags, payload) to avoid races where a reader
            // could see INDIRECT while payload==0.
            uint32_t flags = 0;
            uint64_t off_or_remote = 0;
            bool ok = header->atomic_snapshot_payload_and_flags(reinterpret_cast<const uint64_t *>(next_ptr), flags, off_or_remote);
            if (!ok)
                return nullptr;

            if (flags & PPE_FLAG_INDIRECT)
            {
                if (off_or_remote == 0)
                    return nullptr; // not published / invalid

                if (!arena_)
                    return nullptr;

                // 3. Try Direct Map / Lazy Load
                const char *blob_hdr = arena_->blob_ptr_from_offset_for_map(off_or_remote);
                if (blob_hdr)
                {
                    // [CHANGED] Use constant from config
                    return reinterpret_cast<const float *>(blob_hdr + StoreLayout::BLOB_HEADER_BYTES);
                }

                // 4. Fallback: Pending Demote Resolution
                uint64_t resolved = arena_->resolve_pending_remote(off_or_remote);
                if (resolved != 0)
                {
                    blob_hdr = arena_->blob_ptr_from_offset_for_map(resolved);
                    if (blob_hdr)
                        // [CHANGED] Use constant from config
                        return reinterpret_cast<const float *>(blob_hdr + StoreLayout::BLOB_HEADER_BYTES);
                    off_or_remote = resolved;
                }

                // 5. Fallback: Explicit Promote (Blocking IO)
                uint64_t new_local = arena_->promote_remote(off_or_remote);
                if (new_local != UINT64_MAX)
                {
                    blob_hdr = arena_->blob_ptr_from_offset_for_map(new_local);
                    if (blob_hdr)
                        // [CHANGED] Use constant from config
                        return reinterpret_cast<const float *>(blob_hdr + StoreLayout::BLOB_HEADER_BYTES);
                }

                return nullptr;
            }

            return reinterpret_cast<const float *>(next_ptr);
        }

        static dist_t dist_wrapper(const void *p1, const void *p2, const void *param)
        {
            if (!param)
                return std::numeric_limits<dist_t>::max();
            auto self = static_cast<const PomaiSpace<dist_t> *>(const_cast<void *>(param));
            return self->distance_impl(p1, p2, 0);
        }

        hnswlib::SpaceInterface<dist_t> *underlying_;
        pomai::memory::PomaiArena *arena_;
        PPPQ *pp_pq_;
    };

} // namespace pomai::ai