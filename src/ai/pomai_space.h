#pragma once
// ai/pomai_space.h
// Pomai wrapper around an underlying SpaceInterface that also stores PPEHeader
// before the payload. This version implements the hnswlib::SpaceInterface
// virtual API (get_data_size/get_dist_func/get_dist_func_param) so it can be
// constructed and used by HierarchicalNSW. It also supports PPPQ-based
// approximate distances when both operands are quantized.

#include "src/ai/hnswlib/hnswalg.h"
#include "src/ai/ppe.h"
#include "src/ai/pppq.h"
#include "src/memory/arena.h"

#include <memory>
#include <cstring>
#include <limits>
#include <type_traits>

namespace pomai::ai
{

    template <typename dist_t>
    class PomaiSpace : public hnswlib::SpaceInterface<dist_t>
    {
    public:
        PomaiSpace(hnswlib::SpaceInterface<dist_t> *underlying) noexcept
            : underlying_(underlying), pp_pq_(nullptr), arena_(nullptr)
        {
        }

        ~PomaiSpace() = default;

        // Allow PPHNSW to attach PPPQ instance (non-owning pointer).
        void setPPPQ(PPPQ *pq) noexcept { pp_pq_ = pq; }

        // Allow PPHNSW to attach a PomaiArena (used by PPHNSW for indirect payloads).
        void set_arena(pomai::memory::PomaiArena *arena) noexcept { arena_ = arena; }

        // ---- hnswlib::SpaceInterface overrides ----
        // Note: the original hnswlib SpaceInterface declares:
        //   virtual size_t get_data_size() = 0;
        //   virtual DISTFUNC<dist_t> get_dist_func() = 0;
        //   virtual void *get_dist_func_param() = 0;
        //
        // We implement those so PomaiSpace is concrete.

        // Return the full element storage size (PPEHeader + underlying payload size).
        size_t get_data_size() override
        {
            return sizeof(PPEHeader) + (underlying_ ? underlying_->get_data_size() : 0);
        }

        // Return the underlying payload size (without PPEHeader).
        size_t underlying_data_size() const noexcept
        {
            if (!underlying_)
                return 0;
            size_t bytes = underlying_->get_data_size();
            if (bytes % sizeof(float) == 0)
                return bytes / sizeof(float); // number of floats
            return bytes;                     // quantized / packed byte size
        }

        // Provide the distance function pointer expected by hnswlib.
        // The returned function will receive (p1, p2, param) where param is
        // the pointer returned by get_dist_func_param() (we return `this`).
        hnswlib::DISTFUNC<dist_t> get_dist_func() override
        {
            return &PomaiSpace<dist_t>::dist_wrapper;
        }

        // Provide param pointer passed to the distance function (we use `this`).
        void *get_dist_func_param() override
        {
            return static_cast<void *>(this);
        }

        // Compatibility alias: expose member that computes the distance.
        // This is the logic previously in operator() in earlier prototypes.
        dist_t distance_impl(const void *pVect1, const void *pVect2, size_t /*qty*/) const
        {
            // suppress unused-parameter warning by commenting name out above or using (void).
            // pVectX are pointers to memory layout: [PPEHeader][payload]
            const char *a = static_cast<const char *>(pVect1);
            const char *b = static_cast<const char *>(pVect2);
            if (!a || !b)
                return static_cast<dist_t>(std::numeric_limits<float>::infinity());

            const PPEHeader *ha = reinterpret_cast<const PPEHeader *>(a);
            const PPEHeader *hb = reinterpret_cast<const PPEHeader *>(b);

            // If both have precision set (i.e., quantized payload) AND we have a PPPQ attached,
            // try PPPQ approximate distance path via labels.
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
                        // PPPQ expects ids within its max range; caller must ensure mapping invariants.
                        try
                        {
                            float d = pp_pq_->approxDist(static_cast<size_t>(la), static_cast<size_t>(lb));
                            if (std::isfinite(d))
                                return static_cast<dist_t>(d);
                        }
                        catch (...)
                        {
                            // fall through to underlying float path
                        }
                    }
                }
            }

            // Default: delegate to underlying space (skip PPEHeader bytes).
            const void *va = static_cast<const void *>(a + sizeof(PPEHeader));
            const void *vb = static_cast<const void *>(b + sizeof(PPEHeader));

            if (underlying_)
            {
                // Most underlying spaces implement DISTFUNC<dist_t>(const void*, const void*, const void*)
                // The third parameter is typically a param pointer (unused here). We pass underlying's param.
                auto f = underlying_->get_dist_func();
                void *param = underlying_->get_dist_func_param();
                if (f)
                    return f(va, vb, param);
            }

            return static_cast<dist_t>(std::numeric_limits<float>::infinity());
        }

    private:
        // static wrapper matching hnswlib::DISTFUNC signature.
        // param is expected to be a (PomaiSpace<dist_t>*).
        static dist_t dist_wrapper(const void *p1, const void *p2, const void *param)
        {
            if (!param)
                return static_cast<dist_t>(std::numeric_limits<float>::infinity());
            auto self = static_cast<const PomaiSpace<dist_t> *>(const_cast<void *>(param));
            return self->distance_impl(p1, p2, 0);
        }

        // non-owning pointer to the underlying hnswlib space that computes raw float distances
        hnswlib::SpaceInterface<dist_t> *underlying_; // non-owning

        // non-owning pointer to PPPQ (may be null)
        PPPQ *pp_pq_; // non-owning pointer to PPPQ (set by PPHNSW)

        // non-owning pointer to PomaiArena (optional)
        pomai::memory::PomaiArena *arena_;
    };

} // namespace pomai::ai