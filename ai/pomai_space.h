#pragma once
// ai/pomai_space.h
//
// PomaiSpace adapts an existing hnswlib::SpaceInterface<dist_t> (e.g. L2Space)
// to insert a small PPEHeader in front of every vector. It exposes the same
// SpaceInterface API to HNSW while ensuring that distance computations
// update PPE headers (touch).
//
// Implementation notes:
// - The distance function returned by get_dist_func() is a small trampoline
//   (static member) which expects the `dist_func_param` pointer to point to
//   the PomaiSpace instance (`this`). This allows the trampoline to call
//   the underlying space's distance function at runtime.
// - get_data_size() reports the combined size: sizeof(PPEHeader) + underlying->get_data_size().
//
// The code intentionally keeps the hot-path (PomaiDistFunc) compact.

#include "ai/ppe.h"
#include "ai/hnswlib/hnswlib.h"

#include <cstring>
#include <cassert>

namespace pomai::ai
{

    template <typename dist_t>
    class PomaiSpace : public hnswlib::SpaceInterface<dist_t>
    {
        using DISTFUNC_T = hnswlib::DISTFUNC<dist_t>;

        hnswlib::SpaceInterface<dist_t> *underlying_space_;
        size_t combined_size_; // sizeof(PPEHeader) + underlying->get_data_size()

    public:
        explicit PomaiSpace(hnswlib::SpaceInterface<dist_t> *base) noexcept
            : underlying_space_(base)
        {
            assert(base != nullptr);
            combined_size_ = sizeof(PPEHeader) + base->get_data_size();
        }

        // Expose combined size (header + vector)
        size_t get_data_size() override { return combined_size_; }

        // The trampoline distance function. `dist_func_param` must be a pointer to
        // the PomaiSpace instance (this). HNSW will pass that pointer through.
        static dist_t PomaiDistFunc(const void *pVect1, const void *pVect2, const void *dist_func_param)
        {
            // dist_func_param is PomaiSpace<dist_t>*
            auto *self = reinterpret_cast<const PomaiSpace<dist_t> *>(dist_func_param);
            // Update headers (touch). We cast-away const because a distance computation
            // semantically mutates the PPE (heuristic stats). This follows the Pomai design.
            PPEHeader *h1 = const_cast<PPEHeader *>(reinterpret_cast<const PPEHeader *>(pVect1));
            PPEHeader *h2 = const_cast<PPEHeader *>(reinterpret_cast<const PPEHeader *>(pVect2));

            // Touch both (cheap atomics)
            h1->touch_now();
            h2->touch_now();

            // Advance pointers to payload vectors (skip header)
            const void *v1 = static_cast<const char *>(pVect1) + sizeof(PPEHeader);
            const void *v2 = static_cast<const char *>(pVect2) + sizeof(PPEHeader);

            // Delegate to underlying space distance function
            DISTFUNC_T df = self->underlying_space_->get_dist_func();
            void *param = self->underlying_space_->get_dist_func_param();
            return df(v1, v2, param);
        }

        // Return trampoline pointer (no capture)
        hnswlib::DISTFUNC<dist_t> get_dist_func() override { return &PomaiDistFunc; }

        // Dist func param will be pointer to this PomaiSpace instance
        void *get_dist_func_param() override { return this; }

        // Helper for callers that need underlying space size
        size_t underlying_data_size() const noexcept { return underlying_space_->get_data_size(); }
    };

} // namespace pomai::ai