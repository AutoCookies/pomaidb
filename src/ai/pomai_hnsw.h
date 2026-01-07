#pragma once
// ai/pomai_hnsw.h
//
// PPHNSW: Pomai-aware wrapper around HierarchicalNSW that embeds a small PPEHeader
// in front of every stored vector and supports:
//  - optional arena-backed indirect payload storage (PomaiArena)
//  - per-node PPE heuristics (hint_M/hint_ef, access prediction)
//  - quantized insertions (4-bit / 8-bit helpers)
//  - background cold-demotion / promotion of blobs (demote to disk + lazy mmap)
//
// This header is intentionally clean and documents the public API. Implementation
// details live in ai/pomai_hnsw.cc.
//
// NOTE: This file was extended to allow attaching a PPIVF (coarse IVF + PQ)
// instance. PPHNSW will register labels in the PPIVF (label -> cluster + PQ
// code) when addPoint is called with a raw float payload. The PPIVF is optional
// and must be created/initialized externally (e.g. by VectorStore) and then set
// using setPPIVF().

#include "src/ai/hnswlib/hnswalg.h"
#include "src/ai/pomai_space.h"
#include "src/memory/arena.h"
#include "src/ai/pp_ivf.h" // optional coarse filter
#include "src/ai/pppq.h"   // PPPQ integration

#include <vector>
#include <queue>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <cstdint>
#include <limits>

namespace pomai::ai
{

    // Small helper: thread-local storage used only during construction to pass a
    // heap-allocated PomaiSpace pointer into the HierarchicalNSW base constructor.
    namespace detail
    {
        extern thread_local void *tmp_pomai_space_void;
    }

    template <typename dist_t>
    inline PomaiSpace<dist_t> *alloc_tmp_pomai_space(hnswlib::SpaceInterface<dist_t> *raw)
    {
        PomaiSpace<dist_t> *p = new PomaiSpace<dist_t>(raw);
        detail::tmp_pomai_space_void = static_cast<void *>(p);
        return p;
    }

    template <typename dist_t>
    inline PomaiSpace<dist_t> *take_tmp_pomai_space()
    {
        PomaiSpace<dist_t> *p = static_cast<PomaiSpace<dist_t> *>(detail::tmp_pomai_space_void);
        detail::tmp_pomai_space_void = nullptr;
        return p;
    }

    // PPHNSW: wrapper class. See implementation file for method definitions.
    template <typename dist_t>
    class PPHNSW : public hnswlib::HierarchicalNSW<dist_t>
    {
        using Base = hnswlib::HierarchicalNSW<dist_t>;

        // Owned PomaiSpace wrapper (this holds the PPE header size + underlying vector size).
        std::unique_ptr<PomaiSpace<dist_t>> pomai_space_owner_;

        // Optional arena used for storing payloads as blobs (indirect storage).
        PomaiArena *pomai_arena_{nullptr};

        // Optional PP-IVF (coarse clustering + PQ) instance.
        // When set, addPoint will register the label->cluster and PQ-code so other
        // components (VectorStore) can probe clusters and perform filtered scans.
        std::unique_ptr<PPIVF> pp_ivf_;

        // Optional PPPQ (product quantization compressor) instance.
        // When set, PPHNSW will call pp_pq_->addVec for raw float insertions and
        // expose approxDistLabels(...) so searches can use PQ-based distances.
        std::unique_ptr<PPPQ> pp_pq_;

        // Background demotion/promote thread state.
        std::thread demote_thread_;
        std::atomic<bool> demote_running_{false};

    public:
        // Construct with an underlying hnswlib space (e.g. L2Space). We create a
        // PomaiSpace wrapper and hand it to the base constructor, then take ownership.
        PPHNSW(hnswlib::SpaceInterface<dist_t> *raw_space, size_t max_elements, size_t M = 16, size_t ef_construction = 200)
            : Base(alloc_tmp_pomai_space<dist_t>(raw_space), max_elements, M, ef_construction)
        {
            pomai_space_owner_.reset(take_tmp_pomai_space<dist_t>());

            // conservative single-layer defaults (can be tuned)
            this->mult_ = 0.0;
            this->maxlevel_ = 0;
            pp_ivf_.reset(nullptr);
            pp_pq_.reset(nullptr);
        }

        // Ensure background thread stops on destruction.
        ~PPHNSW()
        {
            stopBackgroundDemoter();
        }

        // Attach a PomaiArena to enable indirect payload storage resolution and demotion.
        // This propagates the arena pointer into the PomaiSpace wrapper as well.
        void setPomaiArena(PomaiArena *arena)
        {
            pomai_arena_ = arena;
            if (pomai_space_owner_)
                pomai_space_owner_->set_arena(arena);
        }

        // Attach an externally created PPIVF instance. Ownership is transferred.
        // The PPIVF must be initialized (centroids, PQ configured) before calling.
        void setPPIVF(std::unique_ptr<PPIVF> ivf) noexcept
        {
            pp_ivf_ = std::move(ivf);
        }

        // Attach an externally created PPPQ instance. Ownership is transferred.
        // PPPQ should be trained/initialized before attaching.
        // We also propagate a non-owning pointer into PomaiSpace so its distance
        // operator can call into PPPQ for approximate distances when appropriate.
        void setPPPQ(std::unique_ptr<PPPQ> pq) noexcept
        {
            pp_pq_ = std::move(pq);
            if (pomai_space_owner_)
                pomai_space_owner_->setPPPQ(pp_pq_.get());
        }

        // Return pointer to PPPQ (non-owning) or nullptr if not attached.
        PPPQ *getPPPQ() noexcept { return pp_pq_.get(); }

        // ----- Primary APIs (implemented out-of-line) -----

        // Insert a datapoint whose payload layout matches PomaiSpace::get_data_size() - sizeof(PPEHeader).
        // The function placement-news a PPEHeader in index memory so atomics are initialized safely.
        void addPoint(const void *datapoint, hnswlib::labeltype label, bool replace_deleted = false) override;

        // Convenience: quantize a float vector (dim floats in [0,1]) into 'bits' (8 or 4)
        // and insert. Underlying PomaiSpace must be created to match the quantized payload size.
        void addQuantizedPoint(const float *vec, size_t dim, int bits, hnswlib::labeltype label, bool replace_deleted = false);

        // Count elements considered "cold" according to PPEHeader::is_cold_ns(threshold_ns).
        size_t countColdSeeds(uint64_t threshold_ns);

        // Adaptive search: temporarily modulate ef_ (panic_factor in [0,1]) and call Base::searchKnn.
        // Returns the same priority_queue form as Base::searchKnn.
        std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>
        searchKnnAdaptive(const void *query_data, size_t k, float panic_factor);

        // Load index from disk, wrapping the provided raw_space inside a PomaiSpace
        // and placement-new'ing PPEHeader objects for loaded elements.
        void loadIndex(const std::string &location, hnswlib::SpaceInterface<dist_t> *raw_space, size_t max_elements = 0);

        // After a raw load (or when needed), placement-new PPEHeader objects for all elements.
        // This is required because loaded bytes do not construct std::atomic fields.
        void restorePPEHeaders();

        // Return the combined seed size from PomaiSpace (PPEHeader + payload)
        size_t getSeedSize()
        {
            PomaiSpace<dist_t> *sp = pomai_space_owner_.get();
            return sp ? sp->get_data_size() : 0;
        }

        // ----- PPPQ helpers -----
        // Return approximate distance between two labels using PPPQ if attached,
        // otherwise returns +inf.
        float approxDistLabels(hnswlib::labeltype a, hnswlib::labeltype b) const;

        // ----- Background demotion / promotion control -----

        // Start the background demotion/promote worker.
        // - interval_ms: how often (milliseconds) the worker scans the index.
        // - lookahead_ns: on promotion, require predict_next_ns() > now + lookahead_ns to promote.
        // The worker is best-effort and tolerant of IO errors; it will not stop the process.
        void startBackgroundDemoter(uint64_t interval_ms = 1000, uint64_t lookahead_ns = 1000000000ULL);

        // Stop the background worker and join the thread. Safe to call multiple times.
        void stopBackgroundDemoter();

        // Query if the demoter is running.
        bool isBackgroundDemoterRunning() const noexcept { return demote_running_.load(std::memory_order_acquire); }

    private:
        // Internal helpers (implementation details in .cc). These are not part of public API,
        // but declared here for completeness; they remain private to the class.
        void demotePassOnce(uint64_t now_ns, uint64_t lookahead_ns);
    };

} // namespace pomai::ai