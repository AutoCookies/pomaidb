#pragma once
#include "ai/hnswlib/hnswalg.h"
#include "pomai_space.h"
#include <vector>
#include <iostream>

namespace pomai::ai
{

    template <typename dist_t>
    class PPHNSW : public hnswlib::HierarchicalNSW<dist_t>
    {
        using Base = hnswlib::HierarchicalNSW<dist_t>;
        PomaiSpace<dist_t> pomai_space_;

    public:
        PPHNSW(hnswlib::SpaceInterface<dist_t> *raw_space, size_t max_elements, size_t M = 16, size_t ef_construction = 200)
            : pomai_space_(raw_space),
              Base(&pomai_space_, max_elements, M, ef_construction)
        {
            // Temporary, conservative integration mode:
            // Force single-layer HNSW (every node level == 0). This avoids
            // exercising the multi-level wiring code path until the PPEHeader
            // storage/layout is reconciled with hnswlib internals.
            //
            // Rationale: getRandomLevel(mult_) uses 'mult_' to scale -log(r).
            // Setting mult_==0 makes all random levels zero.
            this->mult_ = 0.0;
            this->maxlevel_ = 0;
            // Ensure entrypoint_node_ stays valid semantics for single-layer.
            // (HierarchicalNSW will set enterpoint_node_ on first insertion.)
            // No further changes needed for now.
        }

        void addPoint(const void *datapoint, hnswlib::labeltype label, bool replace_deleted = false) override
        {
            size_t full_size = pomai_space_.get_data_size();
            void *seed_buffer = alloca(full_size);

            PPEHeader *header = new (seed_buffer) PPEHeader();
            header->last_access_ns = 0;
            header->ema_interval_ns = 0;
            header->flags = 0;

            void *vec_dst = static_cast<char *>(seed_buffer) + sizeof(PPEHeader);
            size_t vec_size = full_size - sizeof(PPEHeader);
            std::memcpy(vec_dst, datapoint, vec_size);

            Base::addPoint(seed_buffer, label, replace_deleted);
        }

        // Count cold seeds by scanning the packed payload area (header at start of each element).
        size_t countColdSeeds(uint64_t threshold_ns)
        {
            size_t cold_count = 0;
            size_t num = this->cur_element_count.load();
            size_t size = this->size_data_per_element_;
            char *ptr = this->data_level0_memory_;

            for (size_t i = 0; i < num; ++i)
            {
                PPEHeader *h = reinterpret_cast<PPEHeader *>(ptr + i * size);
                if (h->is_cold_ns(static_cast<int64_t>(threshold_ns)))
                {
                    cold_count++;
                }
            }
            return cold_count;
        }

        // Adaptive search: temporarily modulate ef_ then call base search.
        std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>
        searchKnnAdaptive(const void *query_data, size_t k, float panic_factor)
        {
            size_t original_ef = this->ef_;
            if (panic_factor > 0.0f)
            {
                size_t reduced_ef = std::max(k, (size_t)(original_ef * (1.0f - panic_factor)));
                const_cast<PPHNSW *>(this)->ef_ = reduced_ef;
            }

            size_t full_size = pomai_space_.get_data_size();
            void *seed_buffer = alloca(full_size);
            std::memset(seed_buffer, 0, sizeof(PPEHeader));
            std::memcpy(static_cast<char *>(seed_buffer) + sizeof(PPEHeader), query_data, full_size - sizeof(PPEHeader));

            auto result = Base::searchKnn(seed_buffer, k);

            if (panic_factor > 0.0f)
            {
                const_cast<PPHNSW *>(this)->ef_ = original_ef;
            }
            return result;
        }

        // Return the actual seed payload size (PPEHeader + underlying vector bytes).
        // Non-const because PomaiSpace::get_data_size() is non-const in this integration.
        size_t getSeedSize()
        {
            return pomai_space_.get_data_size();
        }
    };

} // namespace pomai::ai