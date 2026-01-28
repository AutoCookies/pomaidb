#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <vector>

#include "types.h"
#include "whispergrain.h"

namespace pomai::core
{

    class OrbitIndex
    {
    public:
        explicit OrbitIndex(std::size_t dim, std::size_t M = 48, std::size_t ef_construction = 200);

        // One-time build. After Build(), the index is immutable and can be searched concurrently.
        void Build(const std::vector<float> &flat_data, const std::vector<Id> &flat_ids);

        // Thread-safe after Build() (read-only).
        SearchResponse Search(const Vector &query, const pomai::ai::Budget &budget) const;

        std::size_t TotalVectors() const noexcept { return total_vectors_.load(std::memory_order_acquire); }
        std::size_t Dim() const noexcept { return dim_; }
        bool Built() const noexcept { return built_.load(std::memory_order_acquire); }

    private:
        struct Candidate
        {
            float dist;
            std::uint32_t idx;
        };

        // Build-time neighbor discovery (uses current graph and limits expansions by ef)
        std::vector<std::uint32_t> FindNeighborsBuild(const float *vec, std::uint32_t curr_idx, std::size_t ef) const;

        // Search-time neighbor discovery (limits expansions by ops_budget)
        std::vector<std::uint32_t> FindNeighborsSearch(const float *q, std::size_t ops_budget) const;

        void Connect(std::uint32_t a, std::uint32_t b);
        void Prune(std::uint32_t node);

    private:
        std::size_t dim_{0};
        std::size_t M_{48};
        std::size_t ef_construction_{200};

        std::vector<float> data_;                       // [N * dim]
        std::vector<Id> ids_;                           // [N]
        std::vector<std::vector<std::uint32_t>> graph_; // adjacency

        std::atomic<std::size_t> total_vectors_{0};
        std::atomic<bool> built_{false};
    };

} // namespace pomai::core
