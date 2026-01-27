#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <shared_mutex>
#include <atomic>
#include <queue>
#include <random>
#include "types.h"
#include "whispergrain.h"

namespace pomai::core
{
    class OrbitIndex
    {
    public:
        // M=48, ef=100 là cấu hình tốt.
        explicit OrbitIndex(std::size_t dim, std::size_t M = 48, std::size_t ef_construction = 100);

        void InsertBatch(const std::vector<float> &new_data, const std::vector<Id> &new_ids);
        SearchResponse Search(const Vector &query, const pomai::ai::Budget &budget) const;

        std::size_t TotalVectors() const { return total_vectors_; }
        void Build(const std::vector<float> &flat_data, const std::vector<Id> &flat_ids);

    private:
        // Helper nhận visited_token để tránh alloc
        std::vector<uint32_t> FindNeighbors(const float *vec, size_t ef, uint32_t skip_id,
                                            std::vector<uint32_t> &visited_list,
                                            uint32_t &visited_token);

        void Connect(uint32_t node_a, uint32_t node_b);
        void Prune(uint32_t node_idx);

        std::size_t dim_;
        std::size_t M_;
        std::size_t ef_construction_;

        mutable std::shared_mutex mu_;

        std::vector<float> data_;
        std::vector<Id> ids_;
        std::atomic<size_t> total_vectors_{0};

        std::vector<std::vector<uint32_t>> graph_;
    };
}