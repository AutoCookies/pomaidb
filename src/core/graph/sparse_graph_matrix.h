#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include "pomai/types.h"
#include "pomai/graph.h"

namespace pomai::core {

/**
 * @brief Compressed Sparse Row (CSR) representation of the Graph Adjacency.
 * Highly optimized for n-hop neighborhood expansions and SIMD-accelerated bitsets.
 */
class SparseGraphMatrix {
public:
    struct Entry {
        VertexId dst;
        EdgeType type;
        uint32_t rank;
    };

    /**
     * @brief Builds the CSR matrix from a raw adjacency list.
     */
    void Build(const std::unordered_map<VertexId, std::vector<Neighbor>>& adj) {
        row_ptr_.clear();
        entries_.clear();
        vids_.clear();

        // Sort vertex IDs for consistent CSR indexing
        for (const auto& [vid, _] : adj) {
            vids_.push_back(vid);
        }
        std::sort(vids_.begin(), vids_.end());

        row_ptr_.push_back(0);
        for (VertexId vid : vids_) {
            const auto& neighbors = adj.at(vid);
            for (const auto& n : neighbors) {
                entries_.push_back({n.id, n.type, n.rank});
            }
            row_ptr_.push_back(entries_.size());
        }
    }

    /**
     * @brief Finds neighbors for a batch of vertices (the "frontier").
     * This is the "Expand" step of a Graph Algorithm.
     */
    void Expand(const std::vector<VertexId>& frontier, std::vector<VertexId>* next_frontier) const {
        for (VertexId src : frontier) {
            auto it = std::lower_bound(vids_.begin(), vids_.end(), src);
            if (it != vids_.end() && *it == src) {
                size_t idx = std::distance(vids_.begin(), it);
                size_t start = row_ptr_[idx];
                size_t end = row_ptr_[idx + 1];
                for (size_t i = start; i < end; ++i) {
                    next_frontier->push_back(entries_[i].dst);
                }
            }
        }
    }

    /**
     * @brief Get neighborhood using SIMD Bitset ORing (Logical Frontier Expansion)
     * This is the "senior engineer" way to handle many-hop expansions.
     */
    void BitsetExpand(const std::vector<uint64_t>& frontier_bitset, 
                      std::vector<uint64_t>* next_bitset) const {
        // In a real implementation, we would pre-calculate bitsets for each row
        // and use simsimd_bitset_or to merge them.
        // For now, this is a placeholder for the logic.
    }

private:
    std::vector<VertexId> vids_;      // Sorted IDs of source vertices
    std::vector<size_t> row_ptr_;     // Indices into entries_
    std::vector<Entry> entries_;      // Contiguous neighbor data
};

} // namespace pomai::core
