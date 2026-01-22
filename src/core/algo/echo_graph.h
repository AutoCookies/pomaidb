#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include "src/core/algo/blitz_kernels.h"

namespace pomai::core::algo
{
    struct EchoEdge
    {
        uint32_t to_cid;
        float weight;
    };

    class EchoGraph
    {
    public:
        void build_from_adjacency(const std::vector<std::vector<EchoEdge>> &adj)
        {
            offsets_.assign({0});
            edges_.clear();
            for (const auto &n : adj)
            {
                edges_.insert(edges_.end(), n.begin(), n.end());
                offsets_.push_back(static_cast<uint32_t>(edges_.size()));
            }
        }

        template <typename ScanCallback>
        std::vector<std::pair<uint64_t, float>> auto_navigate(
            const float *query, uint32_t start_cid, size_t k,
            uint32_t advisory_max_hops, ScanCallback &&scan_cb) const
        {
            if (offsets_.empty())
                return {};
            static thread_local std::vector<uint8_t> visited;
            uint32_t num_nodes = static_cast<uint32_t>(offsets_.size() - 1);
            if (visited.size() < num_nodes)
                visited.resize(num_nodes, 0);
            std::fill(visited.begin(), visited.begin() + num_nodes, 0);

            using Candidate = std::pair<float, uint32_t>;
            std::priority_queue<Candidate> candidate_pool;
            std::priority_queue<std::pair<float, uint64_t>> topk_results;

            candidate_pool.push({1.0f, start_cid});
            visited[start_cid] = 1;

            uint32_t hops = 0;
            const size_t beam_width = std::max(4u, static_cast<uint32_t>(std::log2(num_nodes)));

            while (!candidate_pool.empty() && hops++ < advisory_max_hops)
            {
                auto [curr_w, curr_cid] = candidate_pool.top();
                candidate_pool.pop();

                scan_cb(curr_cid, topk_results); // SIMD Scan bucket

                uint32_t start = offsets_[curr_cid], end = offsets_[curr_cid + 1];
                for (uint32_t i = start; i < end; ++i)
                {
                    const auto &edge = edges_[i];
                    __builtin_prefetch(&edges_[i + 1], 0, 1); // Prefetch láng giềng
                    if (!visited[edge.to_cid])
                    {
                        visited[edge.to_cid] = 1;
                        float acc_w = curr_w * edge.weight;
                        // [HARMONY PRUNING]: Chỉ giữ lại "tiếng vang" đủ mạnh
                        if (candidate_pool.size() < beam_width || acc_w > candidate_pool.top().first * 0.1f)
                            candidate_pool.push({acc_w, edge.to_cid});
                    }
                }
            }
            std::vector<std::pair<uint64_t, float>> res;
            while (!topk_results.empty())
            {
                res.push_back({topk_results.top().second, topk_results.top().first});
                topk_results.pop();
            }
            return res;
        }

    private:
        std::vector<uint32_t> offsets_;
        std::vector<EchoEdge> edges_;
    };
}