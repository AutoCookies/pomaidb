#pragma once
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
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
            uint32_t num_nodes = static_cast<uint32_t>(offsets_.size() - 1);
            thread_local std::vector<uint8_t> visited;
            if (visited.size() < num_nodes)
                visited.resize(num_nodes);
            std::memset(visited.data(), 0, num_nodes);

            using Cand = std::pair<float, uint32_t>;
            std::vector<Cand> frontier;
            frontier.reserve(64);
            frontier.emplace_back(1.0f, start_cid);
            visited[start_cid] = 1;

            std::vector<Cand> next;
            next.reserve(256);

            std::priority_queue<std::pair<float, uint64_t>> topk; // max-heap by distance

            uint32_t hops = 0;
            uint32_t beam_width = std::max<uint32_t>(8, static_cast<uint32_t>(std::log2(std::max<uint32_t>(1, num_nodes)) * 2));
            beam_width = std::min<uint32_t>(beam_width, num_nodes);
            const float prune_factor = 0.01f;

            while (!frontier.empty() && hops++ < advisory_max_hops)
            {
                next.clear();
                float max_frontier_w = 0.0f;
                for (const auto &p : frontier)
                    if (p.first > max_frontier_w)
                        max_frontier_w = p.first;

                for (const auto &p : frontier)
                {
                    uint32_t cid = p.second;
                    scan_cb(cid, topk);

                    uint32_t start = offsets_[cid];
                    uint32_t end = offsets_[cid + 1];
                    for (uint32_t idx = start; idx < end; ++idx)
                    {
                        const auto &edge = edges_[idx];
                        uint32_t to = edge.to_cid;
                        if (to >= num_nodes)
                            continue;
                        if (visited[to])
                            continue;
                        float acc_w = p.first * edge.weight;
                        if (acc_w < max_frontier_w * prune_factor)
                            continue;
                        visited[to] = 1;
                        next.emplace_back(acc_w, to);
                    }
                }

                if (next.empty())
                    break;

                if (next.size() > beam_width)
                {
                    std::nth_element(next.begin(), next.begin() + beam_width, next.end(),
                                     [](const Cand &a, const Cand &b)
                                     { return a.first > b.first; });
                    next.resize(beam_width);
                }

                frontier.swap(next);
            }

            std::vector<std::pair<uint64_t, float>> res;
            while (!topk.empty())
            {
                res.emplace_back(topk.top().second, topk.top().first);
                topk.pop();
            }
            std::reverse(res.begin(), res.end());
            if (res.size() > k)
                res.resize(k);
            return res;
        }

    private:
        std::vector<uint32_t> offsets_;
        std::vector<EchoEdge> edges_;
    };
}