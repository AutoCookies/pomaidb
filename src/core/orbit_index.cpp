#include "orbit_index.h"
#include "cpu_kernels.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include <random>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <cstring>

namespace pomai::core
{
    // Thread-local storage để tránh malloc liên tục
    // Mỗi thread (Shard) sẽ có một vùng nhớ riêng để đánh dấu visited
    struct ThreadScratch
    {
        std::vector<uint32_t> visited_list; // Lưu token
        uint32_t visited_token = 0;

        void Reset(size_t size)
        {
            if (visited_list.size() < size)
            {
                visited_list.resize(size, 0);
            }
            visited_token++;
            if (visited_token == 0)
            { // Tràn số (rất hiếm), reset toàn bộ
                std::fill(visited_list.begin(), visited_list.end(), 0);
                visited_token = 1;
            }
        }

        bool IsVisited(uint32_t idx) const
        {
            return visited_list[idx] == visited_token;
        }

        void Mark(uint32_t idx)
        {
            visited_list[idx] = visited_token;
        }
    };

    static thread_local ThreadScratch scratch;

    OrbitIndex::OrbitIndex(std::size_t dim, std::size_t M, std::size_t ef_construction)
        : dim_(dim), M_(M), ef_construction_(ef_construction)
    {
    }

    void OrbitIndex::Build(const std::vector<float> &flat_data, const std::vector<Id> &flat_ids)
    {
        InsertBatch(flat_data, flat_ids);
    }

    void OrbitIndex::InsertBatch(const std::vector<float> &new_data, const std::vector<Id> &new_ids)
    {
        std::unique_lock<std::shared_mutex> lock(mu_);

        size_t batch_size = new_ids.size();
        size_t start_idx = total_vectors_;

        data_.insert(data_.end(), new_data.begin(), new_data.end());
        ids_.insert(ids_.end(), new_ids.begin(), new_ids.end());
        graph_.resize(start_idx + batch_size);

        for (size_t i = 0; i < batch_size; ++i)
        {
            graph_[start_idx + i].reserve(M_ + 1);
        }

        // RNG cho Random Links
        std::mt19937 rng(123 + total_vectors_);

        for (size_t i = 0; i < batch_size; ++i)
        {
            uint32_t curr_idx = start_idx + i;
            const float *vec_ptr = data_.data() + curr_idx * dim_;

            // Chuẩn bị visited buffer (O(1) cost)
            total_vectors_++;
            scratch.Reset(total_vectors_);

            if (curr_idx == 0)
                continue;

            // 1. Tìm hàng xóm (Exploitation)
            std::vector<uint32_t> neighbors = FindNeighbors(vec_ptr, ef_construction_, curr_idx, scratch.visited_list, scratch.visited_token);

            // 2. Thêm Random Links (Exploration - Chữa bệnh Recall thấp)
            // Bắt buộc kết nối với 2 điểm ngẫu nhiên để tạo "Wormholes"
            if (total_vectors_ > 10)
            {
                std::uniform_int_distribution<uint32_t> dist(0, total_vectors_ - 2);
                for (int r = 0; r < 2; ++r)
                {
                    uint32_t rnd = dist(rng);
                    if (rnd == curr_idx)
                        rnd = (rnd + 1) % total_vectors_;
                    neighbors.push_back(rnd);
                }
            }

            // 3. Kết nối
            size_t connect_count = std::min(neighbors.size(), M_);
            for (size_t k = 0; k < connect_count; ++k)
            {
                Connect(curr_idx, neighbors[k]);
            }
        }
    }

    void OrbitIndex::Connect(uint32_t node_a, uint32_t node_b)
    {
        if (node_a == node_b)
            return;

        // A -> B (Check duplicate thủ công để tiết kiệm memory so với set)
        bool exists_a = false;
        for (auto n : graph_[node_a])
            if (n == node_b)
            {
                exists_a = true;
                break;
            }
        if (!exists_a)
        {
            graph_[node_a].push_back(node_b);
            if (graph_[node_a].size() > M_ * 1.5)
                Prune(node_a); // Prune lười (Lazy)
        }

        // B -> A
        bool exists_b = false;
        for (auto n : graph_[node_b])
            if (n == node_a)
            {
                exists_b = true;
                break;
            }
        if (!exists_b)
        {
            graph_[node_b].push_back(node_a);
            if (graph_[node_b].size() > M_ * 1.5)
                Prune(node_b);
        }
    }

    void OrbitIndex::Prune(uint32_t node_idx)
    {
        auto &links = graph_[node_idx];
        const float *vec = data_.data() + node_idx * dim_;

        std::sort(links.begin(), links.end(), [&](uint32_t a, uint32_t b)
                  {
            float da = kernels::L2Sqr(vec, data_.data() + a * dim_, dim_);
            float db = kernels::L2Sqr(vec, data_.data() + b * dim_, dim_);
            return da < db; });

        if (links.size() > M_)
        {
            links.resize(M_);
        }
    }

    std::vector<uint32_t> OrbitIndex::FindNeighbors(const float *vec, size_t ef, uint32_t skip_id,
                                                    std::vector<uint32_t> &visited_list,
                                                    uint32_t &visited_token)
    {
        using Node = std::pair<float, uint32_t>;
        auto cmp = [](const Node &a, const Node &b)
        { return a.first > b.first; };
        std::priority_queue<Node, std::vector<Node>, decltype(cmp)> candidates(cmp);
        std::priority_queue<Node> top_k;

        size_t current_N = total_vectors_;

        // Entry points: 0 và vài điểm random
        // Điều này cực quan trọng để tránh bị kẹt ở local optima
        std::vector<uint32_t> eps;
        eps.push_back(0);
        if (current_N > 50)
        {
            eps.push_back(current_N / 2);
            eps.push_back(current_N - 1);
        }

        for (uint32_t ep : eps)
        {
            if (ep >= current_N || ep == skip_id)
                continue;
            if (visited_list[ep] == visited_token)
                continue;

            float d = kernels::L2Sqr(vec, data_.data() + ep * dim_, dim_);
            candidates.push({d, ep});
            top_k.push({d, ep});
            visited_list[ep] = visited_token;
        }

        while (!candidates.empty())
        {
            Node curr = candidates.top();
            candidates.pop();

            if (top_k.size() >= ef && curr.first > top_k.top().first)
                break;

            for (uint32_t neighbor : graph_[curr.second])
            {
                if (neighbor >= current_N || neighbor == skip_id)
                    continue;
                if (visited_list[neighbor] == visited_token)
                    continue;

                visited_list[neighbor] = visited_token;

                float dist = kernels::L2Sqr(vec, data_.data() + neighbor * dim_, dim_);

                if (top_k.size() < ef || dist < top_k.top().first)
                {
                    candidates.push({dist, neighbor});
                    top_k.push({dist, neighbor});
                    if (top_k.size() > ef)
                        top_k.pop();
                }
            }
        }

        std::vector<uint32_t> result;
        while (!top_k.empty())
        {
            result.push_back(top_k.top().second);
            top_k.pop();
        }
        return result;
    }

    SearchResponse OrbitIndex::Search(const Vector &query, const pomai::ai::Budget & /*budget*/) const
    {
        std::shared_lock<std::shared_mutex> lock(mu_);
        SearchResponse resp;
        if (total_vectors_ == 0)
            return resp;

        // Reset scratch cho Search Thread
        scratch.Reset(total_vectors_);

        size_t ef = 128;
        auto neighbors = const_cast<OrbitIndex *>(this)->FindNeighbors(
            query.data.data(), ef, (uint32_t)-1, scratch.visited_list, scratch.visited_token);

        // Sort và lấy kết quả
        std::vector<std::pair<float, uint32_t>> final_sorted;
        for (auto idx : neighbors)
        {
            float d = kernels::L2Sqr(query.data.data(), data_.data() + idx * dim_, dim_);
            final_sorted.push_back({d, idx});
        }
        std::sort(final_sorted.begin(), final_sorted.end(), [](auto &a, auto &b)
                  { return a.first < b.first; });

        size_t limit = 20;
        if (final_sorted.size() > limit)
            final_sorted.resize(limit);

        resp.items.resize(final_sorted.size());
        for (size_t i = 0; i < final_sorted.size(); ++i)
        {
            resp.items[i] = {ids_[final_sorted[i].second], -final_sorted[i].first};
        }

        return resp;
    }
}