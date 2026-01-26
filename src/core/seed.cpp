#include "pomai/seed.h"
#include <cmath>
#include <queue>
#include <algorithm> 

namespace pomai
{

    Seed::Seed(std::size_t dim) : dim_(dim) {}

    void Seed::ApplyUpserts(const std::vector<UpsertRequest> &batch)
    {
        for (const auto &r : batch)
        {
            if (r.vec.data.size() != dim_)
                continue;
            store_[r.id] = r.vec.data;
        }
    }

    Seed::Snapshot Seed::MakeSnapshot() const
    {
        // Step 2: copy-on-snapshot (simple + correct). Later: immutable segments / mmap.
        return std::make_shared<const Store>(store_);
    }

    // Very simple L2 exact scan (product correctness first; performance later).
    SearchResponse Seed::SearchSnapshot(const Snapshot &snap, const SearchRequest &req)
    {
        SearchResponse resp;
        if (!snap)
            return resp;
        if (req.query.data.empty())
            return resp;

        // max-heap of best K (score = -distance)
        struct Node
        {
            float score;
            Id id;
        };
        auto cmp = [](const Node &a, const Node &b)
        { return a.score > b.score; };
        std::priority_queue<Node, std::vector<Node>, decltype(cmp)> heap(cmp);

        for (const auto &[id, v] : *snap)
        {
            if (v.size() != req.query.data.size())
                continue;
            float dist = 0.0f;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                float d = v[i] - req.query.data[i];
                dist += d * d;
            }
            float score = -dist;

            if (heap.size() < req.topk)
                heap.push({score, id});
            else if (score > heap.top().score)
            {
                heap.pop();
                heap.push({score, id});
            }
        }

        resp.items.reserve(heap.size());
        while (!heap.empty())
        {
            resp.items.push_back({heap.top().id, heap.top().score});
            heap.pop();
        }
        // currently lowest->highest; reverse to best first
        std::reverse(resp.items.begin(), resp.items.end());
        return resp;
    }

} // namespace pomai
