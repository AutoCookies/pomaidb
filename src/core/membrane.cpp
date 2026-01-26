#include "pomai/membrane.h"
#include <stdexcept>
#include <algorithm>
#include <future>
#include <thread>


namespace pomai
{

    MembraneRouter::MembraneRouter(std::vector<std::unique_ptr<Shard>> shards)
        : shards_(std::move(shards))
    {
        if (shards_.empty())
            throw std::runtime_error("must have at least 1 shard");
    }

    void MembraneRouter::Start()
    {
        for (auto &s : shards_)
            s->Start();
    }

    void MembraneRouter::Stop()
    {
        for (auto &s : shards_)
            s->Stop();
    }

    std::size_t MembraneRouter::PickShard(Id id) const
    {
        return static_cast<std::size_t>(id % shards_.size());
    }

    std::future<Lsn> MembraneRouter::Upsert(Id id, Vector vec, bool wait_durable)
    {
        UpsertRequest r;
        r.id = id;
        r.vec = std::move(vec);
        std::vector<UpsertRequest> batch;
        batch.push_back(std::move(r));
        return UpsertBatch(std::move(batch), wait_durable);
    }

    std::future<Lsn> MembraneRouter::UpsertBatch(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        if (batch.empty())
        {
            std::promise<Lsn> p;
            auto f = p.get_future();
            p.set_value(0);
            return f;
        }

        // Partition by shard
        std::vector<std::vector<UpsertRequest>> parts(shards_.size());
        for (auto &r : batch)
        {
            parts[PickShard(r.id)].push_back(std::move(r));
        }

        // Enqueue non-empty parts
        std::vector<std::future<Lsn>> futs;
        futs.reserve(shards_.size());

        for (std::size_t i = 0; i < parts.size(); ++i)
        {
            if (!parts[i].empty())
            {
                futs.push_back(shards_[i]->EnqueueUpserts(std::move(parts[i]), wait_durable));
            }
        }

        // Aggregate future
        std::promise<Lsn> done;
        auto out = done.get_future();

        std::thread([futs = std::move(futs), done = std::move(done)]() mutable
                    {
    Lsn max_lsn = 0;
    try {
      for (auto& f : futs) {
        Lsn l = f.get();
        if (l > max_lsn) max_lsn = l;
      }
      done.set_value(max_lsn);
    } catch (...) {
      done.set_exception(std::current_exception());
    } })
            .detach();

        return out;
    }

    std::size_t MembraneRouter::TotalApproxCountUnsafe() const
    {
        std::size_t sum = 0;
        for (const auto &s : shards_)
            sum += s->ApproxCountUnsafe();
        return sum;
    }

    SearchResponse MembraneRouter::Search(const SearchRequest &req) const
    {
        // Parallelize across shards
        std::vector<std::future<SearchResponse>> futs;
        futs.reserve(shards_.size());

        for (const auto &s : shards_)
        {
            futs.push_back(std::async(std::launch::async, [&req, shard = s.get()]
                                      { return shard->Search(req); }));
        }

        // Merge (simple: concatenate then take topK by score)
        std::vector<SearchResultItem> all;
        for (auto &f : futs)
        {
            auto r = f.get();
            all.insert(all.end(), r.items.begin(), r.items.end());
        }

        std::sort(all.begin(), all.end(), [](const auto &a, const auto &b)
                  { return a.score > b.score; });
        if (all.size() > req.topk)
            all.resize(req.topk);

        SearchResponse out;
        out.items = std::move(all);
        return out;
    }

} // namespace pomai
