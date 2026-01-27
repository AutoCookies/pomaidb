#include "membrane.h"
#include <stdexcept>
#include <algorithm>
#include <future>
#include <thread>
#include <chrono>
#include <vector>

namespace pomai
{

    MembraneRouter::MembraneRouter(std::vector<std::unique_ptr<Shard>> shards, pomai::server::WhisperConfig w_cfg)
        : shards_(std::move(shards)), brain_(w_cfg)
    {
        if (shards_.empty())
            throw std::runtime_error("must have at least 1 shard");
    }

    void MembraneRouter::Start()
    {
        // PARALLEL BOOT: Khởi động tất cả Shard cùng lúc
        std::vector<std::future<void>> futures;
        futures.reserve(shards_.size());

        for (auto &s : shards_)
        {
            // Launch async: Mỗi shard start trên một thread riêng biệt
            futures.push_back(std::async(std::launch::async, [&s]()
                                         { s->Start(); }));
        }

        // Chờ tất cả Shard khởi động xong trước khi cho Server nhận request
        for (auto &f : futures)
        {
            f.get();
        }
    }

    void MembraneRouter::Stop()
    {
        // Stop cũng nên song song để tắt nhanh, nhưng tuần tự cho an toàn cũng được
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

        std::vector<std::vector<UpsertRequest>> parts(shards_.size());
        for (auto &r : batch)
        {
            parts[PickShard(r.id)].push_back(std::move(r));
        }

        std::vector<std::future<Lsn>> futs;
        futs.reserve(shards_.size());

        for (std::size_t i = 0; i < parts.size(); ++i)
        {
            if (!parts[i].empty())
            {
                futs.push_back(shards_[i]->EnqueueUpserts(std::move(parts[i]), wait_durable));
            }
        }

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
        auto start = std::chrono::steady_clock::now();

        auto budget = brain_.compute_budget(false);

        std::vector<std::future<SearchResponse>> futs;
        futs.reserve(shards_.size());

        for (const auto &s : shards_)
        {
            futs.push_back(std::async(std::launch::async, [&req, &budget, shard = s.get()]
                                      { return shard->Search(req, budget); }));
        }

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

        auto end = std::chrono::steady_clock::now();
        float latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
        brain_.observe_latency(latency_ms);

        return out;
    }

}