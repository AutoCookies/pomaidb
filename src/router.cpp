#include "pomai/core/router.h"

namespace pomai::core
{

    Router::Router(std::vector<std::unique_ptr<ShardRuntime>> shards, RouterOptions opt)
        : shards_(std::move(shards)), opt_(opt) {}

    pomai::Status Router::Start()
    {
        for (auto &s : shards_)
        {
            auto st = s->Start();
            if (!st.ok())
                return st;
        }
        return pomai::Status::OK();
    }

    void Router::Stop()
    {
        for (auto &s : shards_)
            s->Stop();
    }

    std::uint32_t Router::PickShard(pomai::VectorId id) const
    {
        return static_cast<std::uint32_t>(id % shards_.size());
    }

    UpsertBatchResult Router::UpsertBatch(std::vector<UpsertItem> items)
    {
        UpsertBatchResult out;
        out.status = pomai::Status::OK();
        if (items.empty())
            return out;

        std::vector<std::vector<UpsertItem>> buckets(shards_.size());
        for (auto &it : items)
        {
            const std::uint32_t sid = PickShard(it.id);
            buckets[sid].push_back(std::move(it));
        }

        struct Pending
        {
            std::size_t n_items{0};
            std::future<pomai::Status> fut;
        };
        std::vector<Pending> pendings;
        pendings.reserve(shards_.size());

        for (std::size_t i = 0; i < shards_.size(); ++i)
        {
            if (buckets[i].empty())
                continue;

            std::size_t n = buckets[i].size();
            CmdUpsert cmd;
            cmd.items = std::move(buckets[i]);
            auto fut = cmd.prom.get_future();

            if (shards_[i]->TryEnqueue(Command{std::move(cmd)}))
            {
                pendings.push_back({n, std::move(fut)});
            }
            else
            {
                out.fail_count += n;
                if (out.status.ok())
                    out.status = pomai::Status::Busy("shard queue full");
            }
        }

        for (auto &p : pendings)
        {
            auto st = p.fut.get();
            if (st.ok())
            {
                out.ok_count += p.n_items;
            }
            else
            {
                out.fail_count += p.n_items;
                if (out.status.ok())
                    out.status = st;
            }
        }

        return out;
    }

    SearchReply Router::Search(const SearchRequest &req)
    {
        std::vector<std::future<SearchReply>> futs;
        futs.reserve(shards_.size());

        for (auto &s : shards_)
        {
            CmdSearch cmd;
            cmd.req = req;
            futs.push_back(cmd.prom.get_future());

            if (!s->TryEnqueue(Command{std::move(cmd)}))
            {
                SearchReply rep;
                rep.status = pomai::Status::Busy("shard queue full (search)");
                return rep;
            }
        }

        SearchReply out;
        out.status = pomai::Status::OK();

        std::vector<pomai::SearchHit> merged;
        for (auto &f : futs)
        {
            auto rep = f.get();
            if (!rep.status.ok() && out.status.ok())
                out.status = rep.status;
            MergeTopK(merged, std::move(rep.hits), req.topk);
        }

        out.hits = std::move(merged);
        return out;
    }

    pomai::Status Router::FlushAll()
    {
        for (auto &s : shards_)
        {
            CmdFlush cmd;
            auto fut = cmd.prom.get_future();
            if (!s->TryEnqueue(Command{std::move(cmd)}))
                return pomai::Status::Busy("flush busy");
            auto st = fut.get();
            if (!st.ok())
                return st;
        }
        return pomai::Status::OK();
    }

    std::vector<ShardStatsSnapshot> Router::GetStats() const
    {
        std::vector<ShardStatsSnapshot> out;
        out.reserve(shards_.size());
        for (auto &s : shards_)
        {
            out.push_back(s->SnapshotStats());
        }
        return out;
    }

} // namespace pomai::core