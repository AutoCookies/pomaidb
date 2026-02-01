#pragma once
#include <atomic>
#include <memory>
#include <thread>
#include <utility>
#include <variant>

#include "pomai/concurrency/bounded_mpsc_queue.h"
#include "pomai/core/command.h"
#include "pomai/core/shard.h"
#include "pomai/core/stats.h"

namespace pomai::core
{

    struct ShardRuntimeOptions
    {
        std::size_t inbox_capacity{4096};
    };

    class ShardRuntime final
    {
    public:
        ShardRuntime(std::uint32_t id,
                     std::unique_ptr<Shard> shard,
                     ShardRuntimeOptions opt)
            : id_(id), shard_(std::move(shard)), inbox_(opt.inbox_capacity) {}

        ~ShardRuntime() { Stop(); }

        ShardRuntime(const ShardRuntime &) = delete;
        ShardRuntime &operator=(const ShardRuntime &) = delete;

        pomai::Status Start()
        {
            auto st = shard_->Start();
            if (!st.ok())
                return st;

            running_.store(true, std::memory_order_release);
            th_ = std::thread([this]
                              { Run(); });

            return pomai::Status::OK();
        }

        void Stop()
        {
            if (!running_.exchange(false, std::memory_order_acq_rel))
                return;

            (void)inbox_.TryPush(Command{CmdStop{}});
            inbox_.Close();

            if (th_.joinable())
                th_.join();

            shard_->Stop();
        }

        bool TryEnqueue(Command cmd)
        {
            return inbox_.TryPush(std::move(cmd));
        }

        ShardStatsSnapshot SnapshotStats() const
        {
            ShardStatsSnapshot s;
            s.shard_id = id_;
            s.queue_depth = inbox_.Size();
            s.upsert_count = shard_->UpsertCount();
            s.search_count = shard_->SearchCount();
            s.upsert_latency_us = shard_->UpsertLatencyWindow().GetP50P99();
            s.search_latency_us = shard_->SearchLatencyWindow().GetP50P99();
            s.wal_fsync_us = shard_->Wal().FsyncLatencyWindow().GetP50P99();
            return s;
        }

    private:
        void Run()
        {
            for (;;)
            {
                auto opt = inbox_.Pop();
                if (!opt.has_value())
                    break;

                Command cmd = std::move(*opt);

                if (std::holds_alternative<CmdStop>(cmd.payload))
                {
                    break;
                }
                else if (auto *u = std::get_if<CmdUpsert>(&cmd.payload))
                {
                    auto st = shard_->ApplyUpsert(std::move(u->items));
                    u->prom.set_value(std::move(st));
                }
                else if (auto *s = std::get_if<CmdSearch>(&cmd.payload))
                {
                    auto rep = shard_->ExecuteSearch(s->req);
                    s->prom.set_value(std::move(rep));
                }
                else if (auto *f = std::get_if<CmdFlush>(&cmd.payload))
                {
                    auto st = shard_->Flush();
                    f->prom.set_value(std::move(st));
                }
            }
        }

        const std::uint32_t id_;
        std::unique_ptr<Shard> shard_;

        pomai::concurrency::BoundedMpscQueue<Command> inbox_;

        std::atomic<bool> running_{false};
        std::thread th_;
    };

} // namespace pomai::core