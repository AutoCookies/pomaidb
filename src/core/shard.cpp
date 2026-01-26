#include "pomai/shard.h"
#include <stdexcept>
#include <utility>

namespace pomai
{

    Shard::Shard(std::string name, std::size_t dim, std::size_t queue_cap, std::string wal_dir)
        : name_(std::move(name)),
          wal_dir_(std::move(wal_dir)),
          wal_(name_, wal_dir_, dim),
          seed_(dim),
          ingest_q_(queue_cap) {}

    Shard::~Shard()
    {
        Stop();
    }

    void Shard::Start()
    {
        if (owner_.joinable())
            return;

        wal_.ReplayToSeed(seed_);

        wal_.Start();

        PublishSnapshot();

        owner_ = std::thread(&Shard::RunLoop, this);
    }

    void Shard::Stop()
    {
        ingest_q_.Close();
        if (owner_.joinable())
            owner_.join();

        // ✅ stop WAL after owner thread ended (no more appends)
        wal_.Stop();
    }

    std::future<Lsn> Shard::EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        UpsertTask task;
        task.batch = std::move(batch);
        task.wait_durable = wait_durable;
        auto fut = task.done.get_future();

        if (!ingest_q_.Push(std::move(task)))
        {
            std::promise<Lsn> p;
            auto f = p.get_future();
            p.set_exception(std::make_exception_ptr(std::runtime_error("shard queue closed")));
            return f;
        }
        return fut;
    }

    void Shard::PublishSnapshot()
    {
        auto snap = seed_.MakeSnapshot();
        std::atomic_store_explicit(&published_, std::move(snap), std::memory_order_release);
    }

    SearchResponse Shard::Search(const SearchRequest &req) const
    {
        auto snap = std::atomic_load_explicit(&published_, std::memory_order_acquire);
        return Seed::SearchSnapshot(snap, req);
    }

    void Shard::RunLoop()
    {
        // publish an initial snapshot
        PublishSnapshot();

        while (true)
        {
            auto opt = ingest_q_.Pop();
            if (!opt.has_value())
                break;

            UpsertTask task = std::move(*opt);

            // ✅ WAL first
            Lsn lsn = wal_.AppendUpserts(task.batch);

            // ✅ then apply to in-memory seed
            seed_.ApplyUpserts(task.batch);

            if (task.wait_durable)
                wal_.WaitDurable(lsn);

            // periodic publish (avoid snapshot on every batch)
            if (++batch_since_publish_ >= kPublishEvery)
            {
                batch_since_publish_ = 0;
                PublishSnapshot();
            }

            task.done.set_value(lsn);
        }

        // final publish on shutdown
        PublishSnapshot();
    }

    std::size_t Shard::ApproxCountUnsafe() const
    {
        return seed_.Count();
    }

} // namespace pomai
