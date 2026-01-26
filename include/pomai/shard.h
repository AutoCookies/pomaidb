#pragma once
#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "bounded_queue.h"
#include "seed.h"
#include "wal.h"

namespace pomai
{

    struct UpsertTask
    {
        std::vector<UpsertRequest> batch;
        bool wait_durable{true};
        std::promise<Lsn> done;
    };

    class Shard
    {
    public:
        // ✅ wal_dir is required now
        Shard(std::string name, std::size_t dim, std::size_t queue_cap, std::string wal_dir);
        ~Shard();

        Shard(const Shard &) = delete;
        Shard &operator=(const Shard &) = delete;

        void Start();
        void Stop();

        std::future<Lsn> EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable);

        // Safe read: uses published snapshot.
        SearchResponse Search(const SearchRequest &req) const;

        std::size_t ApproxCountUnsafe() const;

    private:
        void RunLoop();
        void PublishSnapshot();

        std::string name_;
        std::string wal_dir_;
        Wal wal_;
        Seed seed_;

        BoundedQueue<UpsertTask> ingest_q_;
        std::thread owner_;

        // Published immutable read view (RCU-like).
        std::atomic<std::shared_ptr<const Seed::Store>> published_{nullptr};

        // Publish every N batches (tune later).
        std::size_t batch_since_publish_{0};
        static constexpr std::size_t kPublishEvery = 64;
    };

} // namespace pomai
