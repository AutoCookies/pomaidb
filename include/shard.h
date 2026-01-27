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
#include "types.h"
#include "orbit_index.h"
#include "whispergrain.h"
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
        Shard(std::string name, std::size_t dim, std::size_t queue_cap, std::string wal_dir);
        ~Shard();

        Shard(const Shard &) = delete;
        Shard &operator=(const Shard &) = delete;

        void Start();
        void Stop();

        std::future<Lsn> EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable);

        SearchResponse Search(const SearchRequest &req, const pomai::ai::Budget &budget) const;

        std::size_t ApproxCountUnsafe() const;

    private:
        void RunLoop();
        void PublishSnapshot();
        void TryBuildIndex();

        std::string name_;
        std::string wal_dir_;
        Wal wal_;
        Seed seed_;

        BoundedQueue<UpsertTask> ingest_q_;
        std::thread owner_;

        std::atomic<std::shared_ptr<const SeedSnapshot>> published_{nullptr};
        std::atomic<std::shared_ptr<pomai::core::OrbitIndex>> index_{nullptr};

        std::size_t batch_since_publish_{0};
        static constexpr std::size_t kPublishEvery = 10;

        std::size_t last_index_build_size_{0};
        static constexpr std::size_t kIndexBuildThreshold = 5000;
    };

}