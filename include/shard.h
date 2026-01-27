#pragma once
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <atomic>
#include <thread>
#include <mutex>

#include "types.h"
#include "wal.h"
#include "seed.h"
#include "orbit_index.h"
#include "bounded_queue.h"

namespace pomai
{

    struct UpsertTask
    {
        std::vector<UpsertRequest> batch;
        bool wait_durable;
        std::promise<Lsn> done;
    };

    class Shard
    {
    public:
        Shard(std::string name, std::size_t dim, std::size_t queue_cap, std::string wal_dir);
        ~Shard();

        void Start();
        void Stop();

        std::future<Lsn> EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable);
        SearchResponse Search(const SearchRequest &req, const pomai::ai::Budget &budget) const;
        std::size_t ApproxCountUnsafe() const;

    private:
        void RunLoop();
        void PublishSnapshot();
        void TryBuildIndex(bool force_sync); // Logic background index build

        std::string name_;
        std::string wal_dir_;

        // Components
        Wal wal_;
        Seed seed_;
        BoundedQueue<UpsertTask> ingest_q_;

        // --- State Management (Thread-Safe) ---
        // Mutex bảo vệ việc tráo đổi Index/Snapshot
        mutable std::mutex state_mu_;

        // Index (Core Search Engine)
        std::shared_ptr<pomai::core::OrbitIndex> index_;

        // Snapshot (Fallback Search)
        std::shared_ptr<const SeedSnapshot> published_;

        // Worker Thread
        std::thread owner_;

        std::size_t last_index_build_size_{0};
        static constexpr size_t kPublishEvery = 1000;
        size_t batch_since_publish_{0};
    };

}