#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace pomai::core
{
    struct MaintenanceOptions
    {
        // Trigger checkpoint when WAL grows too large (per shard)
        std::uint64_t wal_bytes_threshold = 256ull * 1024ull * 1024ull; // 256MB

        // Minimum time between checkpoints (per shard)
        std::uint64_t min_interval_ms = 10ull * 60ull * 1000ull; // 10 minutes

        // Poll interval
        std::uint64_t tick_ms = 50;

        // Backpressure: if shard is busy, skip maintenance
        std::uint32_t pause_if_queue_depth_ge = 4096;

        // Hard cap: if WAL exceeds this, try checkpoint even under load
        std::uint64_t wal_hard_limit_bytes = 8ull * 1024ull * 1024ull * 1024ull; // 8GB
        bool force_on_hard_limit = true;
    };

    class ShardRuntime; // fwd

    class MaintenanceScheduler final
    {
    public:
        explicit MaintenanceScheduler(MaintenanceOptions opt) : opt_(opt) {}
        ~MaintenanceScheduler() { Stop(); }

        MaintenanceScheduler(const MaintenanceScheduler &) = delete;
        MaintenanceScheduler &operator=(const MaintenanceScheduler &) = delete;

        void RegisterShard(std::shared_ptr<ShardRuntime> s) { shards_.push_back(std::move(s)); }

        void Start();
        void Stop();

    private:
        void Loop();

        MaintenanceOptions opt_;
        std::atomic<bool> running_{false};
        std::thread th_;
        std::vector<std::shared_ptr<ShardRuntime>> shards_;
    };

} // namespace pomai::core
