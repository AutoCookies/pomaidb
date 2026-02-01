#include "pomai/core/maintenance.h"
#include "pomai/core/shard_runtime.h"

#include <chrono>
#include <thread>

namespace pomai::core
{
    void MaintenanceScheduler::Start()
    {
        if (running_.exchange(true, std::memory_order_acq_rel))
            return;

        th_ = std::thread([this]
                          { Loop(); });
    }

    void MaintenanceScheduler::Stop()
    {
        if (!running_.exchange(false, std::memory_order_acq_rel))
            return;

        if (th_.joinable())
            th_.join();
    }

    void MaintenanceScheduler::Loop()
    {
        using namespace std::chrono;

        while (running_.load(std::memory_order_acquire))
        {
            for (auto &rt : shards_)
            {
                if (!rt)
                    continue;

                const auto s = rt->SnapshotStats();

                const bool hard_limit =
                    (opt_.wal_hard_limit_bytes > 0 && s.wal_bytes >= opt_.wal_hard_limit_bytes);

                if (!hard_limit)
                {
                    if (opt_.pause_if_queue_depth_ge > 0 &&
                        s.queue_depth >= opt_.pause_if_queue_depth_ge)
                        continue;

                    if (opt_.min_interval_ms > 0 &&
                        s.ms_since_last_checkpoint < opt_.min_interval_ms)
                        continue;

                    if (opt_.wal_bytes_threshold > 0 &&
                        s.wal_bytes < opt_.wal_bytes_threshold)
                        continue;
                }
                else
                {
                    if (!opt_.force_on_hard_limit)
                        continue;
                }

                // Non-blocking: shard does checkpoint on its own thread.
                rt->RequestCheckpoint();
            }

            std::this_thread::sleep_for(milliseconds(opt_.tick_ms));
        }
    }

} // namespace pomai::core
