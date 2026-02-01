#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace pomai::core
{

    // Rolling window of latency samples (microseconds), thread-safe.
    // Writers: shard thread only (single writer in our design) but we still protect for safety.
    // Readers: /stats endpoint.
    class LatencyWindow final
    {
    public:
        explicit LatencyWindow(std::size_t cap = 4096) : cap_(cap)
        {
            buf_.reserve(cap_);
        }

        void Add(std::uint64_t us)
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (buf_.size() < cap_)
            {
                buf_.push_back(us);
            }
            else
            {
                // circular overwrite
                buf_[idx_] = us;
                idx_ = (idx_ + 1) % cap_;
                filled_ = true;
            }
        }

        struct Percentiles
        {
            std::uint64_t p50{0};
            std::uint64_t p99{0};
            std::size_t n{0};
        };

        Percentiles GetP50P99() const
        {
            std::vector<std::uint64_t> tmp;
            {
                std::lock_guard<std::mutex> lk(mu_);
                tmp = buf_;
            }
            Percentiles out;
            out.n = tmp.size();
            if (tmp.empty())
                return out;

            std::sort(tmp.begin(), tmp.end());
            auto pick = [&](double q) -> std::uint64_t
            {
                if (tmp.empty())
                    return 0;
                double pos = q * static_cast<double>(tmp.size() - 1);
                std::size_t i = static_cast<std::size_t>(pos);
                return tmp[i];
            };

            out.p50 = pick(0.50);
            out.p99 = pick(0.99);
            return out;
        }

    private:
        const std::size_t cap_;
        mutable std::mutex mu_;
        mutable std::vector<std::uint64_t> buf_;
        std::size_t idx_{0};
        bool filled_{false};
    };

    struct ShardStatsSnapshot
    {
        std::uint32_t shard_id{0};
        std::uint64_t upsert_count{0};
        std::uint64_t search_count{0};

        std::size_t queue_depth{0};

        LatencyWindow::Percentiles upsert_latency_us{};
        LatencyWindow::Percentiles search_latency_us{};
        LatencyWindow::Percentiles wal_fsync_us{};
    };

} // namespace pomai::core