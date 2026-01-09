// src/memory/arena_async_demote.cc
//
// Async demote helper implementation for PomaiArena.
// Schedules a background task that calls the synchronous demote_blob_data()
// and stores the result in a PendingDemote object so callers can resolve it.
//
// Notes:
//  - This implementation is intentionally simple: it detaches a thread per request.
//    For production you'd replace this with a bounded thread-pool / task queue,
//    retries/backoff, failure handling, and robust IO semantics.

#include "src/memory/arena.h"

#include <thread>
#include <vector>
#include <cstring>
#include <iostream>
#include <chrono>

namespace pomai::memory
{

    uint64_t PomaiArena::demote_blob_async(const void *data, uint32_t len)
    {
        if (!data || len == 0)
            return 0;

        // allocate monotonic placeholder id
        uint64_t ctr = pending_counter_.fetch_add(1, std::memory_order_acq_rel);
        uint64_t placeholder = make_placeholder(ctr);

        // create pending entry
        auto pend = std::make_shared<PendingDemote>();
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            pending_map_.emplace(placeholder, pend);
        }

        // copy blob (caller buffer may be transient)
        std::vector<char> blob;
        blob.resize(static_cast<size_t>(len));
        std::memcpy(blob.data(), data, len);

        // spawn background thread (detached) to perform synchronous demote and fulfill promise
        std::thread([this, placeholder, pend, blob = std::move(blob), len]() mutable
                    {
                        uint64_t final_remote = 0;
                        try
                        {
                            // call existing synchronous demote helper
                            // demote_blob_data expects const char* and len; returns remote_id or 0 on error
                            final_remote = this->demote_blob_data(blob.data(), len);
                        }
                        catch (const std::exception &e)
                        {
                            std::cerr << "[PomaiArena] async demote exception: " << e.what() << "\n";
                            final_remote = 0;
                        }
                        catch (...)
                        {
                            std::cerr << "[PomaiArena] async demote unknown exception\n";
                            final_remote = 0;
                        }

                        // publish result into pending entry and notify waiters
                        {
                            std::lock_guard<std::mutex> lk(pend->m);
                            pend->final_remote_id = final_remote;
                            pend->done = true;
                        }
                        pend->cv.notify_all();

                        // cleanup entry from pending_map_
                        {
                            std::lock_guard<std::mutex> lk(this->pending_mu_);
                            this->pending_map_.erase(placeholder);
                        } })
            .detach();

        return placeholder;
    }

    uint64_t PomaiArena::resolve_pending_remote(uint64_t maybe_placeholder, uint64_t timeout_ms)
    {
        // If not a placeholder, return as-is.
        if (!is_placeholder_id(maybe_placeholder))
            return maybe_placeholder;

        std::shared_ptr<PendingDemote> pend;
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            auto it = pending_map_.find(maybe_placeholder);
            if (it == pending_map_.end())
            {
                // unknown placeholder - maybe already removed or invalid
                return 0;
            }
            pend = it->second;
        }

        if (!pend)
            return 0;

        // wait for completion up to timeout_ms
        if (timeout_ms == 0)
        {
            // non-blocking check
            std::lock_guard<std::mutex> lk(pend->m);
            if (pend->done)
                return pend->final_remote_id;
            return 0;
        }
        else
        {
            std::unique_lock<std::mutex> lk(pend->m);
            if (!pend->done)
            {
                auto dur = std::chrono::milliseconds(timeout_ms);
                pend->cv.wait_for(lk, dur, [&pend]()
                                  { return pend->done; });
            }
            if (pend->done)
                return pend->final_remote_id;
            return 0;
        }
    }

} // namespace pomai::memory