/*
 * src/memory/arena_async_demote.cc
 *
 * Queue-based async demote implementation for PomaiArena.
 *
 * - Enqueues DemoteTask to the single background worker started by allocate_region().
 * - Returns a placeholder id (MSB set) immediately. Callers can resolve the placeholder
 * with resolve_pending_remote(placeholder, timeout_ms).
 * - If the demote queue is full, falls back to synchronous demote_blob_data (writes inline)
 * and returns the final remote id (non-placeholder) so caller still gets a usable id.
 */

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

        // [FIX CRITICAL] Backpressure & OOM Protection
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            if (pending_map_.size() >= max_pending_demotes_)
            {
                // [FIXED] Cast void* -> const char* để sửa lỗi biên dịch
                return demote_blob_data(static_cast<const char*>(data), len);
            }
        }

        // Copy payload to local buffer
        std::vector<char> blob;
        blob.resize(static_cast<size_t>(len));
        std::memcpy(blob.data(), data, len);

        // Reserve a concrete remote id now so worker can use deterministic filename
        uint64_t encoded_remote_id = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            uint64_t id = next_remote_id_++;
            encoded_remote_id = blob_region_bytes_ + id;
            // publish placeholder remote_map_ entry (pending)
            remote_map_[encoded_remote_id] = std::string();
        }

        // Create pending handle + placeholder id
        uint64_t ctr = pending_counter_.fetch_add(1, std::memory_order_acq_rel);
        uint64_t placeholder = make_placeholder(ctr);
        auto pend = std::make_shared<PendingDemote>();
        
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            pending_map_.emplace(placeholder, pend);
        }

        // Prepare task
        DemoteTask task;
        task.remote_id = encoded_remote_id;
        task.placeholder = placeholder;
        task.payload = std::move(blob);
        task.pend = pend;

        // Try to enqueue to demote_queue_ (bounded)
        {
            std::lock_guard<std::mutex> qlk(demote_mu_);
            if (demote_queue_.size() >= max_pending_demotes_)
            {
                // Queue full: fallback to synchronous demote
                {
                    std::lock_guard<std::mutex> plk(pending_mu_);
                    pending_map_.erase(placeholder);
                }
                uint64_t final_remote = demote_blob_data(task.payload.data(), static_cast<uint32_t>(task.payload.size()));
                return final_remote;
            }
            demote_queue_.push_back(std::move(task));
            demote_cv_.notify_one();
        }

        // Return placeholder to caller for later resolution
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