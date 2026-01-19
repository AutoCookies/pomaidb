/*
 * src/memory/arena_async_demote.cc
 *
 * Queue-based async demote implementation for PomaiArena using mmap-backed writes.
 *
 * - Enqueues DemoteTask to the single background worker started by allocate_region().
 * - Returns a placeholder id (MSB set) immediately. Callers can resolve the placeholder
 *   with resolve_pending_remote(placeholder, timeout_ms).
 * - If the demote queue is full, falls back to synchronous demote_blob_data (writes inline)
 *   and returns the final remote id (non-placeholder) so caller still gets a usable id.
 *
 * NOTE:
 * - This file assumes PomaiArena has members used below:
 *     std::deque<DemoteTask> demote_queue_;
 *     std::mutex demote_mu_;
 *     std::condition_variable demote_cv_;
 *     std::mutex pending_mu_;
 *     std::unordered_map<uint64_t, std::shared_ptr<PendingDemote>> pending_map_;
 *     std::mutex mu_; // protects remote_map_
 *     std::unordered_map<uint64_t, std::string> remote_map_;
 *     uint64_t next_remote_id_;
 *     uint64_t blob_region_bytes_;
 *     std::string remote_dir_;
 *     std::atomic<uint64_t> pending_counter_;
 *     size_t max_pending_demotes_;
 *     bool worker_running_;
 *     std::thread demote_worker_thread_;
 *
 * - PendingDemote is expected to contain at least:
 *     std::mutex m;
 *     std::condition_variable cv;
 *     bool done = false;
 *     uint64_t final_remote_id = 0;
 *
 * If your real class layout differs, adapt names accordingly.
 */

#include "src/memory/arena.h"

#include <thread>
#include <vector>
#include <cstring>
#include <iostream>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <system_error>

namespace pomai::memory
{

    uint64_t PomaiArena::demote_blob_async(const void *data, uint32_t len)
    {
        if (!data || len == 0)
            return 0;

        // Backpressure: if too many pending demotes, fallback to synchronous demote.
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            if (pending_map_.size() >= max_pending_demotes_)
            {
                // Fallback to synchronous demote function already implemented in arena.
                return demote_blob_data(static_cast<const char *>(data), len);
            }
        }

        // Copy payload to local buffer (so caller can free its memory safely)
        std::vector<char> blob;
        blob.resize(static_cast<size_t>(len));
        std::memcpy(blob.data(), data, len);

        // Reserve deterministic remote id (encoded_remote_id = blob_region_bytes_ + id)
        uint64_t encoded_remote_id = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            uint64_t id = next_remote_id_++;
            encoded_remote_id = blob_region_bytes_ + id;
            // publish empty entry so lookups see a record (filled by worker)
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
                // Queue full: rollback pending_map_ and fallback to synchronous demote
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

        // Ensure worker thread is running (allocate_region() normally starts it; startup-check here is defensive)
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!worker_running_)
            {
                worker_running_ = true;
                demote_worker_thread_ = std::thread(&PomaiArena::demote_worker_loop, this);
            }
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

    // Background worker: dequeue demote tasks and persist using mmap (preferred) or write fallback.
    void PomaiArena::demote_worker_loop()
    {
        while (true)
        {
            DemoteTask task;
            {
                std::unique_lock<std::mutex> qlk(demote_mu_);
                demote_cv_.wait(qlk, [this]
                                { return !demote_queue_.empty() || !worker_running_; });
                if (!worker_running_ && demote_queue_.empty())
                    break;
                if (demote_queue_.empty())
                    continue;
                task = std::move(demote_queue_.front());
                demote_queue_.pop_front();
            }

            uint64_t remote_id = task.remote_id;
            uint64_t idnum = 0;
            if (remote_id >= blob_region_bytes_)
                idnum = remote_id - blob_region_bytes_;
            else
                idnum = remote_id; // defensive

            // Build filename deterministically
            std::string filename;
            {
                // sanitize remote_dir_
                std::string dir = remote_dir_;
                if (!dir.empty() && dir.back() == '/')
                    dir.pop_back();
                filename = dir + "/remote_" + std::to_string(idnum) + ".blob";
            }

            bool success = false;
            uint64_t final_remote = 0;

            // Try mmap-based write
            int fd = ::open(filename.c_str(), O_RDWR | O_CREAT, 0644);
            if (fd >= 0)
            {
                off_t want = static_cast<off_t>(task.payload.size() + sizeof(uint32_t));
                if (::ftruncate(fd, want) == 0)
                {
                    void *map = ::mmap(nullptr, static_cast<size_t>(want), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                    if (map != MAP_FAILED)
                    {
                        // Write length header + payload
                        uint32_t len32 = static_cast<uint32_t>(task.payload.size());
                        std::memcpy(static_cast<char *>(map), &len32, sizeof(len32));
                        if (len32)
                            std::memcpy(static_cast<char *>(map) + sizeof(len32), task.payload.data(), len32);

                        // Ensure durability to page cache; msync to push to disk if desired
                        if (::msync(map, static_cast<size_t>(want), MS_SYNC) == 0)
                        {
                            success = true;
                            final_remote = remote_id;
                        }
                        else
                        {
                            // msync failed, but we may still accept the file as created; keep success=false to fallback
                            std::cerr << "[PomaiArena] msync failed for " << filename << ": " << strerror(errno) << "\n";
                        }
                        ::munmap(map, static_cast<size_t>(want));
                    }
                    else
                    {
                        std::cerr << "[PomaiArena] mmap failed for " << filename << ": " << strerror(errno) << "\n";
                    }
                }
                else
                {
                    std::cerr << "[PomaiArena] ftruncate failed for " << filename << ": " << strerror(errno) << "\n";
                }
                ::close(fd);
            }
            else
            {
                std::cerr << "[PomaiArena] open failed for " << filename << ": " << strerror(errno) << "\n";
            }

            // If mmap path failed, fallback to synchronous write (write()+fsync)
            if (!success)
            {
                int fdw = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fdw >= 0)
                {
                    uint32_t len32 = static_cast<uint32_t>(task.payload.size());
                    ssize_t w = ::write(fdw, &len32, sizeof(len32));
                    if (w == sizeof(len32) && len32 > 0)
                    {
                        ssize_t w2 = ::write(fdw, task.payload.data(), len32);
                        if (w2 == static_cast<ssize_t>(len32))
                        {
                            if (::fsync(fdw) == 0)
                            {
                                success = true;
                                final_remote = remote_id;
                            }
                        }
                    }
                    ::close(fdw);
                }
            }

            // If still failed, as last resort call demote_blob_data synchronous helper (may write to different layout)
            if (!success)
            {
                try
                {
                    final_remote = demote_blob_data(task.payload.data(), static_cast<uint32_t>(task.payload.size()));
                }
                catch (...)
                {
                    final_remote = 0;
                }
            }

            // Update remote_map_ (if success) and notify pending
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (final_remote != 0)
                    remote_map_[final_remote] = filename;
                else
                    remote_map_.erase(remote_id); // indicate failure
            }

            // mark pending done and notify waiters
            if (task.pend)
            {
                {
                    std::lock_guard<std::mutex> lk(task.pend->m);
                    task.pend->final_remote_id = final_remote;
                    task.pend->done = true;
                }
                task.pend->cv.notify_all();
            }

            // remove from pending_map_
            {
                std::lock_guard<std::mutex> plk(pending_mu_);
                pending_map_.erase(task.placeholder);
            }
        } // while
    }

} // namespace pomai::memory