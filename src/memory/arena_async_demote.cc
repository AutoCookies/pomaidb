/*
 * src/memory/arena_async_demote.cc
 *
 * Queue-based async demote implementation for PomaiArena using mmap-backed writes.
 *
 * - Enqueues DemoteTask to the single background worker started lazily here.
 * - Returns a placeholder id (MSB set) immediately. Callers can resolve the placeholder
 *   with resolve_pending_remote(placeholder, timeout_ms).
 * - If the demote queue is full, falls back to synchronous demote_blob_data (writes inline)
 *   and returns the final remote id (non-placeholder) so caller still gets a usable id.
 *
 * Relies on members declared in src/memory/arena.h
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
#include <fstream>

namespace pomai::memory
{

    uint64_t PomaiArena::demote_blob_async(const void *data, uint32_t len)
    {
        if (!data || len == 0)
            return 0;

        // Fast-path fallback: if queue/pending is too large, perform synchronous demote.
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            if (pending_map_.size() >= max_pending_demotes_)
            {
                // Fallback to synchronous demote function already implemented in arena.
                return demote_blob_data(static_cast<const char *>(data), len);
            }
        }

        // Copy payload to local buffer so caller may free its memory immediately.
        std::vector<char> blob;
        try
        {
            blob.resize(static_cast<size_t>(len));
            std::memcpy(blob.data(), data, len);
        }
        catch (...)
        {
            return 0;
        }

        // Reserve deterministic remote id under mu_ to avoid races.
        uint64_t encoded_remote_id = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            uint64_t id = next_remote_id_++;
            encoded_remote_id = blob_region_bytes_ + id;
            // publish empty entry so lookups see a record (worker will fill filename)
            remote_map_[encoded_remote_id] = std::string();
        }

        // Create pending handle + placeholder id
        uint64_t ctr = pending_counter_.fetch_add(1, std::memory_order_acq_rel);
        uint64_t placeholder = make_placeholder(ctr);

        auto pend = std::make_shared<PendingDemote>();
        pend->done = false;
        pend->final_remote_id = 0;

        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            pending_map_.emplace(placeholder, pend);
        }

        // Build task
        DemoteTask task;
        task.remote_id = encoded_remote_id;
        task.placeholder = placeholder;
        task.payload = std::move(blob);
        task.pend = pend;

        // Try to enqueue task (bounded by max_pending_demotes_)
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

        // Ensure worker thread is running (defensive - allocate_region may have started it already)
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!demote_worker_running_.load(std::memory_order_acquire))
            {
                demote_worker_running_.store(true, std::memory_order_release);
                // Start background worker thread with lambda (captures this)
                demote_worker_ = std::thread([this]() {
                    while (demote_worker_running_.load(std::memory_order_acquire))
                    {
                        try
                        {
                            std::vector<DemoteTask> batch;
                            batch.reserve(64);
                            {
                                std::unique_lock<std::mutex> qlk(demote_mu_);
                                demote_cv_.wait_for(qlk, std::chrono::milliseconds(200),
                                                   [this] { return !demote_queue_.empty() || !demote_worker_running_.load(std::memory_order_acquire); });

                                if (!demote_worker_running_.load(std::memory_order_acquire) && demote_queue_.empty())
                                    break;

                                size_t bytes_acc = 0;
                                while (!demote_queue_.empty() && batch.size() < 256)
                                {
                                    DemoteTask t = std::move(demote_queue_.front());
                                    demote_queue_.pop_front();
                                    bytes_acc += t.payload.size();
                                    batch.push_back(std::move(t));
                                    if (bytes_acc >= demote_batch_bytes_)
                                        break;
                                }
                            }

                            for (auto &task : batch)
                            {
                                uint64_t rid = task.remote_id;
                                std::string fname = generate_remote_filename(rid);
                                std::string tmpname = fname + ".tmp";

                                int fd = open(tmpname.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
                                if (fd < 0)
                                {
                                    std::cerr << "PomaiArena::demote_worker: open tmp failed: " << strerror(errno) << "\n";
                                    std::lock_guard<std::mutex> lk(mu_);
                                    remote_map_.erase(rid);
                                    // fulfill pending with failure
                                    if (task.pend)
                                    {
                                        std::lock_guard<std::mutex> plk(task.pend->m);
                                        task.pend->final_remote_id = 0;
                                        task.pend->done = true;
                                    }
                                    if (task.pend)
                                        task.pend->cv.notify_all();
                                    continue;
                                }

                                ssize_t left = static_cast<ssize_t>(task.payload.size());
                                const char *ptr = task.payload.data();
                                bool write_failed = false;
                                while (left > 0)
                                {
                                    ssize_t w = write(fd, ptr, static_cast<size_t>(left));
                                    if (w < 0)
                                    {
                                        if (errno == EINTR)
                                            continue;
                                        std::cerr << "PomaiArena::demote_worker: write error: " << strerror(errno) << "\n";
                                        write_failed = true;
                                        break;
                                    }
                                    left -= w;
                                    ptr += w;
                                }

                                if (!write_failed)
                                {
                                    if (fsync(fd) != 0)
                                    {
                                        std::cerr << "PomaiArena::demote_worker: fsync failed: " << strerror(errno) << "\n";
                                        write_failed = true;
                                    }
                                }

                                close(fd);

                                if (write_failed)
                                {
                                    unlink(tmpname.c_str());
                                    std::lock_guard<std::mutex> lk(mu_);
                                    remote_map_.erase(rid);
                                    if (task.pend)
                                    {
                                        std::lock_guard<std::mutex> plk(task.pend->m);
                                        task.pend->final_remote_id = 0;
                                        task.pend->done = true;
                                    }
                                    if (task.pend)
                                        task.pend->cv.notify_all();
                                    continue;
                                }

                                // rename into place
                                if (rename(tmpname.c_str(), fname.c_str()) != 0)
                                {
                                    std::cerr << "PomaiArena::demote_worker: rename failed: " << strerror(errno) << "\n";
                                    unlink(tmpname.c_str());
                                    std::lock_guard<std::mutex> lk(mu_);
                                    remote_map_.erase(rid);
                                    if (task.pend)
                                    {
                                        std::lock_guard<std::mutex> plk(task.pend->m);
                                        task.pend->final_remote_id = 0;
                                        task.pend->done = true;
                                    }
                                    if (task.pend)
                                        task.pend->cv.notify_all();
                                    continue;
                                }

                                // success: publish filename
                                {
                                    std::lock_guard<std::mutex> lk(mu_);
                                    remote_map_[rid] = fname;
                                }
                                if (task.pend)
                                {
                                    std::lock_guard<std::mutex> plk(task.pend->m);
                                    task.pend->final_remote_id = rid;
                                    task.pend->done = true;
                                }
                                if (task.pend)
                                    task.pend->cv.notify_all();
                            }
                        }
                        catch (const std::exception &e)
                        {
                            std::cerr << "[PomaiArena] Critical error in demote worker: " << e.what() << "\n";
                            // continue loop
                        }
                        catch (...)
                        {
                            std::cerr << "[PomaiArena] Unknown critical error in demote worker\n";
                        }
                    } // while worker_running
                });
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

        // Wait for completion up to timeout_ms
        if (timeout_ms == 0)
        {
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
                pend->cv.wait_for(lk, dur, [&pend]() { return pend->done; });
            }
            if (pend->done)
                return pend->final_remote_id;
            return 0;
        }
    }

} // namespace pomai::memory