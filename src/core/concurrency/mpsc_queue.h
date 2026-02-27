// mpsc_queue.h — Intrusive, lock-free MPSC queue for zero-allocation messaging.
// Inspired by DragonflyDB's MPSCIntrusiveQueue.
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <atomic>
#include <optional>
#include "core/concurrency/concurrency_macros.h"

namespace pomai::core::concurrency {

/**
 * MPSCQueueEntry: Base class for objects that want to be queued.
 * Being "Intrusive" means the queue pointer is stored inside the object itself,
 * which eliminates the need for 'new' or allocation when pushing to the queue.
 */
struct MPSCQueueEntry {
    std::atomic<MPSCQueueEntry*> next{nullptr};
};

/**
 * MPSCIntrusiveQueue: A lock-free, many-to-one queue.
 * Optimal for "Thread-per-core" architectures where multiple shards/threads
 * send messages to a single owner thread.
 */
template <typename T>
class MPSCIntrusiveQueue {
public:
    MPSCIntrusiveQueue() : head_(&stub_), tail_(&stub_) {}

    // No copying or moving
    MPSCIntrusiveQueue(const MPSCIntrusiveQueue&) = delete;
    MPSCIntrusiveQueue& operator=(const MPSCIntrusiveQueue&) = delete;

    /**
     * Push: Add an item to the queue. Thread-safe for multiple producers.
     * Uses atomic exchange for O(1) lock-free insertion.
     */
    void Push(MPSCQueueEntry* entry) {
        entry->next.store(nullptr, std::memory_order_relaxed);
        MPSCQueueEntry* prev = head_.exchange(entry, std::memory_order_acq_rel);
        prev->next.store(entry, std::memory_order_release);
    }

    /**
     * Pop: Retrieve an item from the queue. ONLY safe for a single consumer thread.
     * Implements the "mechanical genius" of MPSC: consumer only checks its local 'tail'.
     */
    T* Pop() {
        MPSCQueueEntry* tail = tail_;
        MPSCQueueEntry* next = tail->next.load(std::memory_order_acquire);

        if (tail == &stub_) {
            if (nullptr == next) return nullptr;
            tail_ = next;
            tail = next;
            next = next->next.load(std::memory_order_acquire);
        }

        if (next) {
            tail_ = next;
            return static_cast<T*>(tail);
        }

        MPSCQueueEntry* head = head_.load(std::memory_order_acquire);
        if (tail != head) return nullptr; // Wait for producer to finish store

        Push(&stub_);
        next = tail->next.load(std::memory_order_acquire);
        if (next) {
            tail_ = next;
            return static_cast<T*>(tail);
        }

        return nullptr;
    }

private:
    POMAI_CACHE_ALIGNED std::atomic<MPSCQueueEntry*> head_;
    MPSCQueueEntry stub_;
    POMAI_CACHE_ALIGNED MPSCQueueEntry* tail_;
};

} // namespace pomai::core::concurrency
