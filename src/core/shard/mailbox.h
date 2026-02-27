// mailbox.h — Lock-free MPSC ring buffer for PomaiDB ShardRuntime dispatch.
//
// Phase 3 (Helio-style shared-nothing): Replaces the mutex-backed
// BoundedMpscQueue with a true lock-free ring buffer. Multiple producers
// (any calling thread) can push concurrently via a single atomic fetch_add
// on the head counter. The single consumer (shard jthread) pops without
// any lock.
//
// Design: Dmitry Vyukov / MCRingBuffer pattern.
//   - Power-of-2 capacity for mask-based wrapping.
//   - Each slot has its own `sequence` atomic to signal readiness.
//   - Producer: CAS-free — fetch_add(head) + store(slot.seq) after write.
//   - Consumer: spin on slot.seq to detect readiness, then pop.
//
// PopBlocking spins with exponential backoff then yields — suitable for
// a dedicated jthread that has no other work.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <optional>
#include <thread>

namespace pomai::core {

template <class T>
class BoundedMpscQueue {
public:
    explicit BoundedMpscQueue(std::size_t capacity)
        : mask_(RoundUpPow2(capacity) - 1),
          head_(0), tail_(0),
          buffer_(new Slot[mask_ + 1])
    {
        for (std::size_t i = 0; i <= mask_; ++i)
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
    }

    ~BoundedMpscQueue() { delete[] buffer_; }

    // Non-copyable / non-movable
    BoundedMpscQueue(const BoundedMpscQueue&) = delete;
    BoundedMpscQueue& operator=(const BoundedMpscQueue&) = delete;

    // ── Producer path (many threads) — lock-free ─────────────────────────────
    bool TryPush(T&& v) {
        std::size_t head = head_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = buffer_[head & mask_];
            std::size_t seq = slot.sequence.load(std::memory_order_acquire);
            std::intptr_t diff = static_cast<std::intptr_t>(seq) -
                                 static_cast<std::intptr_t>(head);
            if (diff == 0) {
                // Slot is ready for our turn — try to claim it
                if (head_.compare_exchange_weak(head, head + 1,
                                                std::memory_order_relaxed))
                {
                    if (closed_.load(std::memory_order_relaxed)) {
                        // Revert: mark slot as unused again
                        slot.sequence.store(head, std::memory_order_release);
                        return false;
                    }
                    slot.data = std::move(v);
                    slot.sequence.store(head + 1, std::memory_order_release);
                    size_atomic_.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                // Lost race — reload head and retry
            } else if (diff < 0) {
                return false; // queue full
            } else {
                head = head_.load(std::memory_order_relaxed);
            }
        }
    }

    bool PushBlocking(T&& v) {
        for (;;) {
            if (closed_.load(std::memory_order_relaxed)) return false;
            if (TryPush(std::move(v))) return true;
            std::this_thread::yield();
        }
    }

    // ── Consumer path (single thread) — lock-free ────────────────────────────
    std::optional<T> PopBlocking() {
        std::size_t tail = tail_;
        Slot& slot = buffer_[tail & mask_];
        // Exponential backoff: spin × 64, then yield × unbounded
        unsigned spin = 0;
        for (;;) {
            std::size_t seq = slot.sequence.load(std::memory_order_acquire);
            std::intptr_t diff = static_cast<std::intptr_t>(seq) -
                                 static_cast<std::intptr_t>(tail + 1);
            if (diff == 0) {
                // Slot is ready
                T val = std::move(slot.data);
                tail_++;
                slot.sequence.store(tail + mask_ + 1, std::memory_order_release);
                size_atomic_.fetch_sub(1, std::memory_order_relaxed);
                return val;
            }
            if (closed_.load(std::memory_order_acquire) &&
                size_atomic_.load(std::memory_order_relaxed) == 0)
                return std::nullopt;
            // Backoff
            if (++spin < 64)
                ; // busy-spin: compiler barrier only
            else
                std::this_thread::yield();
        }
    }

    // Timed pop: wait up to `timeout` duration, return nullopt if nothing arrives.
    // Spins with exponential backoff then yields — suitable for a dedicated jthread.
    template <typename Rep, typename Period>
    std::optional<T> PopFor(std::chrono::duration<Rep, Period> timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        unsigned spin = 0;
        for (;;) {
            auto v = TryPop();
            if (v.has_value()) return v;
            if (closed_.load(std::memory_order_relaxed) &&
                size_atomic_.load(std::memory_order_relaxed) == 0)
                return std::nullopt;
            if (std::chrono::steady_clock::now() >= deadline)
                return std::nullopt;
            if (++spin < 64)
                ; // busy-spin
            else
                std::this_thread::yield();
        }
    }

    std::optional<T> TryPop() {
        std::size_t tail = tail_;
        Slot& slot = buffer_[tail & mask_];
        std::size_t seq = slot.sequence.load(std::memory_order_acquire);
        std::intptr_t diff = static_cast<std::intptr_t>(seq) -
                             static_cast<std::intptr_t>(tail + 1);
        if (diff != 0) return std::nullopt;
        T val = std::move(slot.data);
        tail_++;
        slot.sequence.store(tail + mask_ + 1, std::memory_order_release);
        size_atomic_.fetch_sub(1, std::memory_order_relaxed);
        return val;
    }


    void Close() {
        closed_.store(true, std::memory_order_release);
    }

    std::size_t Size() const noexcept {
        return size_atomic_.load(std::memory_order_relaxed);
    }

    bool Closed() const noexcept {
        return closed_.load(std::memory_order_relaxed);
    }

private:
    // ── Round n up to the next power of 2 ────────────────────────────────────
    static std::size_t RoundUpPow2(std::size_t n) noexcept {
        if (n == 0) return 1;
        --n;
        n |= n >> 1; n |= n >> 2; n |= n >> 4;
        n |= n >> 8; n |= n >> 16; n |= n >> 32;
        return ++n;
    }

    // ── Slot — padded to one cache line to prevent false sharing ─────────────
    struct alignas(64) Slot {
        std::atomic<std::size_t> sequence{0};
        T data;
    };

    const std::size_t mask_;
    // Producers update head; align to separate cache line from tail.
    alignas(64) std::atomic<std::size_t> head_;
    // Consumer owns tail exclusively — no atomic needed, but align for separation.
    alignas(64) std::size_t tail_;
    Slot* buffer_;
    std::atomic<std::size_t> size_atomic_{0};
    std::atomic<bool> closed_{false};
};

} // namespace pomai::core
