#pragma once
// src/util/spsc_ring.h
//
// Very small bounded SPSC ring buffer for pointers (single-producer single-consumer).
// - Capacity must be a power of two.
// - push/pop are wait-free and return bool indicating success.
// - Designed for pointer types (T*). Lightweight, no dynamic allocation.
//
// Usage:
//   SpscRing<Task*> ring(capacity_power_of_two);
//   bool ok = ring.push(ptr);   // producer
//   Task* item = nullptr; bool got = ring.pop(item); // consumer
//
// Memory ordering:
//  - push uses acquire/release to publish pointer to consumer.
//  - pop uses acquire/release to read published data.

#include <atomic>
#include <cstddef>
#include <vector>
#include <cassert>

template <typename T>
class SpscRing {
public:
    explicit SpscRing(size_t capacity_power_of_two) {
        assert((capacity_power_of_two & (capacity_power_of_two - 1)) == 0 && "capacity must be power of two");
        mask_ = capacity_power_of_two - 1;
        buf_.resize(capacity_power_of_two);
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    // non-copyable
    SpscRing(const SpscRing &) = delete;
    SpscRing &operator=(const SpscRing &) = delete;

    // try to push; returns false if full
    bool push(T item) noexcept {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t next = tail + 1;
        if (next - head_.load(std::memory_order_acquire) > buf_.size()) {
            return false; // full
        }
        buf_[tail & mask_] = item;
        tail_.store(next, std::memory_order_release);
        return true;
    }

    // try to pop; returns false if empty
    bool pop(T &out) noexcept {
        size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_.load(std::memory_order_acquire)) {
            return false; // empty
        }
        out = buf_[head & mask_];
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
    }

    // capacity (power of two)
    size_t capacity() const noexcept { return buf_.size(); }

private:
    std::vector<T> buf_;
    size_t mask_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};