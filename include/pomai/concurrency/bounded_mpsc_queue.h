#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>

namespace pomai::concurrency
{
    // Bounded MPSC queue (multi-producer, single-consumer)
    // - Producers: TryPush (non-blocking, returns false if full/closed)
    // - Consumer: Pop (blocking-ish with spin/yield), returns nullopt if closed and drained
    //
    // NOTE:
    // - This implementation uses C++20 atomic wait/notify for efficiency if available.
    // - If your project is not C++20, you MUST either enable C++20 or replace wait/notify with condvar.
    //
    // Capacity is fixed; memory is pre-allocated.
    template <typename T>
    class BoundedMpscQueue final
    {
    public:
        explicit BoundedMpscQueue(std::size_t capacity)
        {
            if (capacity < 1)
                capacity = 1;
            capacity_ = capacity;

            slots_ = std::make_unique<Slot[]>(capacity_);
            for (std::size_t i = 0; i < capacity_; ++i)
            {
                // Turn sequence: even = empty, odd = full.
                slots_[i].turn.store(i * 2, std::memory_order_relaxed);
            }

            head_.store(0, std::memory_order_relaxed);
            tail_.store(0, std::memory_order_relaxed);
            closed_.store(false, std::memory_order_relaxed);
            approx_size_.store(0, std::memory_order_relaxed);
        }

        ~BoundedMpscQueue() { Close(); }

        BoundedMpscQueue(const BoundedMpscQueue &) = delete;
        BoundedMpscQueue &operator=(const BoundedMpscQueue &) = delete;

        void Close()
        {
            const bool was = closed_.exchange(true, std::memory_order_acq_rel);
            if (!was)
            {
                // Wake the consumer if it's waiting.
#if defined(__cpp_lib_atomic_wait) && (__cpp_lib_atomic_wait >= 201907L)
                wake_.fetch_add(1, std::memory_order_release);
                wake_.notify_all();
#endif
            }
        }

        bool IsClosed() const { return closed_.load(std::memory_order_acquire); }

        std::size_t Size() const { return approx_size_.load(std::memory_order_acquire); }

        bool TryPush(T item)
        {
            if (closed_.load(std::memory_order_acquire))
                return false;

            // Fast check (not exact): if tail - head >= capacity => full
            const std::size_t head = head_.load(std::memory_order_acquire);
            std::size_t tail = tail_.load(std::memory_order_relaxed);
            if (tail >= head + capacity_)
                return false;

            while (true)
            {
                if (closed_.load(std::memory_order_acquire))
                    return false;

                const std::size_t idx = tail % capacity_;
                Slot &slot = slots_[idx];

                const std::size_t turn = slot.turn.load(std::memory_order_acquire);
                const std::size_t expect = tail * 2; // empty when == expect
                if (turn == expect)
                {
                    if (tail_.compare_exchange_weak(tail, tail + 1, std::memory_order_acq_rel))
                    {
                        slot.storage = std::move(item);
                        slot.turn.store(expect + 1, std::memory_order_release); // mark full

                        approx_size_.fetch_add(1, std::memory_order_release);

#if defined(__cpp_lib_atomic_wait) && (__cpp_lib_atomic_wait >= 201907L)
                        // Wake consumer
                        wake_.fetch_add(1, std::memory_order_release);
                        wake_.notify_one();
#endif
                        return true;
                    }
                }
                else
                {
                    // Another producer is racing or consumer hasn't advanced; re-check full
                    const std::size_t h2 = head_.load(std::memory_order_acquire);
                    const std::size_t t2 = tail_.load(std::memory_order_relaxed);
                    if (t2 >= h2 + capacity_)
                        return false;

                    // small backoff
                    std::this_thread::yield();
                    tail = tail_.load(std::memory_order_relaxed);
                }
            }
        }

        std::optional<T> Pop()
        {
            while (true)
            {
                std::size_t head = head_.load(std::memory_order_relaxed);
                const std::size_t tail = tail_.load(std::memory_order_acquire);

                if (head == tail)
                {
                    // empty
                    if (closed_.load(std::memory_order_acquire))
                        return std::nullopt;

#if defined(__cpp_lib_atomic_wait) && (__cpp_lib_atomic_wait >= 201907L)
                    // Wait for wake signal
                    std::size_t w = wake_.load(std::memory_order_acquire);
                    wake_.wait(w, std::memory_order_relaxed);
#else
                    // fallback: yield
                    std::this_thread::yield();
#endif
                    continue;
                }

                const std::size_t idx = head % capacity_;
                Slot &slot = slots_[idx];

                const std::size_t expect_full = head * 2 + 1;
                const std::size_t turn = slot.turn.load(std::memory_order_acquire);

                if (turn == expect_full)
                {
                    // Claim head
                    if (head_.compare_exchange_weak(head, head + 1, std::memory_order_acq_rel))
                    {
                        T out = std::move(slot.storage);
                        // mark empty for next cycle: (head+capacity)*2
                        slot.turn.store(head * 2 + 2, std::memory_order_release);

                        approx_size_.fetch_sub(1, std::memory_order_release);
                        return out;
                    }
                }
                else
                {
                    // producer hasn't published fully yet
                    std::this_thread::yield();
                }
            }
        }

    private:
        struct Slot
        {
            std::atomic<std::size_t> turn;
            T storage;
        };

        std::size_t capacity_{0};
        std::unique_ptr<Slot[]> slots_;

        std::atomic<std::size_t> head_{0};
        std::atomic<std::size_t> tail_{0};

        std::atomic<bool> closed_{false};
        std::atomic<std::size_t> approx_size_{0};

#if defined(__cpp_lib_atomic_wait) && (__cpp_lib_atomic_wait >= 201907L)
        std::atomic<std::size_t> wake_{0};
#endif
    };

} // namespace pomai::concurrency