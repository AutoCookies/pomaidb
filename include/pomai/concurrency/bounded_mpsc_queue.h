#pragma once
#include <atomic>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <type_traits>

namespace pomai::concurrency
{

    // Định nghĩa cứng cache line size để tránh warning của GCC 12+ về ABI interference
    constexpr std::size_t kCacheLineSize = 64;

    template <typename T>
    class BoundedMpscQueue final
    {
    public:
        explicit BoundedMpscQueue(std::size_t capacity)
        {
            if (capacity < 1)
                capacity = 1;
            capacity_ = capacity;

            // FIX: Dùng unique_ptr quản lý mảng thay vì vector để tránh lỗi copy/move atomic
            slots_ = std::make_unique<Slot[]>(capacity_);

            for (std::size_t i = 0; i < capacity_; ++i)
            {
                // Init turn: 0, 2, 4...
                slots_[i].turn.store(i * 2, std::memory_order_relaxed);
            }

            head_.store(0, std::memory_order_relaxed);
            tail_.store(0, std::memory_order_relaxed);
        }

        ~BoundedMpscQueue()
        {
            Close();
        }

        BoundedMpscQueue(const BoundedMpscQueue &) = delete;
        BoundedMpscQueue &operator=(const BoundedMpscQueue &) = delete;

        // Non-blocking Push
        bool TryPush(T item)
        {
            const std::size_t head = head_.load(std::memory_order_acquire);
            std::size_t tail = tail_.load(std::memory_order_relaxed);

            if (tail >= head + capacity_)
                return false;

            while (true)
            {
                if (closed_.load(std::memory_order_acquire))
                    return false;

                const std::size_t idx = tail % capacity_;
                auto &slot = slots_[idx];
                const std::size_t turn = slot.turn.load(std::memory_order_acquire);
                const std::size_t diff = turn - (tail * 2);

                if (diff == 0)
                {
                    if (tail_.compare_exchange_weak(tail, tail + 1, std::memory_order_relaxed))
                    {
                        slot.storage = std::move(item);
                        slot.turn.store(tail * 2 + 1, std::memory_order_release);
                        slot.turn.notify_all();
                        return true;
                    }
                }
                else if (static_cast<std::make_signed_t<std::size_t>>(diff) < 0)
                {
                    return false;
                }
                else
                {
                    const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
                    if (current_tail == tail)
                    {
                        std::this_thread::yield();
                    }
                    tail = current_tail;
                }
            }
        }

        // Blocking Push
        bool Push(T item)
        {
            std::size_t tail = tail_.load(std::memory_order_relaxed);
            while (true)
            {
                if (closed_.load(std::memory_order_acquire))
                    return false;

                const std::size_t idx = tail % capacity_;
                auto &slot = slots_[idx];
                const std::size_t turn = slot.turn.load(std::memory_order_acquire);
                const std::size_t diff = turn - (tail * 2);

                if (diff == 0)
                {
                    if (tail_.compare_exchange_weak(tail, tail + 1, std::memory_order_relaxed))
                    {
                        slot.storage = std::move(item);
                        slot.turn.store(tail * 2 + 1, std::memory_order_release);
                        slot.turn.notify_all();
                        return true;
                    }
                }
                else if (static_cast<std::make_signed_t<std::size_t>>(diff) < 0)
                {
                    // Wait on address using C++20 atomic wait
                    slot.turn.wait(turn, std::memory_order_relaxed);
                    tail = tail_.load(std::memory_order_relaxed);
                }
                else
                {
                    const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
                    if (current_tail == tail)
                        std::this_thread::yield();
                    tail = current_tail;
                }
            }
        }

        // Blocking Pop
        std::optional<T> Pop()
        {
            const std::size_t head = head_.load(std::memory_order_relaxed);
            const std::size_t idx = head % capacity_;
            auto &slot = slots_[idx];

            while (true)
            {
                const std::size_t turn = slot.turn.load(std::memory_order_acquire);
                const std::size_t target = head * 2 + 1;

                if (turn == target)
                {
                    T item = std::move(slot.storage);
                    slot.turn.store((head + capacity_) * 2, std::memory_order_release);
                    head_.store(head + 1, std::memory_order_relaxed);
                    slot.turn.notify_all();
                    return item;
                }

                if (closed_.load(std::memory_order_acquire))
                {
                    return std::nullopt;
                }

                slot.turn.wait(turn, std::memory_order_relaxed);
            }
        }

        void Close()
        {
            bool expected = false;
            if (closed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
            {
                // Wake up all waiters
                for (std::size_t i = 0; i < capacity_; ++i)
                {
                    slots_[i].turn.notify_all();
                }
            }
        }

        std::size_t Size() const
        {
            const std::size_t head = head_.load(std::memory_order_relaxed);
            const std::size_t tail = tail_.load(std::memory_order_relaxed);
            if (tail >= head)
                return tail - head;
            return 0;
        }

    private:
        struct Slot
        {
            std::atomic<std::size_t> turn;
            T storage;
        };

        alignas(kCacheLineSize) std::atomic<std::size_t> head_;
        alignas(kCacheLineSize) std::atomic<std::size_t> tail_;
        alignas(kCacheLineSize) std::atomic<bool> closed_{false};

        std::size_t capacity_;

        // FIX: unique_ptr array thay vì vector
        std::unique_ptr<Slot[]> slots_;
    };

} // namespace pomai::concurrency