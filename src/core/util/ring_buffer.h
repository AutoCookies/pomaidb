#pragma once
#include <vector>
#include <array>
#include <optional>
#include <stdexcept>

namespace pomai::core::util {

    /**
     * A simple, fixed-capacity circular buffer (Ring Buffer).
     * Designed for zero-allocation performance in single-threaded environments.
     */
    template <typename T, std::size_t Capacity>
    class StaticRingBuffer {
    public:
        StaticRingBuffer() : head_(0), tail_(0), size_(0) {}

        bool push_back(T&& value) {
            if (size_ == Capacity) return false; // Full
            data_[tail_] = std::move(value);
            tail_ = (tail_ + 1) % Capacity;
            size_++;
            return true;
        }

        std::optional<T> pop_front() {
            if (size_ == 0) return std::nullopt; // Empty
            T value = std::move(data_[head_]);
            head_ = (head_ + 1) % Capacity;
            size_--;
            return value;
        }

        bool empty() const { return size_ == 0; }
        std::size_t size() const { return size_; }
        std::size_t capacity() const { return Capacity; }

        T& front() {
            if (size_ == 0) throw std::runtime_error("Buffer empty");
            return data_[head_];
        }

        void pop_front_no_return() {
            if (size_ > 0) {
                head_ = (head_ + 1) % Capacity;
                size_--;
            }
        }

        void clear() {
            head_ = 0;
            tail_ = 0;
            size_ = 0;
        }

    private:
        std::array<T, Capacity> data_;
        std::size_t head_;
        std::size_t tail_;
        std::size_t size_;
    };

} // namespace pomai::core::util
