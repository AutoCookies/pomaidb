#include <vector>
#include <cstddef>
#include <cstring>
#include "palloc.h"

namespace pomai::util {

/**
 * AlignedVector: A std::vector-like container that uses palloc for 64-byte alignment.
 * Essential for AVX-512 optimization and preventing cache-line splits.
 */
template <typename T>
class AlignedVector {
public:
    using value_type = T;

    AlignedVector() = default;
    
    ~AlignedVector() {
        if (data_) palloc_free(data_);
    }

    // Move-only for simplicity in hot path
    AlignedVector(const AlignedVector&) = delete;
    AlignedVector& operator=(const AlignedVector&) = delete;

    AlignedVector(AlignedVector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    AlignedVector& operator=(AlignedVector&& other) noexcept {
        if (this != &other) {
            if (data_) palloc_free(data_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void resize(size_t n) {
        if (n <= capacity_) {
            size_ = n;
            return;
        }
        
        size_t new_cap = n * 2;
        if (new_cap < 16) new_cap = 16;
        
        void* new_ptr = palloc_malloc_aligned(new_cap * sizeof(T), 64);
        if (data_) {
            std::memcpy(new_ptr, data_, size_ * sizeof(T));
            palloc_free(data_);
        }
        data_ = static_cast<T*>(new_ptr);
        capacity_ = new_cap;
        size_ = n;
    }

    void reserve(size_t n) {
        if (n <= capacity_) return;
        
        void* new_ptr = palloc_malloc_aligned(n * sizeof(T), 64);
        if (data_) {
            std::memcpy(new_ptr, data_, size_ * sizeof(T));
            palloc_free(data_);
        }
        data_ = static_cast<T*>(new_ptr);
        capacity_ = n;
    }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T* data() { return data_; }
    const T* data() const { return data_; }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

    void push_back(const T& val) {
        if (size_ == capacity_) {
            resize(size_ + 1);
        } else {
            size_++;
        }
        data_[size_ - 1] = val;
    }

private:
    T* data_{nullptr};
    size_t size_{0};
    size_t capacity_{0};
};

} // namespace pomai::util
