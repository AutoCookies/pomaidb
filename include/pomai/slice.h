#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <span>

namespace pomai {

/**
 * Cleanable: A simple resource tracker that runs cleanup functions on destruction.
 * Simplified version of RocksDB's Cleanable for edge use-cases.
 */
class Cleanable {
 public:
  using CleanupFunction = void (*)(void* arg1, void* arg2);

  Cleanable() : cleanup_{nullptr, nullptr, nullptr, nullptr} {}
  virtual ~Cleanable() { DoCleanup(); }

  Cleanable(const Cleanable&) = delete;
  Cleanable& operator=(const Cleanable&) = delete;

  Cleanable(Cleanable&& other) noexcept : cleanup_(other.cleanup_) {
    other.cleanup_.function = nullptr;
  }

  Cleanable& operator=(Cleanable&& other) noexcept {
    if (this != &other) {
      DoCleanup();
      cleanup_ = other.cleanup_;
      other.cleanup_.function = nullptr;
    }
    return *this;
  }

  void RegisterCleanup(CleanupFunction function, void* arg1, void* arg2) {
    assert(function != nullptr);
    if (cleanup_.function == nullptr) {
      cleanup_.function = function;
      cleanup_.arg1 = arg1;
      cleanup_.arg2 = arg2;
    } else {
      auto* c = new Cleanup{function, arg1, arg2, cleanup_.next};
      cleanup_.next = c;
    }
  }

  void DelegateCleanupsTo(Cleanable* other) {
    assert(other != nullptr);
    if (cleanup_.function == nullptr) return;
    
    other->RegisterCleanup(cleanup_.function, cleanup_.arg1, cleanup_.arg2);
    
    if (cleanup_.next != nullptr) {
       Cleanup* tail = cleanup_.next;
       while (tail->next != nullptr) tail = tail->next;
       tail->next = other->cleanup_.next;
       other->cleanup_.next = cleanup_.next;
    }
    
    cleanup_.function = nullptr;
    cleanup_.next = nullptr;
  }

  void Reset() {
    DoCleanup();
    cleanup_ = {nullptr, nullptr, nullptr, nullptr};
  }

 protected:
  struct Cleanup {
    CleanupFunction function;
    void* arg1;
    void* arg2;
    Cleanup* next;
  };
  Cleanup cleanup_;

 private:
  void DoCleanup() {
    if (cleanup_.function) {
      cleanup_.function(cleanup_.arg1, cleanup_.arg2);
      for (Cleanup* c = cleanup_.next; c != nullptr; ) {
        c->function(c->arg1, c->arg2);
        Cleanup* next = c->next;
        delete c;
        c = next;
      }
    }
  }
};

/**
 * Slice: A wrapper around a pointer and a size.
 * Uses std::byte for generic data to avoid signedness issues.
 */
struct Slice {
  const std::byte* data_ = nullptr;
  size_t size_ = 0;

  constexpr Slice() = default;
  constexpr Slice(const void* d, size_t n) 
      : data_(static_cast<const std::byte*>(d)), size_(n) {}
  
  /* implicit */ Slice(const std::string& s) 
      : data_(reinterpret_cast<const std::byte*>(s.data())), size_(s.size()) {}
  
  /* implicit */ Slice(std::string_view sv) 
      : data_(reinterpret_cast<const std::byte*>(sv.data())), size_(sv.size()) {}
  
  /* implicit */ Slice(const char* s) 
      : data_(reinterpret_cast<const std::byte*>(s)), size_(s ? strlen(s) : 0) {}

  template<typename T, size_t Extent>
  /* implicit */ Slice(std::span<T, Extent> s)
      : data_(reinterpret_cast<const std::byte*>(s.data())), size_(s.size_bytes()) {}

  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] constexpr const std::byte* data() const noexcept { return data_; }
  [[nodiscard]] constexpr size_t size() const noexcept { return size_; }

  void remove_prefix(size_t n) {
    assert(n <= size_);
    data_ += n;
    size_ -= n;
  }

  std::string_view ToStringView() const noexcept {
    return {reinterpret_cast<const char*>(data_), size_};
  }
};

inline bool operator==(const Slice& x, const Slice& y) {
  return (x.size() == y.size()) && (memcmp(x.data(), y.data(), x.size()) == 0);
}

/**
 * PinnableSlice: A Slice that can pin resources (like mmap regions).
 * Zero-copy oriented.
 */
class PinnableSlice : public Slice, public Cleanable {
 public:
  PinnableSlice() = default;
  
  PinnableSlice(PinnableSlice&& other) noexcept 
      : Slice(other), Cleanable(std::move(other)) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  PinnableSlice& operator=(PinnableSlice&& other) noexcept {
    if (this != &other) {
      Slice::operator=(other);
      Cleanable::operator=(std::move(other));
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // Pin external data with a cleanup function
  void PinSlice(Slice s, CleanupFunction f, void* arg1, void* arg2) {
    Reset();
    this->data_ = s.data();
    this->size_ = s.size();
    RegisterCleanup(f, arg1, arg2);
    pinned_ = true;
  }

  // Pin data and delegate cleanup from another Cleanable
  void PinSlice(Slice s, Cleanable* cleanable) {
    Reset();
    this->data_ = s.data();
    this->size_ = s.size();
    if (cleanable) {
      cleanable->DelegateCleanupsTo(this);
    }
    pinned_ = true;
  }

  // Internal buffer for data that must be copied
  void PinSelf(Slice s) {
    Reset();
    buf_.assign(reinterpret_cast<const char*>(s.data()), s.size());
    this->data_ = reinterpret_cast<const std::byte*>(buf_.data());
    this->size_ = buf_.size();
    pinned_ = false;
  }

  void Reset() {
    Cleanable::Reset();
    this->data_ = nullptr;
    this->size_ = 0;
    pinned_ = false;
  }

  [[nodiscard]] bool IsPinned() const { return pinned_; }

 private:
  std::string buf_; // For non-pinned data (fallback)
  bool pinned_ = false;
};

} // namespace pomai
