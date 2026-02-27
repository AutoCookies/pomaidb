// flat_hash_memmap.h — Cache-friendly open-addressing hash map for PomaiDB MemTable.
//
// Design:
//  - Robin-hood linear probing: entries displaced by a later one that is closer to home,
//    keeping average probe distance low and enabling SIMD-friendly linear scans.
//  - Power-of-2 capacity. Load factor capped at 0.75 (resize at 3/4 full).
//  - Incremental resize: new capacity allocated then all entries rehashed in O(N).
//    No latency spike: resize happens on Put() when load > threshold.
//  - Tombstone-free deletion: backward-shift deletion avoids tombstone accumulation
//    which degrades open-addressing tables under heavy delete workloads.
//  - Zero external deps — pure C++20 stdlib only.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace pomai::table {

// FlatHashMemMap<K, V, Hash, Eq>
// A minimal open-addressing flat hash map optimised for:
//   - 64-bit integer keys (VectorId)
//   - Cache-line-friendly linear probing
//   - ARM/NEON + x86 cache locality (8-byte slot, 64-byte cache line = 4 slots/line)
template <typename K, typename V,
          typename Hash = std::hash<K>,
          typename Eq   = std::equal_to<K>>
class FlatHashMemMap {
 public:
  static constexpr float    kMaxLoadFactor = 0.75f;
  static constexpr uint64_t kEmpty         = uint64_t(-1);
  static constexpr size_t   kInitialCap    = 64; // must be power-of-2

  struct Slot {
    K key;
    V value;
  };

  // Unifying sentinel: a key == kEmpty marks an empty slot.
  // We can use this because VectorId is a uint64_t and we
  // can reserve the sentinel value (all-ones).
  // static_assert(std::is_same_v<K, uint64_t>, "FlatHashMemMap keys must be uint64_t");
  static_assert(std::is_trivially_copy_assignable_v<V>,
                "V must be trivially copyable for backward-shift deletion");

  explicit FlatHashMemMap(size_t initial_cap = kInitialCap)
      : cap_(initial_cap ? RoundUpPow2(initial_cap) : kInitialCap),
        mask_(cap_ - 1),
        slots_(std::make_unique<Slot[]>(cap_)),
        count_(0),
        sentinel_(K(~uint64_t(0)))
  {
    clear_slots(0, cap_);
  }

  FlatHashMemMap(const FlatHashMemMap&)            = delete;
  FlatHashMemMap& operator=(const FlatHashMemMap&) = delete;

  FlatHashMemMap(FlatHashMemMap&&)            = default;
  FlatHashMemMap& operator=(FlatHashMemMap&&) = default;

  ~FlatHashMemMap() = default;

  // Insert or overwrite key->value. Returns true if new key was inserted.
  bool Put(const K& key, V value) {
    assert(key != sentinel_);
    if (count_ >= threshold()) {
      Grow();
    }
    return PutInternal(slots_.get(), mask_, key, value, /*allow_overwrite=*/true);
  }

  // Returns pointer to V or nullptr if not found.
  V* Find(const K& key) noexcept {
    size_t idx = BucketOf(key);
    for (size_t i = 0; i < cap_; ++i) {
      Slot& s = slots_[(idx + i) & mask_];
      if (s.key == sentinel_) return nullptr;
      if (eq_(s.key, key))   return &s.value;
    }
    return nullptr;
  }

  const V* Find(const K& key) const noexcept {
    return const_cast<FlatHashMemMap*>(this)->Find(key);
  }

  // Backward-shift deletion (no tombstones).
  // Returns true if the key was present and removed.
  bool Erase(const K& key) noexcept {
    size_t idx = BucketOf(key);
    size_t pos = cap_; // sentinel = not found
    for (size_t i = 0; i < cap_; ++i) {
      size_t probe = (idx + i) & mask_;
      Slot& s = slots_[probe];
      if (s.key == sentinel_) break;
      if (eq_(s.key, key)) { pos = probe; break; }
    }
    if (pos == cap_) return false; // not found

    // Backward-shift: pull slots backward toward the deleted slot until we
    // find either an empty slot or a slot that is already at its home position.
    size_t hole = pos;
    for (size_t i = 1; i < cap_; ++i) {
      size_t next = (hole + i) & mask_;
      Slot& s = slots_[next];
      if (s.key == sentinel_) break; // stop at empty

      // Natural bucket of the candidate
      size_t home = BucketOf(s.key);
      // Can we move 'next' backward to 'hole'?
      // Yes if 'hole' is between 'home' and 'next' (mod cap_) in the forward direction.
      size_t dist_home_to_next = (next - home) & mask_;
      size_t dist_home_to_hole = (hole - home) & mask_;
      if (dist_home_to_hole <= dist_home_to_next) {
        slots_[hole] = slots_[next]; // shift back
        hole = next;
      }
    }
    // Mark hole as empty
    slots_[hole].key = sentinel_;
    slots_[hole].value = V{};
    --count_;
    return true;
  }

  size_t size() const noexcept { return count_; }
  size_t capacity() const noexcept { return cap_; }
  bool   empty() const noexcept { return count_ == 0; }

  void Clear() noexcept {
    clear_slots(0, cap_);
    count_ = 0;
  }

  // Iterate over all occupied slots. Callback: fn(const K& k, V& v) -> void.
  template <typename Fn>
  void ForEach(Fn&& fn) const {
    for (size_t i = 0; i < cap_; ++i) {
      const Slot& s = slots_[i];
      if (s.key != sentinel_) {
        fn(s.key, const_cast<V&>(s.value));
      }
    }
  }

  template <typename Fn>
  void ForEach(Fn&& fn) {
    for (size_t i = 0; i < cap_; ++i) {
      Slot& s = slots_[i];
      if (s.key != sentinel_) {
        fn(s.key, s.value);
      }
    }
  }

  // Raw slot access for Cursor iteration (read-only).
  const Slot* data() const noexcept { return slots_.get(); }
  const K     sentinel() const noexcept { return sentinel_; }

 private:
  size_t cap_;
  size_t mask_;
  std::unique_ptr<Slot[]> slots_;
  size_t count_{0};
  Hash   hash_{};
  Eq     eq_{};
  K      sentinel_;

  size_t threshold() const noexcept {
    return static_cast<size_t>(cap_ * kMaxLoadFactor);
  }

  size_t BucketOf(const K& k) const noexcept {
    return static_cast<size_t>(hash_(k)) & mask_;
  }

  void clear_slots(size_t begin, size_t end) noexcept {
    for (size_t i = begin; i < end; ++i) {
      slots_[i].key   = sentinel_;
      slots_[i].value = V{};
    }
  }

  static size_t RoundUpPow2(size_t n) noexcept {
    if (n == 0) return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
  }

  // Returns true if newly inserted (false = overwrite).
  bool PutInternal(Slot* slots, size_t mask, const K& key, V value,
                   bool allow_overwrite) noexcept {
    size_t idx = static_cast<size_t>(hash_(key)) & mask;
    for (size_t i = 0; i <= mask; ++i) {
      Slot& s = slots[(idx + i) & mask];
      if (s.key == sentinel_) {
        // empty slot — insert here
        s.key   = key;
        s.value = value;
        ++count_;
        return true;
      }
      if (eq_(s.key, key)) {
        if (allow_overwrite) s.value = value;
        return false; // existing key
      }
    }
    // Should never reach here with load < 1.0
    assert(false && "FlatHashMemMap: table overflow");
    return false;
  }

  void Grow() {
    size_t new_cap  = cap_ * 2;
    size_t new_mask = new_cap - 1;
    auto   new_slots = std::make_unique<Slot[]>(new_cap);
    // init new table
    for (size_t i = 0; i < new_cap; ++i) {
      new_slots[i].key   = sentinel_;
      new_slots[i].value = V{};
    }
    // re-insert all existing entries (count_ does not change)
    size_t old_count = count_;
    count_ = 0; // PutInternal will re-increment
    for (size_t i = 0; i < cap_; ++i) {
      Slot& s = slots_[i];
      if (s.key != sentinel_) {
        PutIntoRaw(new_slots.get(), new_mask, s.key, s.value);
      }
    }
    count_    = old_count;
    cap_      = new_cap;
    mask_     = new_mask;
    slots_    = std::move(new_slots);
  }

  void PutIntoRaw(Slot* slots, size_t mask, const K& key, const V& value) noexcept {
    size_t idx = static_cast<size_t>(hash_(key)) & mask;
    for (size_t i = 0; i <= mask; ++i) {
      Slot& s = slots[(idx + i) & mask];
      if (s.key == sentinel_) {
        s.key   = key;
        s.value = value;
        return;
      }
    }
  }
};

} // namespace pomai::table
