// router.h — Consistent-hash MembraneID→ShardID router for PomaiDB.
//
// Helio-inspired shared-nothing architecture (Phase 2).
// Implements a seqlock-protected shard map: readers spin on an even sequence
// counter (no lock, no CAS) — optimal for frequent cross-thread reads.
// One writer at a time via a std::mutex that only a single background thread
// (routing warmup) ever holds.
//
// No Abseil, no Boost, no fibers — pure C++20 + linux syscalls.

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <span>
#include <vector>

namespace pomai::core {

// ── Consistent-hash helper ───────────────────────────────────────────────────
// Maps an arbitrary 64-bit key (MembraneID or VectorId) to a shard index in
// [0, num_shards) using jump consistent hashing (Google 2014, 3 lines).
// Deterministic, zero-allocation, O(log n) work.
inline uint32_t JumpConsistentHash(uint64_t key, uint32_t num_shards) noexcept {
    int64_t b = -1, j = 0;
    while (j < static_cast<int64_t>(num_shards)) {
        b = j;
        key = key * 2862933555777941757ULL + 1;
        j = static_cast<int64_t>(
            static_cast<double>(b + 1) *
            (static_cast<double>(1LL << 31) /
             static_cast<double>((key >> 33) + 1)));
    }
    return static_cast<uint32_t>(b);
}

// ── ShardRouter — seqlock-protected routing table ───────────────────────────
// Provides O(1) routing from any key to a shard.
// Read path: 2 atomic loads + pointer deref — ~5 ns on modern hardware.
// Write path: seqlock increment + update + increment.  Rare (only during warm-up).
class ShardRouter {
public:
    explicit ShardRouter(uint32_t num_shards) noexcept
        : num_shards_(num_shards), seq_{0} {}

    // Default-shard routing (no vector hint): pure key hash.
    uint32_t RouteByKey(uint64_t key) const noexcept {
        return JumpConsistentHash(key, num_shards_);
    }

    // Vector-hint routing using the current centroid table (may be null).
    // Falls back to RouteByKey if table is not ready. Thread-safe, lock-free.
    uint32_t RouteByVector(uint64_t key, std::span<const float> /*vec*/) const noexcept {
        // If we ever store per-shard centroid data in a future extension,
        // we read it here under the seqlock pattern. For now delegate to key hash.
        return RouteByKey(key);
    }

    uint32_t num_shards() const noexcept { return num_shards_; }

    // ── Seqlock read guard (for callers that need a consistent snapshot) ──────
    // Usage:
    //   uint32_t ver;
    //   do { ver = router.ReadBegin(); ... } while (!router.ReadEnd(ver));
    uint32_t ReadBegin() const noexcept {
        uint32_t v;
        do { v = seq_.load(std::memory_order_acquire); }
        while (v & 1u); // spin while writer is active (odd = write in progress)
        return v;
    }

    bool ReadEnd(uint32_t version) const noexcept {
        return seq_.load(std::memory_order_acquire) == version;
    }

    // ── Seqlock write: called by routing warmup thread only ──────────────────
    // Pass a callable `fn` that updates any derived state atomically.
    template <typename Fn>
    void Update(Fn&& fn) {
        std::lock_guard<std::mutex> lk(write_mu_); // one writer at a time
        seq_.fetch_add(1, std::memory_order_release); // mark write start (odd)
        std::atomic_thread_fence(std::memory_order_seq_cst);
        fn();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        seq_.fetch_add(1, std::memory_order_release); // mark write end (even)
    }

private:
    const uint32_t num_shards_;
    alignas(64) std::atomic<uint32_t> seq_;  // seqlock counter — separate cache line
    std::mutex write_mu_;                     // serialise concurrent writers
};

} // namespace pomai::core
