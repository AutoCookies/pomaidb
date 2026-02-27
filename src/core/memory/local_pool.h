// local_pool.h — Thread-local Slab/Arena allocator.
// Inspired by ScyllaDB's log-structured allocator principles.
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include "core/concurrency/concurrency_macros.h"

namespace pomai::core::memory {

/**
 * LocalPool: A specialized, per-shard memory pool.
 * It allocates large "Slabs" and carves them into smaller objects.
 * This eliminates the overhead of the global allocator (even fast ones like mimalloc)
 * during high-frequency vector operations.
 */
class LocalPool {
public:
    static constexpr size_t kSlabSize = 1024 * 1024; // 1MB Slabs

    LocalPool() = default;
    
    // Non-copyable
    LocalPool(const LocalPool&) = delete;
    LocalPool& operator=(const LocalPool&) = delete;

    /**
     * Allocate: Rapidly carve memory from the current slab.
     */
    void* Allocate(size_t size) {
        // Ensure 16-byte alignment for SIMD vector data
        size = (size + 15) & ~15;

        if (current_offset_ + size > kSlabSize) {
            AllocateNewSlab();
        }

        void* ptr = slabs_.back().get() + current_offset_;
        current_offset_ += size;
        return ptr;
    }

    /**
     * Reset: Clear all slabs for reuse. 
     * Extremely fast O(1) in-shard memory reclamation.
     */
    void Reset() {
        current_offset_ = 0;
        // Keep the slabs allocated to avoid oscillation, just reset the pointer
        // If we want to shrink, we would pop_back some entries here.
    }

    /**
     * Clear: Full release of memory.
     */
    void Clear() {
        slabs_.clear();
        current_offset_ = kSlabSize; 
    }

private:
    void AllocateNewSlab() {
        // Use POMAI_CACHE_ALIGNED heap allocation for the slab itself
        auto slab = std::make_unique<uint8_t[]>(kSlabSize);
        slabs_.push_back(std::move(slab));
        current_offset_ = 0;
    }

    std::vector<std::unique_ptr<uint8_t[]>> slabs_;
    size_t current_offset_ = kSlabSize; // Trigger allocation on first call
};

/**
 * ShardMemoryManager: High-level wrapper for shard-local memory resources.
 */
class ShardMemoryManager {
public:
    POMAI_HOT void* AllocTask(size_t size) { return task_pool_.Allocate(size); }
    POMAI_HOT void* AllocVector(size_t size) { return vector_pool_.Allocate(size); }
    
    void ResetHotPools() {
        task_pool_.Reset();
        vector_pool_.Reset();
    }

private:
    LocalPool task_pool_;
    LocalPool vector_pool_;
};

} // namespace pomai::core::memory
