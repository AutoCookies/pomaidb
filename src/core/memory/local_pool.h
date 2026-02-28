// local_pool.h — Thread-local Slab/Arena allocator.
// Inspired by ScyllaDB's log-structured allocator principles.
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include "core/concurrency/concurrency_macros.h"
#include "palloc.h"

namespace pomai::core::memory {

/**
 * LocalPool: A specialized, per-shard memory pool.
 * Enhanced to use shard-private palloc heaps for zero-contention allocation.
 */
class LocalPool {
public:
    static constexpr size_t kSlabSize = 1024 * 1024; // 1MB Slabs

    LocalPool() = default;
    
    // Non-copyable
    LocalPool(const LocalPool&) = delete;
    LocalPool& operator=(const LocalPool&) = delete;

    void SetHeap(palloc_heap_t* hp) {
        heap_ = hp;
    }

    /**
     * Allocate: Rapidly carve memory from the current slab.
     * Aligned to 64 bytes for SIMD (AVX-512) optimization.
     */
    void* Allocate(size_t size) {
        // Ensure 64-byte alignment for SIMD vector data
        size = (size + 63) & ~63;

        if (current_offset_ + size > kSlabSize) {
            AllocateNewSlab();
        }

        void* ptr = slabs_.back() + current_offset_;
        current_offset_ += size;
        return ptr;
    }

    /**
     * Reset: Clear all slabs for reuse. 
     */
    void Reset() {
        current_offset_ = 0;
    }

    /**
     * Clear: Full release of memory.
     */
    void Clear() {
        for (auto slab : slabs_) {
            palloc_free(slab);
        }
        slabs_.clear();
        current_offset_ = kSlabSize; 
    }

    ~LocalPool() {
        Clear();
    }

private:
    void AllocateNewSlab() {
        // Allocate slab from the shard's private heap
        void* slab = nullptr;
        if (heap_) {
            slab = palloc_heap_malloc_aligned(heap_, kSlabSize, 64);
        } else {
            slab = palloc_malloc_aligned(kSlabSize, 64);
        }
        slabs_.push_back(static_cast<uint8_t*>(slab));
        current_offset_ = 0;
    }

    palloc_heap_t* heap_{nullptr};
    std::vector<uint8_t*> slabs_;
    size_t current_offset_ = kSlabSize; 
};

/**
 * ShardMemoryManager: High-level wrapper for shard-local memory resources.
 */
class ShardMemoryManager {
public:
    void Initialize(palloc_heap_t* heap) {
        task_pool_.SetHeap(heap);
        vector_pool_.SetHeap(heap);
    }

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
