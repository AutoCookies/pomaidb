#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>

#include "pomai/status.h"
#include "pomai/types.h"
#include "util/posix_file.h"

namespace pomai::table
{

    // On-disk format:
    // [Header]
    // [Entry 0: ID (8 bytes) | Vector (dim * 4 bytes)]
    // ...
    // [Entry N-1]
    // [CRC32C (4 bytes) - covers Header + All Entries]

    struct SegmentHeader
    {
        char magic[12]; // "pomai.seg.v1" (padded/null terminated)
        uint32_t version;
        uint32_t count;
        uint32_t dim;
        uint32_t reserved[8];
    };

    class SegmentReader
    {
    public:
        static pomai::Status Open(std::string path, std::unique_ptr<SegmentReader> *out);

        ~SegmentReader();

        // Looks up a vector by ID. Returns NotFound if not present.
        // On success, *out_vec points to internal memory (valid until SegmentReader closed).
        pomai::Status Get(pomai::VectorId id, std::span<const float> *out_vec) const;

        // Iteration
        // Retuns Ok unless callback stops (not implemented yet) or error.
        // Takes a callback: void(pomai::VectorId, std::span<const float>)
        template <typename F>
        void ForEach(F &&func) const
        {
            if (count_ == 0) return;
            const uint8_t* p = base_addr_ + sizeof(SegmentHeader);
            for (uint32_t i = 0; i < count_; ++i) {
                 uint64_t id = *reinterpret_cast<const uint64_t*>(p);
                 const float* vec_ptr = reinterpret_cast<const float*>(p + sizeof(uint64_t));
                 std::span<const float> vec(vec_ptr, dim_);
                 
                 func(static_cast<pomai::VectorId>(id), vec);
                 
                 p += entry_size_;
            }
        }

        uint32_t Count() const { return count_; }
        uint32_t Dim() const { return dim_; }
        std::string Path() const { return path_; }

    private:
        SegmentReader();

        std::string path_;
        pomai::util::PosixFile file_;
        uint32_t count_ = 0;
        uint32_t dim_ = 0;
        std::size_t entry_size_ = 0;
        
        const uint8_t* base_addr_ = nullptr;
        std::size_t file_size_ = 0;
        
        // Mapped memory or buffer could be used.
        // For V1, let's use simple pread or mmap if PosixFile supports it.
        // The existing PosixFile seems basic (PRead/PWrite). 
        // For efficient search we might want mmap later, but for now we might load all or seek.
        // Given typically these are small-ish or we want low latency, let's assume we might need to cache.
        // But for "Get", PRead is fine.
        
        // Cache metadata for binary search? 
        // To do binary search on disk without loading everything:
        // We need to read the ID at distinct offsets.
        
        // Ideally we map the file. 
        // Since PosixFile wrapper doesn't show Map support in previous view, we will implement
        // a simple in-memory load for now if small, or just PRead for Get.
        // Let's stick to PRead for O(logN) lookups to avoid huge memory usage for now,
        // OR just load everything if we want speed for Search.
        // 
        // Optimization: Let's assume we map it or read-only map.
        // The task says "Support loading segments".
        // Let's add mmap support to PosixFile later if needed. 
        // For now, let's implement Get using PRead (Binary Search on disk).
    };

    class SegmentBuilder
    {
    public:
        SegmentBuilder(std::string path, uint32_t dim);
        
        // Add a vector. Must be added in any order, but Finish will sort them?
        // Or we require caller (MemTable flush) to provide them sorted?
        // MemTable is a map, so iteration is not sorted by ID? 
        // Wait, std::unordered_map.
        // So Builder should buffer and sort, or we pass a sorted iterator.
        // Let's support Add() and we buffer/sort inside before writing to ensure on-disk is sorted.
        pomai::Status Add(pomai::VectorId id, std::span<const float> vec);

        pomai::Status Finish();
        
        uint32_t Count() const { return static_cast<uint32_t>(entries_.size()); }

    private:
        struct Entry {
            pomai::VectorId id;
            std::vector<float> vec;
        };
        
        std::string path_;
        uint32_t dim_;
        std::vector<Entry> entries_;
    };

} // namespace pomai::table
