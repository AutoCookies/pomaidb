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

    // On-disk format V2:
    // [Header]
    // [Entry 0: ID (8 bytes) | Flags (1 byte) | Vector (dim * 4 bytes)] (Packed/Unaligned potentially, but we'll manually serialize)
    // ...
    // [Entry N-1]
    // [CRC32C (4 bytes)]

    struct SegmentHeader
    {
        char magic[12]; // "pomai.seg.v1" (v1 reader compatibility check? No we bump version)
                        // Let's keep magic same but check version.
        uint32_t version; // 2
        uint32_t count;
        uint32_t dim;
        uint32_t reserved[8];
    };

    // Flags
    constexpr uint8_t kFlagNone = 0;
    constexpr uint8_t kFlagTombstone = 1 << 0;

    class SegmentReader
    {
    public:
        static pomai::Status Open(std::string path, std::unique_ptr<SegmentReader> *out);

        ~SegmentReader();

        // Looks up a vector by ID. 
        // If ID is found but is a Tombstone, returns NotFound("tombstone").
        // (Or should we return specific status? For Get, it's effectively NotFound).
        pomai::Status Get(pomai::VectorId id, std::span<const float> *out_vec) const;

        // Check if ID exists (handling tombstones).
        // Returns Ok + true if present and alive.
        // Returns Ok + false if present but tombstone (explicit delete).
        // Returns Ok + false if not present at all?
        // Wait, Exists needs to distinguish "Known Deleted" vs "Unknown".
        // Upper layers (ShardRuntime) iterate segments Newest -> Oldest.
        // If Newest has Tombstone -> STOP, return Deleted.
        // If Newest has Alive -> STOP, return Exists.
        // If Not Found -> Continue.
        // So we need a way to return "FoundTombstone".
        enum class FindResult {
            kFound,
            kFoundTombstone,
            kNotFound
        };
        FindResult Find(pomai::VectorId id, std::span<const float> *out_vec) const;
        
        // Read entry at index [0, Count()-1]
        // Returns ID, Vector (if not deleted), and Deleted Status.
        pomai::Status ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted) const;

        // Iteration
        // Callback: void(VectorId, span<float>, bool is_deleted)
        template <typename F>
        void ForEach(F &&func) const
        {
            if (count_ == 0) return;
            const uint8_t* p = base_addr_ + sizeof(SegmentHeader);
            for (uint32_t i = 0; i < count_; ++i) {
                 uint64_t id = *reinterpret_cast<const uint64_t*>(p);
                 uint8_t flags = *(p + 8);
                 const float* vec_ptr = reinterpret_cast<const float*>(p + 12);
                 
                 std::span<const float> vec(vec_ptr, dim_);
                 bool is_deleted = (flags & kFlagTombstone);
                 
                 func(static_cast<pomai::VectorId>(id), vec, is_deleted);
                 
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
    };

    class SegmentBuilder
    {
    public:
        SegmentBuilder(std::string path, uint32_t dim);
        
        // Add a vector. 
        // If is_deleted is true, vec content is ignored (will be zeroed on disk).
        // vec.size() must match dim unless is_deleted is true (then it can be empty).
        pomai::Status Add(pomai::VectorId id, std::span<const float> vec, bool is_deleted);

        pomai::Status Finish();
        
        uint32_t Count() const { return static_cast<uint32_t>(entries_.size()); }

    private:
        struct Entry {
            pomai::VectorId id;
            std::vector<float> vec; // Empty if deleted
            bool is_deleted;
        };
        
        std::string path_;
        uint32_t dim_;
        std::vector<Entry> entries_;
    };

} // namespace pomai::table
