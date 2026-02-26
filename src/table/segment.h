#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <cstring>

#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/metadata.h"
#include "pomai/quantization/scalar_quantizer.h"
#include "util/posix_file.h"

// Forward declare in correct namespace
namespace pomai::index { class IvfFlatIndex; }

namespace pomai::table
{

    // On-disk format V4 (Metadata & Quantization support):
    // [Header]
    // [Entry 0: ID (8 bytes) | Flags (1 byte) | Vector (dim * 4 bytes if V3, dim * 1 bytes if V4 quantized)]
    // ...
    // [Entry N-1]
    // [Metadata Block (Optional, V3+)]
    //    [Offsets: (count+1) * 8 bytes] (Relative to start of Blob)
    //    [Blob: variable bytes]
    // [CRC32C (4 bytes)]

    struct SegmentHeader
    {
        char magic[12]; // "pomai.seg.v1"
        uint32_t version; // 4
        uint32_t count;
        uint32_t dim;
        uint32_t metadata_offset; // V3+: Offset to metadata block (0 if none)
        uint8_t  is_quantized;    // V4+: 1 if vectors are SQ8 (uint8_t)
        uint8_t  reserved1[3];
        float    quant_min;       // SQ8 minimum bound
        float    quant_inv_scale; // SQ8 global inverse scale
        uint32_t entries_start_offset; // V5+: Start of entries block (must be 4096-aligned)
        uint32_t entry_size;           // V5+: Padded entry size
        uint32_t reserved2[2];
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
        pomai::Status Get(pomai::VectorId id, std::span<const float> *out_vec, pomai::Metadata* out_meta) const;
        pomai::Status Get(pomai::VectorId id, std::span<const float> *out_vec) const;
        
        // V4: Quantized raw lookup
        bool IsQuantized() const { return is_quantized_; }
        const core::ScalarQuantizer8Bit* GetQuantizer() const { return quantizer_.get(); }
        pomai::Status GetQuantized(pomai::VectorId id, std::span<const uint8_t>* out_codes, pomai::Metadata* out_meta) const;

        enum class FindResult {
            kFound,
            kFoundTombstone,
            kNotFound
        };
        FindResult Find(pomai::VectorId id, std::span<const float> *out_vec, pomai::Metadata* out_meta) const;
        FindResult Find(pomai::VectorId id, std::span<const float> *out_vec) const;
        
        // Internal raw search helper
        FindResult FindRaw(pomai::VectorId id, const uint8_t** raw_payload, pomai::Metadata* out_meta) const;

        // Decodes a quantized vector on-the-fly, or returns mapped float vec
        FindResult FindAndDecode(pomai::VectorId id, std::span<const float>* out_vec_mapped, std::vector<float>* out_vec_decoded, pomai::Metadata* out_meta) const;
        
        // Approximate Search via IVF Index.
        pomai::Status Search(std::span<const float> query, uint32_t nprobe, 
                             std::vector<uint32_t>* out_candidates) const;

        bool HasIndex() const { return index_ != nullptr; }

        
        // Read entry at index [0, Count()-1]
        pomai::Status ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted, pomai::Metadata* out_meta) const;
        pomai::Status ReadAtCodes(uint32_t index, pomai::VectorId* out_id, std::span<const uint8_t>* out_codes, bool* out_deleted, pomai::Metadata* out_meta) const;
        pomai::Status ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted) const;


        // Iteration
        // Callback: void(VectorId, span<float>, bool is_deleted, const Metadata* meta)
        template <typename F>
        void ForEach(F &&func) const
        {
            if (count_ == 0) return;
            const uint8_t* p = base_addr_ + entries_start_offset_;
            const uint8_t* meta_offsets_base = nullptr;
            const char* meta_blob = nullptr;
            
            if (metadata_offset_ > 0) {
                meta_offsets_base = base_addr_ + metadata_offset_;
                 // Blob starts after offsets array
                 // count_ entries -> (count_ + 1) offsets
                 meta_blob = reinterpret_cast<const char*>(meta_offsets_base + (count_ + 1) * sizeof(uint64_t));
            }
            
            for (uint32_t i = 0; i < count_; ++i) {
                 uint64_t id = 0;
                 std::memcpy(&id, p, sizeof(id));
                 uint8_t flags = *(p + 8);
                 const void* vec_ptr = reinterpret_cast<const void*>(p + 12);
                 
                 bool is_deleted = (flags & kFlagTombstone);
                 
                 pomai::Metadata meta_obj;
                 const pomai::Metadata* meta_ptr = nullptr;
                 
                 if (meta_offsets_base && meta_blob) {
                     uint64_t start = 0;
                     uint64_t end = 0;
                     std::memcpy(&start, meta_offsets_base + i * sizeof(uint64_t), sizeof(start));
                     std::memcpy(&end, meta_offsets_base + (i + 1) * sizeof(uint64_t), sizeof(end));
                     if (end > start) {
                         meta_obj.tenant = std::string(meta_blob + start, end - start);
                         meta_ptr = &meta_obj;
                     }
                 }
                 
                 std::vector<float> decoded;
                 std::span<const float> vec_span;
                 
                 if (!is_deleted) {
                     if (is_quantized_) {
                         std::span<const uint8_t> codes(static_cast<const uint8_t*>(vec_ptr), dim_);
                         decoded = quantizer_->Decode(codes);
                         vec_span = decoded;
                     } else {
                         vec_span = std::span<const float>(static_cast<const float*>(vec_ptr), dim_);
                     }
                 }
                 
                 func(static_cast<pomai::VectorId>(id), vec_span, is_deleted, meta_ptr);
                 
                 p += entry_size_;
            }
        }

        uint32_t Count() const { return count_; }
        uint32_t Dim() const { return dim_; }
        std::string Path() const { return path_; }

        const uint8_t* GetBaseAddr() const { return base_addr_; }
        uint32_t GetEntriesStartOffset() const { return entries_start_offset_; }
        std::size_t GetEntrySize() const { return entry_size_; }

    private:
        SegmentReader();

        std::string path_;
        pomai::util::PosixFile file_;
        uint32_t count_ = 0;
        uint32_t dim_ = 0;
        std::size_t entry_size_ = 0;
        uint32_t entries_start_offset_ = 0;
        uint32_t metadata_offset_ = 0;
        
        // V4: Quantization properties
        bool is_quantized_{false};
        std::unique_ptr<core::ScalarQuantizer8Bit> quantizer_;
        
        const uint8_t* base_addr_ = nullptr;
        std::size_t file_size_ = 0;
        
        std::unique_ptr<pomai::index::IvfFlatIndex> index_;
        
        // Internal helper
        void GetMetadata(uint32_t index, pomai::Metadata* out) const;
    };

    class SegmentBuilder
    {
    public:
        SegmentBuilder(std::string path, uint32_t dim);
        
        pomai::Status Add(pomai::VectorId id, pomai::VectorView vec, bool is_deleted, const pomai::Metadata& meta);
        pomai::Status Add(pomai::VectorId id, pomai::VectorView vec, bool is_deleted);

        pomai::Status Finish();
        
        pomai::Status BuildIndex(uint32_t nlist);

        uint32_t Count() const { return static_cast<uint32_t>(entries_.size()); }

    private:
        struct Entry {
            pomai::VectorId id;
            pomai::VectorView vec;
            bool is_deleted;
            pomai::Metadata meta; // Added
        };
        
        std::string path_;
        uint32_t dim_;
        std::vector<Entry> entries_;
        std::vector<float> zero_buffer_;
    };

} // namespace pomai::table
