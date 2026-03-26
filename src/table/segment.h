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
#include "pomai/options.h"
#include "pomai/quantization/scalar_quantizer.h"
#include "pomai/quantization/half_float_quantizer.h"
#include "core/quantization/pomai_pq.h"
#include "core/storage/io_provider.h"
#include "util/slice.h"

// Forward declare in correct namespace
namespace pomai::index { class HnswIndex; class IvfFlatIndex; }
namespace pomai::core { class LexicalIndex; struct LexicalHit; }

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
        char magic[12]; // "pomai.seg.v2" 
        uint32_t version; // 6
        uint32_t count;
        uint32_t dim;
        uint32_t metadata_offset; // structured metadata (src_vid, timestamp, tenant)
        uint8_t  quant_type;      
        uint8_t  reserved1[3];
        float    quant_min;       
        float    quant_inv_scale; 
        uint32_t entries_start_offset; 
        uint32_t entry_size;           
        uint32_t temporal_index_offset; // V6+: Sorted (timestamp, index) pairs
        uint32_t reserved2[1];
    };

    // Flags
    constexpr uint8_t kFlagNone = 0;
    constexpr uint8_t kFlagTombstone = 1 << 0;

    class SegmentReader
    {
    public:
        /** Deleter for palloc-backed SegmentReader (use with unique_ptr). */
        static void PallocDeleter(SegmentReader* p);

        using Ptr = std::unique_ptr<SegmentReader, void(*)(SegmentReader*)>;

        static pomai::Status Open(std::string path, Ptr* out);

        ~SegmentReader();

        // Zero-copy lookup using PinnableSlice
        pomai::Status Get(pomai::VectorId id, pomai::PinnableSlice* out_vec, pomai::Metadata* out_meta) const;
        pomai::Status Get(pomai::VectorId id, pomai::PinnableSlice* out_vec) const;
        
        // V4: Quantized raw lookup
        pomai::QuantizationType GetQuantType() const { return quant_type_; }
        const core::VectorQuantizer<float>* GetQuantizer() const { return quantizer_.get(); }
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

        // Temporal Range Search
        pomai::Status SearchTemporal(uint64_t start_ts, uint64_t end_ts, std::vector<uint32_t>* out_indices) const;

        // Lexical / Keyword Search (BM25)
        pomai::Status SearchLexical(const std::string& query, uint32_t topk, std::vector<core::LexicalHit>* out) const;
        
        // Read entry at index [0, Count()-1]
        pomai::Status ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted, pomai::Metadata* out_meta) const;
        pomai::Status ReadAtCodes(uint32_t index, pomai::VectorId* out_id, std::span<const uint8_t>* out_codes, bool* out_deleted, pomai::Metadata* out_meta) const;
        pomai::Status ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted) const;
        pomai::Status ReadAtMetadata(uint32_t index, bool* out_deleted, pomai::Metadata* out_meta) const;


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
                         const uint8_t* blob_ptr = reinterpret_cast<const uint8_t*>(meta_blob + start);
                        std::memcpy(&meta_obj.src_vid, blob_ptr, 8);
                        std::memcpy(&meta_obj.timestamp, blob_ptr + 8, 8);
                        std::memcpy(&meta_obj.lsn, blob_ptr + 16, 8);
                        std::memcpy(&meta_obj.lat, blob_ptr + 24, 8);
                        std::memcpy(&meta_obj.lon, blob_ptr + 32, 8);
                         
                        size_t cursor = 40;
                         auto read_str = [&](std::string* s) {
                             uint32_t len = 0;
                             std::memcpy(&len, blob_ptr + cursor, 4);
                             cursor += 4;
                             if (len > 0) {
                                 *s = std::string(reinterpret_cast<const char*>(blob_ptr + cursor), len);
                                 cursor += len;
                             } else {
                                 *s = "";
                             }
                         };
                         
                        read_str(&meta_obj.device_id);
                        read_str(&meta_obj.location_id);
                        read_str(&meta_obj.tenant);
                         read_str(&meta_obj.text);
                         read_str(&meta_obj.payload);
                         meta_ptr = &meta_obj;
                     }
                 }
                 
                 std::vector<float> decoded;
                 std::span<const float> vec_span;
                 
                 if (!is_deleted) {
                     if (quant_type_ == pomai::QuantizationType::kPq8 && pq_) {
                         // Decode PQ codes back to float (used during flush/iteration)
                         decoded.resize(dim_);
                         pq_->Decode(static_cast<const uint8_t*>(vec_ptr), decoded.data());
                         vec_span = decoded;
                     } else if (quant_type_ != pomai::QuantizationType::kNone && quantizer_) {
                         size_t codes_bytes = dim_;
                         if (quant_type_ == pomai::QuantizationType::kFp16) codes_bytes *= 2;
                         else if (quant_type_ == pomai::QuantizationType::kBit) codes_bytes = (dim_ + 7) / 8;
                         std::span<const uint8_t> codes(static_cast<const uint8_t*>(vec_ptr), codes_bytes);
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

        const pomai::index::HnswIndex* GetHnswIndex() const { return hnsw_index_.get(); }
        bool HasIndex() const { return index_ != nullptr; }
        bool HasHnswIndex() const { return hnsw_index_ != nullptr; }
        /// Returns the Product Quantizer loaded from the .pq sidecar (nullptr if not present).
        const core::ProductQuantizer* GetPqQuantizer() const { return pq_.get(); }

    private:
        SegmentReader();

        std::string path_;
        std::unique_ptr<storage::MemoryMappedFile> mmap_file_;
        uint32_t count_ = 0;
        uint32_t dim_ = 0;
        std::size_t entry_size_ = 0;
        uint32_t entries_start_offset_ = 0;
        uint32_t metadata_offset_ = 0;
        uint32_t temporal_index_offset_ = 0;
        
        // V4: Quantization properties
        pomai::QuantizationType quant_type_{pomai::QuantizationType::kNone};
        std::unique_ptr<core::VectorQuantizer<float>> quantizer_;
        
        const uint8_t* base_addr_ = nullptr;
        std::size_t file_size_ = 0;
        
        std::unique_ptr<pomai::index::IvfFlatIndex> index_;
        std::unique_ptr<pomai::index::HnswIndex> hnsw_index_;
        // kPq8: Product Quantizer loaded from .pq sidecar for ADC scoring.
        std::unique_ptr<core::ProductQuantizer> pq_;
        
        // V7: Lexical Index support
        mutable std::unique_ptr<core::LexicalIndex> lexical_index_;
        
        // Internal helpers
        void GetMetadata(uint32_t index, pomai::Metadata* out) const;
        void EnsureLexicalIndexBuilt() const;
    };

    class SegmentBuilder
    {
    public:
        SegmentBuilder(std::string path, uint32_t dim, pomai::IndexParams index_params = {}, pomai::MetricType metric = pomai::MetricType::kL2);
        
        pomai::Status Add(pomai::VectorId id, pomai::VectorView vec, bool is_deleted, const pomai::Metadata& meta);
        pomai::Status Add(pomai::VectorId id, pomai::VectorView vec, bool is_deleted);

        pomai::Status Finish();
        
        pomai::Status BuildIndex();

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
        pomai::IndexParams index_params_;
        pomai::MetricType metric_;
        std::vector<Entry> entries_;
        std::vector<float> zero_buffer_;
    };

} // namespace pomai::table
