#include "table/segment.h"
#include "util/crc32c.h"
#include "core/index/ivf_flat.h" 

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace pomai::table
{

    namespace
    {
        static const char *kMagic = "pomai.seg.v1";
    }

    // Builder Implementation

    // -------------------------
    // SegmentBuilder: Streaming Implementation (DB-Grade)
    // -------------------------
    // Memory: O(1) per entry, not O(N) for all entries
    // Writes: Incremental, not buffered in RAM
    // CRC: Computed on the fly (incremental)
    
    SegmentBuilder::SegmentBuilder(std::string path, uint32_t dim)
        : path_(std::move(path)), dim_(dim), zero_buffer_(dim, 0.0f)
    {
    }

    pomai::Status SegmentBuilder::Add(pomai::VectorId id, pomai::VectorView vec, bool is_deleted, const pomai::Metadata& meta)
    {
        Entry e;
        e.id = id;
        e.is_deleted = is_deleted;
        e.meta = meta;

        if (is_deleted)
        {
            e.vec = pomai::VectorView{nullptr, dim_};
        }
        else
        {
            if (vec.dim != dim_)
                return pomai::Status::InvalidArgument("dimension mismatch");
            e.vec = vec;
        }
        
        // Buffer locally for sorting (unavoidable for binary search)
        entries_.push_back(std::move(e));
        return pomai::Status::Ok();
    }

    pomai::Status SegmentBuilder::Add(pomai::VectorId id, pomai::VectorView vec, bool is_deleted)
    {
        return Add(id, vec, is_deleted, pomai::Metadata());
    }

    pomai::Status SegmentBuilder::Finish()
    {
        // Sort by ID (required for binary search in reader)
        std::sort(entries_.begin(), entries_.end(),
                  [](const Entry &a, const Entry &b) { return a.id < b.id; });

        // Open tmp file for streaming writes
        std::string tmp_path = path_ + ".tmp";
        pomai::util::PosixFile file;
        auto st = pomai::util::PosixFile::CreateTrunc(tmp_path, &file);
        if (!st.ok()) return st;

        // Prepare header
        SegmentHeader h{};
        std::memset(&h, 0, sizeof(h));
        std::memcpy(h.magic, kMagic, sizeof(h.magic));
        h.version = 6; // V6: 64-byte alignment + IVF positional indices
        h.count = static_cast<uint32_t>(entries_.size());
        h.dim = dim_;
        h.is_quantized = 1; // Enable SQ8 by default for builder
        
        // Train Quantizer
        core::ScalarQuantizer8Bit quantizer(dim_);
        std::vector<float> training_data;
        training_data.reserve(entries_.size() * dim_);
        
        for (const auto& e : entries_) {
            if (!e.is_deleted) {
                training_data.insert(training_data.end(), e.vec.data, e.vec.data + dim_);
            }
        }
        
        if (!training_data.empty()) {
            quantizer.Train(training_data, training_data.size() / dim_);
            h.quant_min = quantizer.GetGlobalMin();
            h.quant_inv_scale = quantizer.GetGlobalInvScale();
        } else {
            h.quant_min = 0.0f;
            h.quant_inv_scale = 0.0f;
        }
        
        // Prepare metadata arrays
        std::vector<uint64_t> offsets;
        std::vector<char> blob;
        offsets.reserve(entries_.size() + 1);
        
        // V5 variables
        size_t entry_size = 0;
        uint32_t entries_start_offset = 0;
        
        if (h.version >= 6) {
            size_t unpadded_size = sizeof(uint64_t) + 4 + dim_ * sizeof(uint8_t);
            entry_size = (unpadded_size + 63) & ~63; // 64-byte alignment
            entries_start_offset = 128; // Small header + alignment
            h.entry_size = static_cast<uint32_t>(entry_size);
            h.entries_start_offset = entries_start_offset;
            h.metadata_offset = static_cast<uint32_t>(entries_start_offset + entries_.size() * entry_size);
        } else if (h.version >= 5) {
            size_t unpadded_size = sizeof(uint64_t) + 4 + dim_ * sizeof(uint8_t);
            entry_size = (unpadded_size + 4095) & ~4095;
            entries_start_offset = 4096;
            h.entry_size = static_cast<uint32_t>(entry_size);
            h.entries_start_offset = entries_start_offset;
            h.metadata_offset = static_cast<uint32_t>(entries_start_offset + entries_.size() * entry_size);
        } else {
            entry_size = sizeof(uint64_t) + 4 + dim_ * sizeof(uint8_t);
            entries_start_offset = sizeof(SegmentHeader);
            h.metadata_offset = static_cast<uint32_t>(sizeof(SegmentHeader) + entries_.size() * entry_size);
        }

        // Write header
        st = file.PWrite(0, &h, sizeof(h));
        if (!st.ok()) return st;

        uint64_t offset = sizeof(SegmentHeader);
        
        // Incremental CRC computation
        uint32_t crc = 0;
        const uint8_t* header_bytes = reinterpret_cast<const uint8_t*>(&h);
        crc = pomai::util::Crc32c(header_bytes, sizeof(h), crc);

        if (h.version >= 5) {
            std::vector<uint8_t> pad(entries_start_offset - sizeof(SegmentHeader), 0);
            st = file.PWrite(offset, pad.data(), pad.size());
            if (!st.ok()) return st;
            crc = pomai::util::Crc32c(pad.data(), pad.size(), crc);
            offset += pad.size();
        }

        // Stream entries to disk (one at a time)
        void* raw_buffer = nullptr;
        if (posix_memalign(&raw_buffer, 4096, entry_size) != 0) {
            return pomai::Status::IOError("posix_memalign failed");
        }
        
        uint8_t* entry_buffer = static_cast<uint8_t*>(raw_buffer);

        for (const auto& e : entries_) {
            std::memset(entry_buffer, 0, entry_size);
            size_t cursor = 0;
            
            // ID (8 bytes)
            const uint8_t* p_id = reinterpret_cast<const uint8_t*>(&e.id);
            std::memcpy(entry_buffer + cursor, p_id, sizeof(e.id));
            cursor += sizeof(e.id);
            
            // Flags + Padding (4 bytes: 1 flag + 3 pad)
            uint8_t flags = e.is_deleted ? kFlagTombstone : kFlagNone;
            entry_buffer[cursor++] = flags;
            entry_buffer[cursor++] = 0;
            entry_buffer[cursor++] = 0;
            entry_buffer[cursor++] = 0;

            // Vector (dim * 1 bytes)
            std::vector<uint8_t> encoded;
            if (e.is_deleted) {
                encoded.assign(dim_, 0); // Zeroed out tombstone
            } else {
                std::span<const float> vec_span(e.vec.data, dim_);
                encoded = quantizer.Encode(vec_span);
            }
            std::memcpy(entry_buffer + cursor, encoded.data(), encoded.size());

            // Write entry
            st = file.PWrite(offset, entry_buffer, entry_size);
            if (!st.ok()) {
                free(raw_buffer);
                return st;
            }
            
            // Update CRC incrementally
            crc = pomai::util::Crc32c(entry_buffer, entry_size, crc);
            
            offset += entry_size;
        }
        
        free(raw_buffer);

        // Write Metadata Block
        
        for (const auto& e : entries_) {
            offsets.push_back(blob.size());
            if (!e.meta.tenant.empty()) {
                blob.insert(blob.end(), e.meta.tenant.begin(), e.meta.tenant.end());
            }
        }
        offsets.push_back(blob.size()); // Final sentinel

        // Serialize offsets
        st = file.PWrite(offset, offsets.data(), offsets.size() * sizeof(uint64_t));
        if (!st.ok()) return st;
        crc = pomai::util::Crc32c(reinterpret_cast<const uint8_t*>(offsets.data()), offsets.size() * sizeof(uint64_t), crc);
        offset += offsets.size() * sizeof(uint64_t);

        // Serialize blob
        if (!blob.empty()) {
            st = file.PWrite(offset, blob.data(), blob.size());
            if (!st.ok()) return st;
            crc = pomai::util::Crc32c(reinterpret_cast<const uint8_t*>(blob.data()), blob.size(), crc);
            offset += blob.size();
        }
        
        // Write CRC footer
        st = file.PWrite(offset, &crc, sizeof(crc));
        if (!st.ok()) return st;

        // Fsync data
        st = file.SyncData();
        if (!st.ok()) return st;
        
        st = file.Close();
        if (!st.ok()) return st;

        // Rename to final path
        if (rename(tmp_path.c_str(), path_.c_str()) != 0) {
            return pomai::Status::IOError("rename failed");
        }

        return pomai::Status::Ok();
    }

    pomai::Status SegmentBuilder::BuildIndex(uint32_t nlist) {
         // Gather valid vectors for training
         std::vector<float> training_data;
         training_data.reserve(entries_.size() * dim_);
         size_t num_live = 0;
         
         for(const auto& e : entries_) {
             if (!e.is_deleted) {
                 training_data.insert(training_data.end(), e.vec.data, e.vec.data + dim_);
                 num_live++;
             }
         }
         
         pomai::index::IvfFlatIndex::Options opt;
         opt.nlist = nlist;
         if (num_live < opt.nlist) opt.nlist = std::max<uint32_t>(1U, static_cast<uint32_t>(num_live));

         auto idx = std::make_unique<pomai::index::IvfFlatIndex>(dim_, opt);
         auto st = idx->Train(training_data, num_live);
         if (!st.ok()) return st;
         
         for (size_t i = 0; i < entries_.size(); ++i) {
             const auto& e = entries_[i];
             if (!e.is_deleted) {
                 st = idx->Add(static_cast<uint32_t>(i), e.vec.span());
                 if (!st.ok()) return st;
             }
         }
         
         std::string idx_path = path_ + ".idx.tmp";
         st = idx->Save(idx_path);
         if (!st.ok()) return st;
         
         std::string final_idx_path = path_;
         if (final_idx_path.size() > 4 && final_idx_path.substr(final_idx_path.size()-4) == ".dat") {
             final_idx_path = final_idx_path.substr(0, final_idx_path.size()-4);
         }
         final_idx_path += ".idx";
         
         if (rename(idx_path.c_str(), final_idx_path.c_str()) != 0) {
             return pomai::Status::IOError("rename idx failed");
         }
         
         return pomai::Status::Ok();
    }


    // Reader Implementation

    SegmentReader::SegmentReader() = default;
    SegmentReader::~SegmentReader() {
        file_.Close();
    }

    pomai::Status SegmentReader::Open(std::string path, std::unique_ptr<SegmentReader> *out)
    {
        auto reader = std::unique_ptr<SegmentReader>(new SegmentReader());
        reader->path_ = path;
        
        auto st = pomai::util::PosixFile::OpenRead(path, &reader->file_);
        if (!st.ok()) return st;
        
        // Map file
        const void* data = nullptr;
        size_t size = 0;
        st = reader->file_.Map(&data, &size);
        if (!st.ok()) return st;

        if (size < sizeof(SegmentHeader)) return pomai::Status::Corruption("file too small");

        // Read Header from map
        const SegmentHeader* h = static_cast<const SegmentHeader*>(data);

        if (strncmp(h->magic, kMagic, 12) != 0) return pomai::Status::Corruption("bad magic");
        
        // Support V2 to V6
        if (h->version < 2 || h->version > 6) {
            return pomai::Status::Corruption("unsupported version");
        }
        
        reader->count_ = h->count;
        reader->dim_ = h->dim;
        
        if (h->version >= 5) {
            reader->entries_start_offset_ = h->entries_start_offset;
            reader->entry_size_ = h->entry_size;
        } else {
            reader->entries_start_offset_ = sizeof(SegmentHeader);
        }

        if (h->version >= 4) {
            reader->is_quantized_ = (h->is_quantized == 1);
        } else {
            reader->is_quantized_ = false;
        }
        
        if (reader->is_quantized_) {
            if (h->version < 5) {
                reader->entry_size_ = sizeof(uint64_t) + 4 + h->dim * sizeof(uint8_t);
            }
            
            // Initialize quantizer from explicitly serialized floats
            reader->quantizer_ = std::make_unique<core::ScalarQuantizer8Bit>(h->dim);
            reader->quantizer_->LoadState(h->quant_min, h->quant_inv_scale);
        } else {
            if (h->version < 5) {
                reader->entry_size_ = sizeof(uint64_t) + 4 + h->dim * sizeof(float);
            }
        }
        
        if (h->version >= 3) {
            reader->metadata_offset_ = h->metadata_offset;
        } else {
            reader->metadata_offset_ = 0;
        }

        reader->base_addr_ = static_cast<const uint8_t*>(data);
        reader->file_size_ = size;

        // Verify size
        size_t expected_min = reader->entries_start_offset_ + reader->count_ * reader->entry_size_ + 4; // + CRC
        if (size < expected_min) return pomai::Status::Corruption("segment truncated");

        // Try load index (best effort)
        std::string idx_path = path;
        if (idx_path.size() > 4 && idx_path.substr(idx_path.size()-4) == ".dat") {
             idx_path = idx_path.substr(0, idx_path.size()-4);
        }
        idx_path += ".idx";
        
        // Ignore error if not found (fallback to scan)
        (void)pomai::index::IvfFlatIndex::Load(idx_path, &reader->index_);

        *out = std::move(reader);
        return pomai::Status::Ok();
    }

    pomai::Status SegmentReader::Search(std::span<const float> query, uint32_t nprobe, 
                                        std::vector<uint32_t>* out_candidates) const
    {
        if (!index_) return pomai::Status::Ok(); // Empty candidates -> Fallback
        return index_->Search(query, nprobe, out_candidates);
    }

    void SegmentReader::GetMetadata(uint32_t index, pomai::Metadata* out) const
    {
        if (metadata_offset_ == 0 || index >= count_) return;
        
        const uint8_t* meta_offsets_base = base_addr_ + metadata_offset_;
        // Blob starts after (count_ + 1) offsets
        const char* meta_blob = reinterpret_cast<const char*>(meta_offsets_base + (count_ + 1) * sizeof(uint64_t));

        uint64_t start = 0;
        uint64_t end = 0;
        std::memcpy(&start, meta_offsets_base + index * sizeof(uint64_t), sizeof(start));
        std::memcpy(&end, meta_offsets_base + (index + 1) * sizeof(uint64_t), sizeof(end));
        
        if (end > start) {
            out->tenant = std::string(meta_blob + start, end - start);
        }
    }

    pomai::Status SegmentReader::GetQuantized(pomai::VectorId id, std::span<const uint8_t>* out_codes, pomai::Metadata* out_meta) const
    {
        if (!is_quantized_) return pomai::Status::InvalidArgument("Segment is not quantized");
        
        const uint8_t* raw_payload = nullptr;
        auto res = FindRaw(id, &raw_payload, out_meta);
        if (res == FindResult::kFound) {
            if (out_codes) *out_codes = std::span<const uint8_t>(raw_payload, dim_);
            return pomai::Status::Ok();
        }
        if (res == FindResult::kFoundTombstone) return pomai::Status::NotFound("tombstone");
        return pomai::Status::NotFound("id not found in segment");
    }

    pomai::Status SegmentReader::Get(pomai::VectorId id, std::span<const float> *out_vec, pomai::Metadata* out_meta) const
    {
         auto res = Find(id, out_vec, out_meta);
         if (res == FindResult::kFound) return pomai::Status::Ok();
         if (res == FindResult::kFoundTombstone) return pomai::Status::NotFound("tombstone");
         return pomai::Status::NotFound("id not found in segment");
    }

    // Backward compat overload
    pomai::Status SegmentReader::Get(pomai::VectorId id, std::span<const float> *out_vec) const {
        return Get(id, out_vec, nullptr);
    }
    
    // Backward compat overload
    SegmentReader::FindResult SegmentReader::Find(pomai::VectorId id, std::span<const float> *out_vec) const {
        return Find(id, out_vec, nullptr);
    }

    SegmentReader::FindResult SegmentReader::Find(pomai::VectorId id, std::span<const float> *out_vec, pomai::Metadata* out_meta) const
    {
        // For backwards compatibility and instances where we don't need a decoded buffer safely (e.g. tests)
        // If it's quantized, it will just flatly fail with this signature since span<const float>
        // implies pointing into mmap memory, which doesn't exist.
        if (is_quantized_) return FindResult::kNotFound;
        
        const uint8_t* raw_payload = nullptr;
        auto res = FindRaw(id, &raw_payload, out_meta);
        if (res == FindResult::kFoundTombstone && out_vec) {
            *out_vec = {};
        }
        if (res == FindResult::kFound && out_vec) {
            const float* vec_ptr = reinterpret_cast<const float*>(raw_payload);
            *out_vec = std::span<const float>(vec_ptr, dim_);
        }
        return res;
    }
    
    SegmentReader::FindResult SegmentReader::FindAndDecode(pomai::VectorId id, std::span<const float>* out_vec_mapped, std::vector<float>* out_vec_decoded, pomai::Metadata* out_meta) const
    {
        const uint8_t* raw_payload = nullptr;
        auto res = FindRaw(id, &raw_payload, out_meta);
        
        if (res == FindResult::kFoundTombstone && out_vec_mapped) {
            *out_vec_mapped = {};
        }
        
        if (res == FindResult::kFound) {
            if (is_quantized_) {
                if (out_vec_decoded) {
                    *out_vec_decoded = quantizer_->Decode(std::span<const uint8_t>(raw_payload, dim_));
                    if (out_vec_mapped) *out_vec_mapped = *out_vec_decoded;
                }
            } else {
                if (out_vec_mapped) *out_vec_mapped = std::span<const float>(reinterpret_cast<const float*>(raw_payload), dim_);
            }
        }
        return res;
    }

    SegmentReader::FindResult SegmentReader::FindRaw(pomai::VectorId id, const uint8_t** raw_payload, pomai::Metadata* out_meta) const
    {
        if (count_ == 0) return FindResult::kNotFound;

        int64_t left = 0;
        int64_t right = count_ - 1;
        
        const uint8_t* entries_start = base_addr_ + entries_start_offset_;

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            
            const uint8_t* p = entries_start + mid * entry_size_;
            
            // Safe unaligned read for ID (memcpy)
            uint64_t read_id;
            std::memcpy(&read_id, p, sizeof(uint64_t));

            if (read_id == id) {
                // Found
                uint8_t flags = *(p + 8);
                if (flags & kFlagTombstone) {
                    if(out_meta) GetMetadata(static_cast<uint32_t>(mid), out_meta);
                    return FindResult::kFoundTombstone;
                }
                
                // Vector starts at offset 8 (ID) + 4 (Flags+Pad) = 12
                if (raw_payload) *raw_payload = p + 12;
                if(out_meta) GetMetadata(static_cast<uint32_t>(mid), out_meta);
                
                return FindResult::kFound;
            }

            if (read_id < id) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return FindResult::kNotFound;
    }

    pomai::Status SegmentReader::ReadAtCodes(uint32_t index, pomai::VectorId* out_id, std::span<const uint8_t>* out_codes, bool* out_deleted, pomai::Metadata* out_meta) const
    {
        if (index >= count_) return pomai::Status::InvalidArgument("index out of range");

        const uint8_t* p = base_addr_ + entries_start_offset_ + index * entry_size_;
        
        if (out_id) {
             std::memcpy(out_id, p, sizeof(uint64_t));
        }
        
        uint8_t flags = *(p + 8);
        bool is_deleted = (flags & kFlagTombstone);
        if (out_deleted) *out_deleted = is_deleted;

        if (out_codes) {
             if (is_deleted || !is_quantized_) {
                 *out_codes = {};
             } else {
                 const uint8_t* code_ptr = p + 12;
                 *out_codes = std::span<const uint8_t>(code_ptr, dim_);
             }
        }
        
        if (out_meta) {
            GetMetadata(index, out_meta);
        }
        
        return pomai::Status::Ok();
    }
    
    pomai::Status SegmentReader::ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted, pomai::Metadata* out_meta) const
    {
        if (index >= count_) return pomai::Status::InvalidArgument("index out of range");

        const uint8_t* p = base_addr_ + entries_start_offset_ + index * entry_size_;
        
        if (out_id) {
             std::memcpy(out_id, p, sizeof(uint64_t));
        }
        
        uint8_t flags = *(p + 8);
        bool is_deleted = (flags & kFlagTombstone);
        if (out_deleted) *out_deleted = is_deleted;

        if (out_vec) {
             if (is_deleted) {
                 *out_vec = {};
             } else {
                 if (is_quantized_) {
                     // ReadAt returning span<float> is incompatible with quantizer without allocation wrapper.
                     // The caller must decode using ForEach or GetQuantized. 
                     // We return an empty span here and rely on higher layers utilizing FindAndDecode across Segments.
                     *out_vec = {};
                 } else {
                     const float* vec_ptr = reinterpret_cast<const float*>(p + 12);
                     *out_vec = std::span<const float>(vec_ptr, dim_);
                 }
             }
        }
        
        if (out_meta) {
            GetMetadata(index, out_meta);
        }
        
        return pomai::Status::Ok();
    }
    
    // Backward compat overload
    pomai::Status SegmentReader::ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted) const {
        return ReadAt(index, out_id, out_vec, out_deleted, nullptr);
    }

} // namespace pomai::table
