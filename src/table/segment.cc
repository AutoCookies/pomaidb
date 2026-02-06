#include "table/segment.h"
#include "util/crc32c.h"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <bit>
#include <array>

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
        : path_(std::move(path)), dim_(dim)
    {
    }

    pomai::Status SegmentBuilder::Add(pomai::VectorId id, std::span<const float> vec, bool is_deleted, const pomai::Metadata& meta)
    {
        Entry e;
        e.id = id;
        e.is_deleted = is_deleted;
        e.meta = meta;

        if (is_deleted)
        {
            // Tombstone: store zeros (deterministic, compressible)
            e.vec.resize(dim_, 0.0f);
        }
        else
        {
            if (vec.size() != dim_)
                return pomai::Status::InvalidArgument("dimension mismatch");
            e.vec.assign(vec.begin(), vec.end());
        }
        
        // Buffer locally for sorting (unavoidable for binary search)
        entries_.push_back(std::move(e));
        return pomai::Status::Ok();
    }

    pomai::Status SegmentBuilder::Add(pomai::VectorId id, std::span<const float> vec, bool is_deleted)
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
        h.version = 3; // V3
        h.count = static_cast<uint32_t>(entries_.size());
        h.dim = dim_;
        
        // Calculate Metadata Offset upfront
        // Header + (Count * EntrySize)
        size_t entry_size = sizeof(uint64_t) + 4 + dim_ * sizeof(float);
        h.metadata_offset = static_cast<uint32_t>(sizeof(SegmentHeader) + entries_.size() * entry_size);

        // Write header
        st = file.PWrite(0, &h, sizeof(h));
        if (!st.ok()) return st;

        uint64_t offset = sizeof(SegmentHeader);
        
        // Incremental CRC computation
        uint32_t crc = 0;
        const uint8_t* header_bytes = reinterpret_cast<const uint8_t*>(&h);
        crc = pomai::util::Crc32c(header_bytes, sizeof(h), crc);

        // Stream entries to disk (one at a time)
        std::vector<uint8_t> entry_buffer;
        entry_buffer.reserve(entry_size);

        for (const auto& e : entries_) {
            entry_buffer.clear();
            
            // ID (8 bytes)
            const uint8_t* p_id = reinterpret_cast<const uint8_t*>(&e.id);
            entry_buffer.insert(entry_buffer.end(), p_id, p_id + sizeof(e.id));
            
            // Flags + Padding (4 bytes: 1 flag + 3 pad)
            uint8_t flags = e.is_deleted ? kFlagTombstone : kFlagNone;
            entry_buffer.push_back(flags);
            entry_buffer.push_back(0);
            entry_buffer.push_back(0);
            entry_buffer.push_back(0);

            // Vector (dim * 4 bytes)
            const uint8_t* p_vec = reinterpret_cast<const uint8_t*>(e.vec.data());
            entry_buffer.insert(entry_buffer.end(), p_vec, p_vec + (dim_ * sizeof(float)));

            // Write entry
            st = file.PWrite(offset, entry_buffer.data(), entry_buffer.size());
            if (!st.ok()) return st;
            
            // Update CRC incrementally
            crc = pomai::util::Crc32c(entry_buffer.data(), entry_buffer.size(), crc);
            
            offset += entry_buffer.size();
        }

        // Write Metadata Block
        // Prepare metadata arrays
        std::vector<uint64_t> offsets;
        std::vector<char> blob;
        offsets.reserve(entries_.size() + 1);
        
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

    pomai::Status SegmentBuilder::BuildSketch(uint32_t block_size) {
         if (block_size == 0) return pomai::Status::InvalidArgument("wsbr block_size must be > 0");

         struct SketchHeader {
             char magic[16];
             uint32_t version;
             uint32_t block_size;
             uint32_t block_count;
             uint32_t reserved;
         };

         struct SketchBlock {
             uint64_t signature;
             uint32_t start_index;
             uint32_t count;
             uint32_t reserved;
         };

         auto sig64 = [&](std::span<const float> vec) -> uint64_t {
             uint64_t out = 0;
             for (uint32_t b = 0; b < 64; ++b) {
                 float sum = 0.0f;
                 for (uint32_t d = 0; d < dim_; ++d) {
                     const uint64_t seed = (static_cast<uint64_t>(b) << 32) ^ static_cast<uint64_t>(d * 2654435761u + 2246822519u);
                     const float sign = ((seed ^ (seed >> 13) ^ (seed >> 29)) & 1ULL) ? 1.0f : -1.0f;
                     sum += vec[d] * sign;
                 }
                 if (sum >= 0.0f) out |= (1ULL << b);
             }
             return out;
         };

         std::vector<SketchBlock> blocks;
         blocks.reserve((entries_.size() + block_size - 1) / block_size);

         for (uint32_t start = 0; start < entries_.size(); start += block_size) {
             const uint32_t count = std::min<uint32_t>(block_size, static_cast<uint32_t>(entries_.size()) - start);
             std::array<int32_t, 64> bit_votes{};
             for (uint32_t i = 0; i < count; ++i) {
                 const auto& e = entries_[start + i];
                 if (e.is_deleted) continue;
                 uint64_t sig = sig64(e.vec);
                 for (uint32_t b = 0; b < 64; ++b) {
                     bit_votes[b] += ((sig >> b) & 1ULL) ? 1 : -1;
                 }
             }
             uint64_t block_sig = 0;
             for (uint32_t b = 0; b < 64; ++b) {
                 if (bit_votes[b] >= 0) block_sig |= (1ULL << b);
             }
             blocks.push_back(SketchBlock{block_sig, start, count, 0});
         }

         SketchHeader h{};
         std::memset(&h, 0, sizeof(h));
         std::memcpy(h.magic, "pomai.wsbr.v1", 13);
         h.version = 1;
         h.block_size = block_size;
         h.block_count = static_cast<uint32_t>(blocks.size());

         std::string sketch_path = path_ + ".wsbr";
         std::string tmp_path = sketch_path + ".tmp";
         pomai::util::PosixFile file;
         auto st = pomai::util::PosixFile::CreateTrunc(tmp_path, &file);
         if (!st.ok()) return st;

         st = file.PWrite(0, &h, sizeof(h));
         if (!st.ok()) return st;
         if (!blocks.empty()) {
             st = file.PWrite(sizeof(h), blocks.data(), blocks.size() * sizeof(SketchBlock));
             if (!st.ok()) return st;
         }
         uint32_t crc = 0;
         crc = pomai::util::Crc32c(&h, sizeof(h), crc);
         if (!blocks.empty()) crc = pomai::util::Crc32c(blocks.data(), blocks.size() * sizeof(SketchBlock), crc);
         st = file.PWrite(sizeof(h) + blocks.size() * sizeof(SketchBlock), &crc, sizeof(crc));
         if (!st.ok()) return st;
         st = file.SyncData();
         if (!st.ok()) return st;
         st = file.Close();
         if (!st.ok()) return st;
         if (rename(tmp_path.c_str(), sketch_path.c_str()) != 0) {
             return pomai::Status::IOError("rename wsbr sketch failed");
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
        
        // Support V2 and V3
        if (h->version != 2 && h->version != 3) {
            return pomai::Status::Corruption("unsupported version");
        }
        
        reader->count_ = h->count;
        reader->dim_ = h->dim;
        reader->entry_size_ = sizeof(uint64_t) + 4 + h->dim * sizeof(float); // ID + Flags/Pad + Vec
        
        if (h->version == 3) {
            reader->metadata_offset_ = h->metadata_offset;
        } else {
            reader->metadata_offset_ = 0;
        }

        reader->base_addr_ = static_cast<const uint8_t*>(data);
        reader->file_size_ = size;

        // Verify size
        size_t expected_min = sizeof(SegmentHeader) + reader->count_ * reader->entry_size_ + 4; // + CRC
        if (size < expected_min) return pomai::Status::Corruption("segment truncated");

        struct SketchHeader {
            char magic[16];
            uint32_t version;
            uint32_t block_size;
            uint32_t block_count;
            uint32_t reserved;
        };

        std::string sketch_path = path + ".wsbr";
        pomai::util::PosixFile sketch_file;
        st = pomai::util::PosixFile::OpenRead(sketch_path, &sketch_file);
        if (!st.ok()) return pomai::Status::Corruption("missing wsbr sketch sidecar");

        const void* sketch_data = nullptr;
        size_t sketch_size = 0;
        st = sketch_file.Map(&sketch_data, &sketch_size);
        if (!st.ok()) return st;
        if (sketch_size < sizeof(SketchHeader) + sizeof(uint32_t)) {
            return pomai::Status::Corruption("wsbr sketch truncated");
        }

        const auto* sh = static_cast<const SketchHeader*>(sketch_data);
        if (std::strncmp(sh->magic, "pomai.wsbr.v1", 13) != 0 || sh->version != 1) {
            return pomai::Status::Corruption("bad wsbr sketch header");
        }
        const size_t blocks_bytes = static_cast<size_t>(sh->block_count) * sizeof(WsbrBlockSketch);
        const size_t expected = sizeof(SketchHeader) + blocks_bytes + sizeof(uint32_t);
        if (sketch_size != expected) return pomai::Status::Corruption("wsbr sketch size mismatch");
        uint32_t file_crc = 0;
        std::memcpy(&file_crc, static_cast<const uint8_t*>(sketch_data) + sizeof(SketchHeader) + blocks_bytes, sizeof(file_crc));
        uint32_t calc_crc = 0;
        calc_crc = pomai::util::Crc32c(sketch_data, sizeof(SketchHeader) + blocks_bytes, calc_crc);
        if (calc_crc != file_crc) return pomai::Status::Corruption("wsbr sketch crc mismatch");

        reader->wsbr_block_size_ = sh->block_size;
        reader->wsbr_blocks_.resize(sh->block_count);
        if (sh->block_count > 0) {
            std::memcpy(reader->wsbr_blocks_.data(), static_cast<const uint8_t*>(sketch_data) + sizeof(SketchHeader), blocks_bytes);
        }

        *out = std::move(reader);
        return pomai::Status::Ok();
    }

    pomai::Status SegmentReader::RouteBlocks(std::uint64_t query_sig,
                                            uint32_t top_blocks,
                                            std::vector<WsbrBlockCandidate>* out_blocks) const
    {
        if (!out_blocks) return pomai::Status::InvalidArgument("out_blocks is null");
        out_blocks->clear();
        if (wsbr_blocks_.empty()) return pomai::Status::Corruption("wsbr sketch has zero blocks");

        std::vector<WsbrBlockCandidate> ranked;
        ranked.reserve(wsbr_blocks_.size());
        for (uint32_t i = 0; i < wsbr_blocks_.size(); ++i) {
            const auto& b = wsbr_blocks_[i];
            ranked.push_back(WsbrBlockCandidate{i, b.start_index, b.count, static_cast<uint32_t>(std::popcount(query_sig ^ b.signature))});
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.hamming != b.hamming) return a.hamming < b.hamming;
            return a.block_id < b.block_id;
        });
        top_blocks = std::min<uint32_t>(top_blocks == 0 ? 1 : top_blocks, static_cast<uint32_t>(ranked.size()));
        out_blocks->assign(ranked.begin(), ranked.begin() + top_blocks);
        return pomai::Status::Ok();
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
        if (count_ == 0) return FindResult::kNotFound;

        int64_t left = 0;
        int64_t right = count_ - 1;
        
        const uint8_t* entries_start = base_addr_ + sizeof(SegmentHeader);

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
                    if(out_vec) *out_vec = {}; // Deleted
                    // Should we return metadata for tombstone? 
                    // Usually tombstones don't have metadata updates attached to them for filtering,
                    // BUT they might hide earlier versions.
                    // Implementation choice: Metadata on tombstone can be useful (e.g. timestamp).
                    // In current Builder, tombstone stores given metadata.
                    if(out_meta) GetMetadata(static_cast<uint32_t>(mid), out_meta);
                    return FindResult::kFoundTombstone;
                }
                
                // Vector starts at offset 8 (ID) + 4 (Flags+Pad) = 12
                const float* vec_ptr = reinterpret_cast<const float*>(p + 12);
                if (out_vec) *out_vec = std::span<const float>(vec_ptr, dim_);
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

    pomai::Status SegmentReader::ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted, pomai::Metadata* out_meta) const
    {
        if (index >= count_) return pomai::Status::InvalidArgument("index out of range");

        const uint8_t* p = base_addr_ + sizeof(SegmentHeader) + index * entry_size_;
        
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
                 const float* vec_ptr = reinterpret_cast<const float*>(p + 12);
                 *out_vec = std::span<const float>(vec_ptr, dim_);
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
