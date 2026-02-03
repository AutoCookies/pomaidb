#include "table/segment.h"
#include "util/crc32c.h"

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

    SegmentBuilder::SegmentBuilder(std::string path, uint32_t dim)
        : path_(std::move(path)), dim_(dim)
    {
    }

    pomai::Status SegmentBuilder::Add(pomai::VectorId id, std::span<const float> vec, bool is_deleted)
    {
        Entry e;
        e.id = id;
        e.is_deleted = is_deleted;

        if (is_deleted)
        {
            // Tombstone: store zeros (or uninit, but zeros compresses better and is deterministic)
            e.vec.resize(dim_, 0.0f);
        }
        else
        {
            if (vec.size() != dim_)
                return pomai::Status::InvalidArgument("dimension mismatch");
            e.vec.assign(vec.begin(), vec.end());
        }
        entries_.push_back(std::move(e));
        return pomai::Status::Ok();
    }

    pomai::Status SegmentBuilder::Finish()
    {
        // Sort by ID
        std::sort(entries_.begin(), entries_.end(),
                  [](const Entry &a, const Entry &b) { return a.id < b.id; });

        // Prepare Header
        SegmentHeader h{};
        std::memset(&h, 0, sizeof(h));
        std::memcpy(h.magic, kMagic, sizeof(h.magic));
        h.version = 2; // V2
        h.count = static_cast<uint32_t>(entries_.size());
        h.dim = dim_;

        // Write to tmp file
        std::string tmp_path = path_ + ".tmp";
        
        std::vector<uint8_t> buffer;
        // Entry: ID(8) + Flags(1) + Pad(3) + Vec(dim*4)
        size_t entry_size = sizeof(uint64_t) + 4 + dim_ * sizeof(float);
        size_t total_size = sizeof(SegmentHeader) + entries_.size() * entry_size + 4;
        
        try {
            buffer.reserve(total_size);
        } catch(...) {
            return pomai::Status::ResourceExhausted("failed to allocate write buffer");
        }

        // Header
        const uint8_t* p_header = reinterpret_cast<const uint8_t*>(&h);
        buffer.insert(buffer.end(), p_header, p_header + sizeof(h));

        // Entries
        for (const auto& e : entries_) {
             // ID
             const uint8_t* p_id = reinterpret_cast<const uint8_t*>(&e.id);
             buffer.insert(buffer.end(), p_id, p_id + sizeof(e.id));
             
             // Flags + Padding
             uint8_t flags = e.is_deleted ? kFlagTombstone : kFlagNone;
             buffer.push_back(flags);
             // Pad 3 bytes (zeros)
             buffer.push_back(0);
             buffer.push_back(0);
             buffer.push_back(0);

             // Vector
             const uint8_t* p_vec = reinterpret_cast<const uint8_t*>(e.vec.data());
             buffer.insert(buffer.end(), p_vec, p_vec + (dim_ * sizeof(float)));
        }

        // CRC
        uint32_t crc = pomai::util::Crc32c(buffer.data(), buffer.size());
        const uint8_t* p_crc = reinterpret_cast<const uint8_t*>(&crc);
        buffer.insert(buffer.end(), p_crc, p_crc + sizeof(crc));

        // Write
        pomai::util::PosixFile file;
        auto st = pomai::util::PosixFile::CreateTrunc(tmp_path, &file);
        if (!st.ok()) return st;
        
        st = file.PWrite(0, buffer.data(), buffer.size());
        if (!st.ok()) return st;
        
        st = file.SyncData();
        if (!st.ok()) return st;
        
        st = file.Close();
        if (!st.ok()) return st;

        // Rename
        if (rename(tmp_path.c_str(), path_.c_str()) != 0) {
            return pomai::Status::IOError("rename failed");
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
        if (h->version != 2) return pomai::Status::Corruption("unsupported version");
        
        reader->count_ = h->count;
        reader->dim_ = h->dim;
        reader->entry_size_ = sizeof(uint64_t) + 4 + h->dim * sizeof(float); // ID + Flags/Pad + Vec
        
        reader->base_addr_ = static_cast<const uint8_t*>(data);
        reader->file_size_ = size;

        // Verify size
        size_t expected_min = sizeof(SegmentHeader) + reader->count_ * reader->entry_size_ + 4; // + CRC
        if (size < expected_min) return pomai::Status::Corruption("segment truncated");

        *out = std::move(reader);
        return pomai::Status::Ok();
    }

    pomai::Status SegmentReader::Get(pomai::VectorId id, std::span<const float> *out_vec) const
    {
         auto res = Find(id, out_vec);
         if (res == FindResult::kFound) return pomai::Status::Ok();
         if (res == FindResult::kFoundTombstone) return pomai::Status::NotFound("tombstone");
         return pomai::Status::NotFound("id not found in segment");
    }

    SegmentReader::FindResult SegmentReader::Find(pomai::VectorId id, std::span<const float> *out_vec) const
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
                    return FindResult::kFoundTombstone;
                }
                
                // Vector starts at offset 8 (ID) + 4 (Flags+Pad) = 12
                const float* vec_ptr = reinterpret_cast<const float*>(p + 12);
                if (out_vec) *out_vec = std::span<const float>(vec_ptr, dim_);
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

    pomai::Status SegmentReader::ReadAt(uint32_t index, pomai::VectorId* out_id, std::span<const float>* out_vec, bool* out_deleted) const
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
        return pomai::Status::Ok();
    }

} // namespace pomai::table
