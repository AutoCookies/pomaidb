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

    pomai::Status SegmentBuilder::Add(pomai::VectorId id, std::span<const float> vec)
    {
        if (vec.size() != dim_)
            return pomai::Status::InvalidArgument("dimension mismatch");
        
        Entry e;
        e.id = id;
        e.vec.assign(vec.begin(), vec.end());
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
        h.version = 1;
        h.count = static_cast<uint32_t>(entries_.size());
        h.dim = dim_;

        // Write to tmp file
        std::string tmp_path = path_ + ".tmp";
        
        // We use C++ fstream or PosixFile? PosixFile is better for consistency but it is append-only/pwrite?
        // Let's use std::ofstream for sequential writing for builder, simpler.
        // Actually, we need to calculate CRC.
        
        std::vector<uint8_t> buffer;
        // Estimate size: Header + N * (8 + 4*dim) + 4
        size_t entry_size = sizeof(uint64_t) + dim_ * sizeof(float);
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
             const uint8_t* p_id = reinterpret_cast<const uint8_t*>(&e.id);
             buffer.insert(buffer.end(), p_id, p_id + sizeof(e.id));
             
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
        if (h->version != 1) return pomai::Status::Corruption("unsupported version");
        
        reader->count_ = h->count;
        reader->dim_ = h->dim;
        reader->entry_size_ = sizeof(uint64_t) + h->dim * sizeof(float);
        
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
        if (count_ == 0) return pomai::Status::NotFound("empty segment");

        // Binary Search
        int64_t left = 0;
        int64_t right = count_ - 1;
        
        const uint8_t* entries_start = base_addr_ + sizeof(SegmentHeader);

        while (left <= right) {
            int64_t mid = left + (right - left) / 2;
            
            // Offset = Header + mid * entry_size
            const uint8_t* p = entries_start + mid * entry_size_;
            
            // Read ID (unaligned access safe on x86, but use memcpy for safety/portability or assuming packed/aligned enough)
            // ID is 8 bytes. Header is 12+4+4+4+32 = 56 bytes? 
            // struct Header { char magic[12]; u32 version; u32 count; u32 dim; u32 reserved[8]; }
            // 12 + 4 + 4 + 4 + 32 = 56. 
            // 56 is divisible by 8. So IDs are 8-byte aligned if file is.
            // float vectors follows 8-byte ID, so they are 4-byte aligned. Safe.
            
            uint64_t read_id = *reinterpret_cast<const uint64_t*>(p);

            if (read_id == id) {
                const float* vec_ptr = reinterpret_cast<const float*>(p + sizeof(uint64_t));
                *out_vec = std::span<const float>(vec_ptr, dim_);
                return pomai::Status::Ok();
            }

            if (read_id < id) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return pomai::Status::NotFound("id not found in segment");
    }

} // namespace pomai::table
