#include "storage/wal/wal.h"

#include <filesystem>
#include <vector>
#include <cstring>
#include <cerrno>
#include <sys/uio.h>

#include "table/memtable.h"
#include "util/crc32c.h"
#include "util/posix_file.h"

namespace fs = std::filesystem;

namespace pomai::storage
{

    enum class Op : std::uint8_t
    {
        kPut = 1,
        kDel = 2,
        kPutMeta = 3
    };

#pragma pack(push, 1)
    struct FrameHeader
    {
        std::uint32_t len; // bytes after this header: [type+payload+crc]
    };

    struct WalFileHeader
    {
        char magic[12]; // "pomai.wal.v1"
        std::uint32_t version;
    };

    struct RecordPrefix
    {
        std::uint64_t seq;
        std::uint8_t op;
        std::uint64_t id;
        std::uint32_t dim; // PUT only, else 0
    };
#pragma pack(pop)

    constexpr char kWalMagic[] = "pomai.wal.v1";
    constexpr std::uint32_t kWalVersion = 1;

    class Wal::Impl
    {
    public:
        pomai::util::PosixFile file;
        std::string path;
    };

    Wal::Wal(std::string db_path,
             std::uint32_t shard_id,
             std::size_t segment_bytes,
             pomai::FsyncPolicy fsync)
        : db_path_(std::move(db_path)),
          shard_id_(shard_id),
          segment_bytes_(segment_bytes),
          fsync_(fsync) {}

    Wal::~Wal()
    {
        if (impl_)
        {
            (void)impl_->file.Close();
            delete impl_;
            impl_ = nullptr;
        }
    }

    std::string Wal::SegmentPath(std::uint64_t gen) const
    {
        return (fs::path(db_path_) / ("wal_" + std::to_string(shard_id_) + "_" + std::to_string(gen) + ".log")).string();
    }

    pomai::Status Wal::Open()
    {
        fs::create_directories(db_path_);

        gen_ = 0;
        while (fs::exists(SegmentPath(gen_)))
            ++gen_;

        impl_ = new Impl();
        impl_->path = SegmentPath(gen_);

        auto st = pomai::util::PosixFile::OpenAppend(impl_->path, &impl_->file);
        if (!st.ok())
        {
            delete impl_;
            impl_ = nullptr;
            return st;
        }

        // Determine current size and append offset
        std::error_code ec;
        std::uint64_t sz = 0;
        if (fs::exists(impl_->path, ec))
            sz = static_cast<std::uint64_t>(fs::file_size(impl_->path, ec));

        if (sz == 0)
        {
            WalFileHeader hdr{};
            std::memcpy(hdr.magic, kWalMagic, sizeof(hdr.magic));
            hdr.version = kWalVersion;
            st = impl_->file.PWrite(0, &hdr, sizeof(hdr));
            if (!st.ok())
            {
                delete impl_;
                impl_ = nullptr;
                return st;
            }
            file_off_ = sizeof(WalFileHeader);
            bytes_in_seg_ = sizeof(WalFileHeader);
        }
        else
        {
            file_off_ = sz;
            bytes_in_seg_ = static_cast<std::size_t>(sz);
        }
        return pomai::Status::Ok();
    }

    pomai::Status Wal::RotateIfNeeded(std::size_t add_bytes)
    {
        if (bytes_in_seg_ + add_bytes <= segment_bytes_)
            return pomai::Status::Ok();

        if (impl_)
        {
            (void)impl_->file.SyncData();
            (void)impl_->file.Close();
            delete impl_;
            impl_ = nullptr;
        }

        ++gen_;
        file_off_ = 0;
        bytes_in_seg_ = 0;

        impl_ = new Impl();
        impl_->path = SegmentPath(gen_);
        auto st = pomai::util::PosixFile::OpenAppend(impl_->path, &impl_->file);
        if (!st.ok())
        {
            delete impl_;
            impl_ = nullptr;
            return st;
        }
        WalFileHeader hdr{};
        std::memcpy(hdr.magic, kWalMagic, sizeof(hdr.magic));
        hdr.version = kWalVersion;
        st = impl_->file.PWrite(0, &hdr, sizeof(hdr));
        if (!st.ok())
            return st;
        file_off_ = sizeof(WalFileHeader);
        bytes_in_seg_ = sizeof(WalFileHeader);
        return pomai::Status::Ok();
    }

    static void AppendBytes(std::vector<std::uint8_t> *dst, const void *p, std::size_t n)
    {
        const auto *b = static_cast<const std::uint8_t *>(p);
        dst->insert(dst->end(), b, b + n);
    }

    static pomai::Status PWritevAll(int fd, std::uint64_t off, std::vector<iovec> iovecs)
    {
        std::size_t idx = 0;
        while (idx < iovecs.size())
        {
            ssize_t w = ::pwritev(fd, &iovecs[idx], static_cast<int>(iovecs.size() - idx),
                                  static_cast<off_t>(off));
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                return pomai::Status::IoError(std::string("pwritev: ") + std::strerror(errno));
            }
            if (w == 0)
                return pomai::Status::IoError("pwritev: wrote 0 bytes");

            off += static_cast<std::uint64_t>(w);
            std::size_t remaining = static_cast<std::size_t>(w);
            while (remaining > 0 && idx < iovecs.size())
            {
                if (remaining >= iovecs[idx].iov_len)
                {
                    remaining -= iovecs[idx].iov_len;
                    ++idx;
                }
                else
                {
                    auto *base = static_cast<std::uint8_t *>(iovecs[idx].iov_base);
                    iovecs[idx].iov_base = base + remaining;
                    iovecs[idx].iov_len -= remaining;
                    remaining = 0;
                }
            }
        }
        return pomai::Status::Ok();
    }

    pomai::Status Wal::AppendPut(pomai::VectorId id, pomai::VectorView vec)
    {
        return AppendPut(id, vec, pomai::Metadata());
    }

    pomai::Status Wal::AppendPut(pomai::VectorId id, pomai::VectorView vec, const pomai::Metadata& meta)
    {
        // If metadata is empty, use standard kPut for compatibility and compactness
        if (meta.tenant.empty()) {
            RecordPrefix rp{};
            rp.seq = ++seq_;
            rp.op = static_cast<std::uint8_t>(Op::kPut);
            rp.id = id;
            rp.dim = vec.dim;

            const std::size_t payload_bytes = vec.size_bytes();

            FrameHeader fh{};
            fh.len = static_cast<std::uint32_t>(sizeof(RecordPrefix) + payload_bytes + sizeof(std::uint32_t));

            std::uint32_t crc = pomai::util::Crc32c(&rp, sizeof(rp));
            crc = pomai::util::Crc32c(vec.data, payload_bytes, crc);

            const std::size_t total_bytes = sizeof(FrameHeader) + fh.len;
            auto st = RotateIfNeeded(total_bytes);
            if (!st.ok()) return st;

            std::vector<iovec> iovecs;
            iovecs.reserve(4);
            iovecs.push_back({&fh, sizeof(fh)});
            iovecs.push_back({&rp, sizeof(rp)});
            iovecs.push_back({const_cast<float *>(vec.data), payload_bytes});
            iovecs.push_back({&crc, sizeof(crc)});

            st = PWritevAll(impl_->file.fd(), file_off_, std::move(iovecs));
            if (!st.ok()) return st;

            file_off_ += total_bytes;
            bytes_in_seg_ += total_bytes;

            if (fsync_ == pomai::FsyncPolicy::kAlways)
                return impl_->file.SyncData();
            return pomai::Status::Ok();
        } 
        else 
        {
            // Use kPutMeta
            RecordPrefix rp{};
            rp.seq = ++seq_;
            rp.op = static_cast<std::uint8_t>(Op::kPutMeta);
            rp.id = id;
            rp.dim = vec.dim;

            const std::size_t vec_bytes = vec.size_bytes();
            const std::size_t meta_len = meta.tenant.size();
            const std::size_t meta_bytes = sizeof(std::uint32_t) + meta_len;
            const std::size_t payload_bytes = vec_bytes + meta_bytes;

            FrameHeader fh{};
            fh.len = static_cast<std::uint32_t>(sizeof(RecordPrefix) + payload_bytes + sizeof(std::uint32_t));
            
            std::uint32_t crc = pomai::util::Crc32c(&rp, sizeof(rp));
            crc = pomai::util::Crc32c(vec.data, vec_bytes, crc);
            std::uint32_t len32 = static_cast<std::uint32_t>(meta_len);
            crc = pomai::util::Crc32c(&len32, sizeof(len32), crc);
            if (meta_len > 0) {
                crc = pomai::util::Crc32c(meta.tenant.data(), meta_len, crc);
            }

            const std::size_t total_bytes = sizeof(FrameHeader) + fh.len;
            auto st = RotateIfNeeded(total_bytes);
            if (!st.ok()) return st;

            std::vector<iovec> iovecs;
            iovecs.reserve(6);
            iovecs.push_back({&fh, sizeof(fh)});
            iovecs.push_back({&rp, sizeof(rp)});
            iovecs.push_back({const_cast<float *>(vec.data), vec_bytes});
            iovecs.push_back({&len32, sizeof(len32)});
            if (meta_len > 0) {
                iovecs.push_back({const_cast<char *>(meta.tenant.data()), meta_len});
            }
            iovecs.push_back({&crc, sizeof(crc)});

            st = PWritevAll(impl_->file.fd(), file_off_, std::move(iovecs));
            if (!st.ok()) return st;

            file_off_ += total_bytes;
            bytes_in_seg_ += total_bytes;
            
            if (fsync_ == pomai::FsyncPolicy::kAlways)
                return impl_->file.SyncData();
            return pomai::Status::Ok();
        }
    }

    pomai::Status Wal::AppendDelete(pomai::VectorId id)
    {
        RecordPrefix rp{};
        rp.seq = ++seq_;
        rp.op = static_cast<std::uint8_t>(Op::kDel);
        rp.id = id;
        rp.dim = 0;

        auto &frame = scratch_;
        frame.clear();

        FrameHeader fh{};
        fh.len = static_cast<std::uint32_t>(sizeof(RecordPrefix) + sizeof(std::uint32_t));
        AppendBytes(&frame, &fh, sizeof(fh));
        AppendBytes(&frame, &rp, sizeof(rp));

        const std::uint32_t crc = pomai::util::Crc32c(frame.data() + sizeof(FrameHeader), fh.len - sizeof(std::uint32_t));
        AppendBytes(&frame, &crc, sizeof(crc));

        auto st = RotateIfNeeded(frame.size());
        if (!st.ok())
            return st;

        st = impl_->file.PWrite(file_off_, frame.data(), frame.size());
        if (!st.ok())
            return st;

        file_off_ += frame.size();
        bytes_in_seg_ += frame.size();

        if (fsync_ == pomai::FsyncPolicy::kAlways)
            return impl_->file.SyncData();
        return pomai::Status::Ok();
    }
// ... (BatchAppend skipped for brevity)





    pomai::Status Wal::AppendBatch(const std::vector<pomai::VectorId>& ids,
                                    const std::vector<pomai::VectorView>& vectors)
    {
        // Validation
        if (ids.size() != vectors.size())
            return pomai::Status::InvalidArgument("ids and vectors size mismatch");
        if (ids.empty())
            return pomai::Status::Ok();  // No-op for empty batch
        
        std::size_t total_bytes = 0;
        for (const auto& vec : vectors) {
            total_bytes += sizeof(FrameHeader) + sizeof(RecordPrefix) + 
                          vec.size_bytes() + sizeof(std::uint32_t);
        }
        
        // Rotate if needed
        auto st = RotateIfNeeded(total_bytes);
        if (!st.ok())
            return st;
        
        std::uint64_t off = file_off_;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            RecordPrefix rp{};
            rp.seq = ++seq_;
            rp.op = static_cast<std::uint8_t>(Op::kPut);
            rp.id = ids[i];
            rp.dim = vectors[i].dim;
            
            const std::size_t payload_bytes = vectors[i].size_bytes();
            
            FrameHeader fh{};
            fh.len = static_cast<std::uint32_t>(sizeof(RecordPrefix) + payload_bytes + sizeof(std::uint32_t));
            
            std::uint32_t crc = pomai::util::Crc32c(&rp, sizeof(rp));
            crc = pomai::util::Crc32c(vectors[i].data, payload_bytes, crc);
            
            std::vector<iovec> iovecs;
            iovecs.reserve(4);
            iovecs.push_back({&fh, sizeof(fh)});
            iovecs.push_back({&rp, sizeof(rp)});
            iovecs.push_back({const_cast<float *>(vectors[i].data), payload_bytes});
            iovecs.push_back({&crc, sizeof(crc)});
            
            st = PWritevAll(impl_->file.fd(), off, std::move(iovecs));
            if (!st.ok())
                return st;
            
            off += sizeof(FrameHeader) + fh.len;
        }
        
        file_off_ += total_bytes;
        bytes_in_seg_ += total_bytes;
        
        // Single fsync for entire batch (KEY OPTIMIZATION)
        if (fsync_ == pomai::FsyncPolicy::kAlways)
            return impl_->file.SyncData();
        return pomai::Status::Ok();
    }

    pomai::Status Wal::Flush()
    {
        if (!impl_)
            return pomai::Status::Ok();
        if (fsync_ == pomai::FsyncPolicy::kNever)
            return pomai::Status::Ok();
        return impl_->file.SyncData(); // Flush() == durability boundary in base
    }

    // Gate #2 requirement: tolerate truncated tail.
    // Replay stops cleanly if it cannot read a full frame header or full body.
    pomai::Status Wal::ReplayInto(pomai::table::MemTable &mem)
    {
        for (std::uint64_t g = 0; fs::exists(SegmentPath(g)); ++g)
        {
            pomai::util::PosixFile f;
            auto st = pomai::util::PosixFile::OpenRead(SegmentPath(g), &f);
            if (!st.ok())
                return st;

            std::error_code ec;
            const std::uint64_t file_size = static_cast<std::uint64_t>(fs::file_size(SegmentPath(g), ec));
            if (ec)
                return pomai::Status::IoError("wal file_size failed");

            std::uint64_t off = 0;
            if (file_size >= sizeof(WalFileHeader))
            {
                WalFileHeader hdr{};
                std::size_t got = 0;
                st = f.ReadAt(0, &hdr, sizeof(hdr), &got);
                if (!st.ok())
                    return st;
                if (got != sizeof(hdr))
                    return pomai::Status::Corruption("wal short file header");

                if (std::memcmp(hdr.magic, kWalMagic, sizeof(hdr.magic)) == 0)
                {
                    if (hdr.version != kWalVersion)
                        return pomai::Status::Corruption("wal version mismatch");
                    off = sizeof(WalFileHeader);
                }
            }
            while (off + sizeof(FrameHeader) <= file_size)
            {
                FrameHeader fh{};
                std::size_t got = 0;
                st = f.ReadAt(off, &fh, sizeof(fh), &got);
                if (!st.ok())
                    return st;
                if (got != sizeof(fh))
                    break; // truncated tail

                const std::uint64_t body_off = off + sizeof(FrameHeader);
                const std::uint64_t body_end = body_off + fh.len;
                if (body_end > file_size)
                    break; // truncated tail

                std::vector<std::uint8_t> body(fh.len);
                st = f.ReadAt(body_off, body.data(), body.size(), &got);
                if (!st.ok())
                    return st;
                if (got != body.size())
                    break; // truncated tail

                if (fh.len < sizeof(RecordPrefix) + sizeof(std::uint32_t))
                {
                    return pomai::Status::Corruption("wal frame too small");
                }

                std::uint32_t stored_crc = 0;
                std::memcpy(&stored_crc, body.data() + (fh.len - sizeof(std::uint32_t)), sizeof(stored_crc));
                const std::uint32_t calc_crc = pomai::util::Crc32c(body.data(), fh.len - sizeof(std::uint32_t));
                if (stored_crc != calc_crc)
                {
                    // corruption inside full frame: this is real corruption, not tail truncation
                    return pomai::Status::Corruption("wal crc mismatch");
                }

                const auto *rp = reinterpret_cast<const RecordPrefix *>(body.data());
                if (rp->op == static_cast<std::uint8_t>(Op::kPut))
                {
                    const std::uint32_t dim = rp->dim;
                    const std::size_t vec_bytes = static_cast<std::size_t>(dim) * sizeof(float);
                    const std::size_t expect = sizeof(RecordPrefix) + vec_bytes + sizeof(std::uint32_t);
                    if (expect != fh.len)
                        return pomai::Status::Corruption("wal put length mismatch");

                    const float* vec_ptr = reinterpret_cast<const float*>(body.data() + sizeof(RecordPrefix));
                    st = mem.Put(rp->id, pomai::VectorView{vec_ptr, dim});
                    if (!st.ok())
                        return st;
                }
                else if (rp->op == static_cast<std::uint8_t>(Op::kPutMeta))
                {
                    const std::uint32_t dim = rp->dim;
                    const std::size_t vec_bytes = static_cast<std::size_t>(dim) * sizeof(float);
                    // Minimal check: headers + vec + meta_len(4) + crc(4)
                    if (fh.len < sizeof(RecordPrefix) + vec_bytes + 4 + 4)
                        return pomai::Status::Corruption("wal putmeta too short");

                    const float* vec_ptr = reinterpret_cast<const float*>(body.data() + sizeof(RecordPrefix));

                    // Decode metadata
                    const uint8_t* meta_ptr = body.data() + sizeof(RecordPrefix) + vec_bytes;
                    uint32_t meta_len = 0;
                    std::memcpy(&meta_len, meta_ptr, sizeof(meta_len));
                    
                    const std::size_t expect = sizeof(RecordPrefix) + vec_bytes + 4 + meta_len + sizeof(std::uint32_t);
                    if (expect != fh.len)
                         return pomai::Status::Corruption("wal putmeta length mismatch");
                         
                    std::string tenant(reinterpret_cast<const char*>(meta_ptr + 4), meta_len);
                    pomai::Metadata meta(std::move(tenant));
                    
                    st = mem.Put(rp->id, pomai::VectorView{vec_ptr, dim}, meta);
                    if (!st.ok()) return st;
                }
                else if (rp->op == static_cast<std::uint8_t>(Op::kDel))
                {
                    if (fh.len != sizeof(RecordPrefix) + sizeof(std::uint32_t))
                        return pomai::Status::Corruption("wal del length mismatch");
                    st = mem.Delete(rp->id);
                    if (!st.ok())
                        return st;
                }
                else
                {
                    return pomai::Status::Corruption("wal unknown op");
                }

                if (rp->seq > seq_)
                    seq_ = rp->seq;
                off = body_end;
            }

            (void)f.Close();
        }
        return pomai::Status::Ok();
    }
    pomai::Status Wal::Reset()
    {
        if (impl_) {
            impl_->file.Close();
            delete impl_;
            impl_ = nullptr;
        }

        // Delete all wal files
        for (std::uint64_t g = 0; ; ++g) {
            std::string p = SegmentPath(g);
            std::error_code ec;
            if (!fs::exists(p, ec)) break;
            fs::remove(p, ec);
        }
        
        // Reset state
        gen_ = 0;
        seq_ = 0; // Safe to reset seq if MemTable is empty/flushed.
        file_off_ = 0;
        bytes_in_seg_ = 0;

        // Re-open (creates new wal_0.log)
        impl_ = new Impl();
        impl_->path = SegmentPath(gen_);
        
        // Create directory just in case (Open does it? No, Open calls create_directories).
        // Let's call Open logic or just do minimal.
        // Replicating Open logic:
        // Open() assumes closed.
        // But here we set impl_ already?
        // Let's reuse Open() logic but Open() scans for gen_.
        // We deleted everything. So Open() will find no files, set gen_=0.
        // So:
        delete impl_; impl_ = nullptr; // Reset impl again
        return Open();
    }

} // namespace pomai::storage
