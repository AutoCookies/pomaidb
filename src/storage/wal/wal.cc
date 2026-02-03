#include "storage/wal/wal.h"

#include <filesystem>
#include <vector>

#include "table/memtable.h"
#include "util/crc32c.h"
#include "util/posix_file.h"

namespace fs = std::filesystem;

namespace pomai::storage
{

    enum class Op : std::uint8_t
    {
        kPut = 1,
        kDel = 2
    };

#pragma pack(push, 1)
    struct FrameHeader
    {
        std::uint32_t len; // bytes after this header: [type+payload+crc]
    };

    struct RecordPrefix
    {
        std::uint64_t seq;
        std::uint8_t op;
        std::uint64_t id;
        std::uint32_t dim; // PUT only, else 0
    };
#pragma pack(pop)

    class Wal::Impl
    {
    public:
        pomai::util::PosixFile file;
        std::string path;
        std::vector<std::uint8_t> scratch; // Reusable buffer for encoding frames
    };

    Wal::Wal(std::string db_path,
             std::uint32_t shard_id,
             std::size_t segment_bytes,
             pomai::FsyncPolicy fsync)
        : db_path_(std::move(db_path)),
          shard_id_(shard_id),
          segment_bytes_(segment_bytes),
          fsync_(fsync) {}

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
            return st;

        // Determine current size and append offset
        std::error_code ec;
        std::uint64_t sz = 0;
        if (fs::exists(impl_->path, ec))
            sz = static_cast<std::uint64_t>(fs::file_size(impl_->path, ec));
        file_off_ = sz;
        bytes_in_seg_ = static_cast<std::size_t>(sz);
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
            return st;
        return pomai::Status::Ok();
    }

    static void AppendBytes(std::vector<std::uint8_t> *dst, const void *p, std::size_t n)
    {
        const auto *b = static_cast<const std::uint8_t *>(p);
        dst->insert(dst->end(), b, b + n);
    }

    pomai::Status Wal::AppendPut(pomai::VectorId id, std::span<const float> vec)
    {
        RecordPrefix rp{};
        rp.seq = ++seq_;
        rp.op = static_cast<std::uint8_t>(Op::kPut);
        rp.id = id;
        rp.dim = static_cast<std::uint32_t>(vec.size());

        const std::size_t payload_bytes = vec.size_bytes();

        // reuse scratch buffer
        auto &frame = impl_->scratch;
        frame.clear();
        // heuristic reserve
        if (frame.capacity() < 128 + payload_bytes)
             frame.reserve(128 + payload_bytes);

        FrameHeader fh{};
        fh.len = static_cast<std::uint32_t>(sizeof(RecordPrefix) + payload_bytes + sizeof(std::uint32_t));
        AppendBytes(&frame, &fh, sizeof(fh));
        AppendBytes(&frame, &rp, sizeof(rp));
        AppendBytes(&frame, vec.data(), payload_bytes);

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

    pomai::Status Wal::AppendDelete(pomai::VectorId id)
    {
        RecordPrefix rp{};
        rp.seq = ++seq_;
        rp.op = static_cast<std::uint8_t>(Op::kDel);
        rp.id = id;
        rp.dim = 0;

        auto &frame = impl_->scratch;
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

                const std::uint32_t stored_crc = *reinterpret_cast<const std::uint32_t *>(
                    body.data() + (fh.len - sizeof(std::uint32_t)));
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

                    const float *vec = reinterpret_cast<const float *>(body.data() + sizeof(RecordPrefix));
                    st = mem.Put(rp->id, std::span<const float>{vec, dim});
                    if (!st.ok())
                        return st;
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
