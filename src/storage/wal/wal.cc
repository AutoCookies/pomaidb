#include "storage/wal/wal.h"

#include <cerrno>
#include <cstring>
#include <vector>

#include "palloc_compat.h"
#include "core/security/aes_gcm.h"
#include "core/graph/graph_membrane_impl.h"
#include "table/memtable.h"
#include "util/crc32c.h"

namespace pomai::storage {

    enum class Op : std::uint8_t
    {
        kPut = 1,
        kDel = 2,
        kPutMeta = 3,
        kRawKV = 4,
        kBatchStart = 5,
        kBatchEnd = 6
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
        std::unique_ptr<pomai::WritableFile> file;
        std::string path;
    };

    Wal::Wal(pomai::Env* env,
             std::string db_path,
             std::uint32_t shard_id,
             std::size_t segment_bytes,
             pomai::FsyncPolicy fsync,
             bool encryption_enabled,
             std::string encryption_key_hex,
             palloc_heap_t* heap)
        : env_(env ? env : pomai::Env::Default()),
          db_path_(std::move(db_path)),
          shard_id_(shard_id),
          segment_bytes_(segment_bytes),
          fsync_(fsync),
          encryption_enabled_(encryption_enabled),
          heap_(heap) {
        if (encryption_enabled_) {
            auto st = pomai::core::AesGcm::ParseKeyHex(encryption_key_hex, &encryption_key_);
            if (!st.ok()) {
                encryption_enabled_ = false;
            }
        }
    }

    Wal::~Wal()
    {
        if (impl_)
        {
            if (impl_->file) (void)impl_->file->Close();
            impl_->~Impl();
            palloc_free(impl_);
            impl_ = nullptr;
        }
    }

    std::string Wal::SegmentPath(std::uint64_t gen) const
    {
        return db_path_ + "/wal_" + std::to_string(shard_id_) + "_" + std::to_string(gen) + ".log";
    }

    pomai::Status Wal::Open()
    {
        pomai::Status st = env_->CreateDirIfMissing(db_path_);
        if (!st.ok()) return st;

        gen_ = 0;
        while (env_->FileExists(SegmentPath(gen_)).ok())
            ++gen_;
        for (std::uint64_t g = 0; g < gen_; ++g) {
            std::unique_ptr<pomai::RandomAccessFile> raf;
            st = env_->NewRandomAccessFile(SegmentPath(g), &raf);
            if (!st.ok() || !raf) return st.ok() ? pomai::Status::IOError("wal header open failed") : st;
            pomai::Slice hs;
            WalFileHeader hdr{};
            st = raf->Read(0, sizeof(hdr), &hs);
            if (!st.ok()) return st;
            if (hs.size() != sizeof(hdr)) return pomai::Status::Aborted("wal header short");
            std::memcpy(&hdr, hs.data(), sizeof(hdr));
            if (std::memcmp(hdr.magic, kWalMagic, sizeof(hdr.magic)) != 0 || hdr.version != kWalVersion) {
                return pomai::Status::Aborted("wal compatibility check failed");
            }
        }

        void* raw = heap_
            ? palloc_heap_malloc_aligned(heap_, sizeof(Impl), alignof(Impl))
            : palloc_malloc_aligned(sizeof(Impl), alignof(Impl));
        if (!raw) return pomai::Status::IOError("WAL Impl allocation failed");
        impl_ = new (raw) Impl();
        impl_->path = SegmentPath(gen_);

        st = env_->NewAppendableFile(impl_->path, &impl_->file);
        if (!st.ok() || !impl_->file)
        {
            impl_->~Impl();
            palloc_free(impl_);
            impl_ = nullptr;
            return st.ok() ? pomai::Status::IOError("NewAppendableFile returned null") : st;
        }

        std::uint64_t sz = 0;
        st = env_->GetFileSize(impl_->path, &sz);
        if (!st.ok()) sz = 0;

        if (sz == 0)
        {
            WalFileHeader hdr{};
            std::memcpy(hdr.magic, kWalMagic, sizeof(hdr.magic));
            hdr.version = kWalVersion;
            st = impl_->file->Append(pomai::Slice(&hdr, sizeof(hdr)));
            if (!st.ok())
            {
                (void)impl_->file->Close();
                impl_->~Impl();
                palloc_free(impl_);
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
            pomai::Status st_sync = impl_->file ? impl_->file->Sync() : pomai::Status::Ok();
            pomai::Status st_close = impl_->file ? impl_->file->Close() : pomai::Status::Ok();
            if (!st_sync.ok()) {
                impl_->~Impl();
                palloc_free(impl_);
                impl_ = nullptr;
                return st_sync;
            }
            if (!st_close.ok()) {
                impl_->~Impl();
                palloc_free(impl_);
                impl_ = nullptr;
                return st_close;
            }
            impl_->~Impl();
            palloc_free(impl_);
            impl_ = nullptr;
        }

        ++gen_;
        file_off_ = 0;
        bytes_in_seg_ = 0;

        void* raw = heap_
            ? palloc_heap_malloc_aligned(heap_, sizeof(Impl), alignof(Impl))
            : palloc_malloc_aligned(sizeof(Impl), alignof(Impl));
        if (!raw) return pomai::Status::IOError("WAL Impl allocation failed");
        impl_ = new (raw) Impl();
        impl_->path = SegmentPath(gen_);
        pomai::Status st = env_->NewAppendableFile(impl_->path, &impl_->file);
        if (!st.ok() || !impl_->file)
        {
            impl_->~Impl();
            palloc_free(impl_);
            impl_ = nullptr;
            return st.ok() ? pomai::Status::IOError("NewAppendableFile returned null") : st;
        }
        WalFileHeader hdr{};
        std::memcpy(hdr.magic, kWalMagic, sizeof(hdr.magic));
        hdr.version = kWalVersion;
        st = impl_->file->Append(pomai::Slice(&hdr, sizeof(hdr)));
        if (!st.ok()) return st;
        file_off_ = sizeof(WalFileHeader);
        bytes_in_seg_ = sizeof(WalFileHeader);
        return pomai::Status::Ok();
    }

    struct Iov { void* base; std::size_t len; };

    static void AppendBytes(std::vector<std::uint8_t> *dst, const void *p, std::size_t n)
    {
        const auto *b = static_cast<const std::uint8_t *>(p);
        dst->insert(dst->end(), b, b + n);
    }

    static void AppendString(std::vector<std::uint8_t>* dst, const std::string& s) {
        const std::uint32_t len = static_cast<std::uint32_t>(s.size());
        AppendBytes(dst, &len, sizeof(len));
        if (len > 0) AppendBytes(dst, s.data(), len);
    }

    static bool ReadString(const std::uint8_t* base, std::size_t len, std::size_t* cursor, std::string* out) {
        if (*cursor + sizeof(std::uint32_t) > len) return false;
        std::uint32_t n = 0;
        std::memcpy(&n, base + *cursor, sizeof(n));
        *cursor += sizeof(n);
        if (*cursor + n > len) return false;
        out->assign(reinterpret_cast<const char*>(base + *cursor), n);
        *cursor += n;
        return true;
    }

    static std::array<std::uint8_t, 12> MakeNonce(std::uint32_t shard_id, std::uint64_t seq, std::uint64_t gen)
    {
        std::array<std::uint8_t, 12> nonce{};
        std::memcpy(nonce.data(), &shard_id, sizeof(shard_id));
        std::memcpy(nonce.data() + 4, &seq, sizeof(seq));
        nonce[11] ^= static_cast<std::uint8_t>(gen & 0xFFu);
        return nonce;
    }

    static pomai::Status AppendIovecs(pomai::WritableFile* file, std::vector<Iov>& iovecs)
    {
        for (const auto& iov : iovecs)
        {
            if (iov.len == 0) continue;
            pomai::Slice s(iov.base, iov.len);
            pomai::Status st = file->Append(s);
            if (!st.ok()) return st;
        }
        return pomai::Status::Ok();
    }

    pomai::Status Wal::AppendPut(pomai::VectorId id, pomai::VectorView vec)
    {
        return AppendPut(id, vec, pomai::Metadata());
    }

    pomai::Status Wal::AppendPut(pomai::VectorId id, pomai::VectorView vec, const pomai::Metadata& meta)
    {
        RecordPrefix rp{};
        rp.seq = ++seq_;
        rp.op = static_cast<std::uint8_t>(meta.tenant.empty() ? Op::kPut : Op::kPutMeta);
        rp.id = id;
        rp.dim = vec.dim;

        std::vector<std::uint8_t> plain;
        AppendBytes(&plain, &rp, sizeof(rp));
        AppendBytes(&plain, vec.data, vec.size_bytes());
        if (!meta.tenant.empty() || !meta.device_id.empty() || !meta.location_id.empty() || meta.timestamp > 0 || meta.src_vid > 0) {
            AppendBytes(&plain, &meta.src_vid, sizeof(meta.src_vid));
            AppendBytes(&plain, &meta.timestamp, sizeof(meta.timestamp));
            AppendBytes(&plain, &meta.lsn, sizeof(meta.lsn));
            AppendBytes(&plain, &meta.lat, sizeof(meta.lat));
            AppendBytes(&plain, &meta.lon, sizeof(meta.lon));
            AppendString(&plain, meta.tenant);
            AppendString(&plain, meta.device_id);
            AppendString(&plain, meta.location_id);
            AppendString(&plain, meta.text);
            AppendString(&plain, meta.payload);
        }

        std::vector<std::uint8_t> body;
        if (encryption_enabled_) {
            const auto nonce = MakeNonce(shard_id_, rp.seq, gen_);
            std::vector<std::uint8_t> cipher;
            std::array<std::uint8_t, 16> tag{};
            auto est = pomai::core::AesGcm::Encrypt(encryption_key_, nonce, plain, &cipher, &tag);
            if (!est.ok()) return est;
            AppendBytes(&body, nonce.data(), nonce.size());
            AppendBytes(&body, cipher.data(), cipher.size());
            AppendBytes(&body, tag.data(), tag.size());
        } else {
            body = std::move(plain);
        }
        const std::uint32_t crc = pomai::util::Crc32c(body.data(), body.size());
        FrameHeader fh{};
        fh.len = static_cast<std::uint32_t>(body.size() + sizeof(std::uint32_t));

        const std::size_t total_bytes = sizeof(FrameHeader) + fh.len;
        auto st = RotateIfNeeded(total_bytes);
        if (!st.ok()) return st;

        std::vector<Iov> iovecs;
        iovecs.reserve(3);
        iovecs.push_back({&fh, sizeof(fh)});
        iovecs.push_back({body.data(), body.size()});
        iovecs.push_back({const_cast<std::uint32_t*>(&crc), sizeof(crc)});
        st = AppendIovecs(impl_->file.get(), iovecs);
        if (!st.ok()) return st;
        file_off_ += total_bytes;
        bytes_in_seg_ += total_bytes;
        bytes_written_total_ += static_cast<std::uint64_t>(total_bytes);
        if (fsync_ == pomai::FsyncPolicy::kAlways) return impl_->file->Sync();
        return pomai::Status::Ok();
    }

    pomai::Status Wal::AppendDelete(pomai::VectorId id)
    {
        RecordPrefix rp{};
        rp.seq = ++seq_;
        rp.op = static_cast<std::uint8_t>(Op::kDel);
        rp.id = id;
        rp.dim = 0;

        std::vector<std::uint8_t> plain;
        AppendBytes(&plain, &rp, sizeof(rp));
        std::vector<std::uint8_t> body;
        if (encryption_enabled_) {
            const auto nonce = MakeNonce(shard_id_, rp.seq, gen_);
            std::vector<std::uint8_t> cipher;
            std::array<std::uint8_t, 16> tag{};
            auto est = pomai::core::AesGcm::Encrypt(encryption_key_, nonce, plain, &cipher, &tag);
            if (!est.ok()) return est;
            AppendBytes(&body, nonce.data(), nonce.size());
            AppendBytes(&body, cipher.data(), cipher.size());
            AppendBytes(&body, tag.data(), tag.size());
        } else {
            body = std::move(plain);
        }
        FrameHeader fh{};
        fh.len = static_cast<std::uint32_t>(body.size() + sizeof(std::uint32_t));
        const std::uint32_t crc = pomai::util::Crc32c(body.data(), body.size());

        auto st = RotateIfNeeded(sizeof(FrameHeader) + fh.len);
        if (!st.ok())
            return st;
        std::vector<Iov> iovecs;
        iovecs.push_back({&fh, sizeof(fh)});
        iovecs.push_back({body.data(), body.size()});
        iovecs.push_back({const_cast<std::uint32_t*>(&crc), sizeof(crc)});
        st = AppendIovecs(impl_->file.get(), iovecs);
        if (!st.ok()) return st;

        file_off_ += sizeof(FrameHeader) + fh.len;
        bytes_in_seg_ += sizeof(FrameHeader) + fh.len;
        bytes_written_total_ += static_cast<std::uint64_t>(sizeof(FrameHeader) + fh.len);

        if (fsync_ == pomai::FsyncPolicy::kAlways)
            return impl_->file->Sync();
        return pomai::Status::Ok();
    }
    pomai::Status Wal::AppendRawKV(std::uint8_t op, pomai::Slice key, pomai::Slice value)
    {
        RecordPrefix rp{};
        rp.seq = ++seq_;
        rp.op = op;
        rp.id = 0;
        rp.dim = 0;

        std::uint32_t klen = static_cast<std::uint32_t>(key.size());
        std::uint32_t vlen = static_cast<std::uint32_t>(value.size());
        std::vector<std::uint8_t> plain;
        AppendBytes(&plain, &rp, sizeof(rp));
        AppendBytes(&plain, &klen, sizeof(klen));
        AppendBytes(&plain, &vlen, sizeof(vlen));
        if (klen > 0) AppendBytes(&plain, key.data(), klen);
        if (vlen > 0) AppendBytes(&plain, value.data(), vlen);
        std::vector<std::uint8_t> body;
        if (encryption_enabled_) {
            const auto nonce = MakeNonce(shard_id_, rp.seq, gen_);
            std::vector<std::uint8_t> cipher;
            std::array<std::uint8_t, 16> tag{};
            auto est = pomai::core::AesGcm::Encrypt(encryption_key_, nonce, plain, &cipher, &tag);
            if (!est.ok()) return est;
            AppendBytes(&body, nonce.data(), nonce.size());
            AppendBytes(&body, cipher.data(), cipher.size());
            AppendBytes(&body, tag.data(), tag.size());
        } else {
            body = std::move(plain);
        }
        std::uint32_t crc = pomai::util::Crc32c(body.data(), body.size());
        FrameHeader fh{};
        fh.len = static_cast<std::uint32_t>(body.size() + sizeof(std::uint32_t));
        const std::size_t total_bytes = sizeof(FrameHeader) + fh.len;
        auto st = RotateIfNeeded(total_bytes);
        if (!st.ok()) return st;
        std::vector<Iov> iovecs;
        iovecs.reserve(3);
        iovecs.push_back({&fh, sizeof(fh)});
        iovecs.push_back({body.data(), body.size()});
        iovecs.push_back(Iov{&crc, sizeof(crc)});

        st = AppendIovecs(impl_->file.get(), iovecs);
        if (!st.ok()) return st;

        file_off_ += total_bytes;
        bytes_in_seg_ += total_bytes;
        bytes_written_total_ += static_cast<std::uint64_t>(total_bytes);

        if (fsync_ == pomai::FsyncPolicy::kAlways)
            return impl_->file->Sync();
        return pomai::Status::Ok();
    }


    pomai::Status Wal::AppendBatch(const std::vector<pomai::VectorId>& ids,
                                    const std::vector<pomai::VectorView>& vectors)
    {
        // Validation
        if (ids.size() != vectors.size())
            return pomai::Status::InvalidArgument("ids and vectors size mismatch");
        if (ids.empty())
            return pomai::Status::Ok();  // No-op for empty batch
        
        if (encryption_enabled_) {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                auto st = AppendPut(ids[i], vectors[i]);
                if (!st.ok()) return st;
            }
            return pomai::Status::Ok();
        }

        std::size_t total_batch_bytes = 0;
        for (const auto& vec : vectors) {
            total_batch_bytes += sizeof(FrameHeader) + sizeof(RecordPrefix) +
                                vec.size_bytes() + sizeof(std::uint32_t);
        }
        auto st = RotateIfNeeded(total_batch_bytes);
        if (!st.ok()) return st;

        struct TmpRecord { FrameHeader fh; RecordPrefix rp; std::uint32_t crc; };
        std::vector<TmpRecord> tmps(ids.size());
        std::vector<Iov> iovecs;
        iovecs.reserve(ids.size() * 4);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            tmps[i].rp.seq = ++seq_;
            tmps[i].rp.op = static_cast<std::uint8_t>(Op::kPut);
            tmps[i].rp.id = ids[i];
            tmps[i].rp.dim = vectors[i].dim;
            const std::size_t payload_bytes = vectors[i].size_bytes();
            tmps[i].fh.len = static_cast<std::uint32_t>(sizeof(RecordPrefix) + payload_bytes + sizeof(std::uint32_t));
            tmps[i].crc = pomai::util::Crc32c(&tmps[i].rp, sizeof(tmps[i].rp));
            tmps[i].crc = pomai::util::Crc32c(vectors[i].data, payload_bytes, tmps[i].crc);
            iovecs.push_back({&tmps[i].fh, sizeof(tmps[i].fh)});
            iovecs.push_back({&tmps[i].rp, sizeof(tmps[i].rp)});
            iovecs.push_back({const_cast<float *>(vectors[i].data), payload_bytes});
            iovecs.push_back({&tmps[i].crc, sizeof(tmps[i].crc)});
        }
        st = AppendIovecs(impl_->file.get(), iovecs);
        if (!st.ok()) return st;
        file_off_ += total_batch_bytes;
        bytes_in_seg_ += total_batch_bytes;
        bytes_written_total_ += static_cast<std::uint64_t>(total_batch_bytes);
        if (fsync_ == pomai::FsyncPolicy::kAlways) return impl_->file->Sync();
        return pomai::Status::Ok();
    }

    pomai::Status Wal::Flush()
    {
        if (!impl_ || !impl_->file)
            return pomai::Status::Ok();
        if (fsync_ == pomai::FsyncPolicy::kNever)
            return pomai::Status::Ok();
        return impl_->file->Sync();
    }

    // Gate #2 requirement: tolerate truncated tail.
    // Replay stops cleanly if it cannot read a full frame header or full body.
    pomai::Status Wal::ReplayInto(pomai::table::MemTable &mem)
    {
        for (std::uint64_t g = 0; env_->FileExists(SegmentPath(g)).ok(); ++g)
        {
            std::unique_ptr<pomai::RandomAccessFile> raf;
            pomai::Status st = env_->NewRandomAccessFile(SegmentPath(g), &raf);
            if (!st.ok() || !raf)
                return st.ok() ? pomai::Status::IOError("NewRandomAccessFile returned null") : st;

            std::uint64_t file_size = 0;
            st = env_->GetFileSize(SegmentPath(g), &file_size);
            if (!st.ok())
                return pomai::Status::IoError("wal GetFileSize failed");

            std::uint64_t off = 0;
            if (file_size >= sizeof(WalFileHeader))
            {
                WalFileHeader hdr{};
                pomai::Slice hdr_slice;
                st = raf->Read(0, sizeof(hdr), &hdr_slice);
                if (!st.ok())
                    return st;
                if (hdr_slice.size() != sizeof(hdr))
                    return pomai::Status::Corruption("wal short file header");
                std::memcpy(&hdr, hdr_slice.data(), sizeof(hdr));

                if (std::memcmp(hdr.magic, kWalMagic, sizeof(hdr.magic)) == 0)
                {
                    if (hdr.version != kWalVersion)
                        return pomai::Status::Aborted("wal version mismatch");
                    off = sizeof(WalFileHeader);
                }
            }
            while (off + sizeof(FrameHeader) <= file_size)
            {
                FrameHeader fh{};
                pomai::Slice fh_slice;
                st = raf->Read(off, sizeof(fh), &fh_slice);
                if (!st.ok())
                    return st;
                if (fh_slice.size() != sizeof(fh))
                    break; // truncated tail
                std::memcpy(&fh, fh_slice.data(), sizeof(fh));

                const std::uint64_t body_off = off + sizeof(FrameHeader);
                const std::uint64_t body_end = body_off + fh.len;
                if (body_end > file_size)
                    break; // truncated tail

                std::vector<std::uint8_t> body(fh.len);
                pomai::Slice body_slice;
                st = raf->Read(body_off, fh.len, &body_slice);
                if (!st.ok())
                    return st;
                if (body_slice.size() != fh.len)
                    break; // truncated tail
                std::memcpy(body.data(), body_slice.data(), body_slice.size());

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

                std::vector<std::uint8_t> plain;
                const std::uint8_t* payload_ptr = body.data();
                std::size_t payload_len = fh.len - sizeof(std::uint32_t);
                if (encryption_enabled_) {
                    if (payload_len < 12 + 16) return pomai::Status::Corruption("wal encrypted frame too small");
                    std::array<std::uint8_t, 12> nonce{};
                    std::array<std::uint8_t, 16> tag{};
                    std::memcpy(nonce.data(), payload_ptr, nonce.size());
                    const std::size_t cipher_len = payload_len - nonce.size() - tag.size();
                    std::vector<std::uint8_t> cipher(cipher_len);
                    std::memcpy(cipher.data(), payload_ptr + nonce.size(), cipher_len);
                    std::memcpy(tag.data(), payload_ptr + nonce.size() + cipher_len, tag.size());
                    auto dst = pomai::core::AesGcm::Decrypt(encryption_key_, nonce, cipher, tag, &plain);
                    if (!dst.ok()) return dst;
                    payload_ptr = plain.data();
                    payload_len = plain.size();
                }

                if (payload_len < sizeof(RecordPrefix)) return pomai::Status::Corruption("wal missing record prefix");
                const auto *rp = reinterpret_cast<const RecordPrefix *>(payload_ptr);
                if (rp->op == static_cast<std::uint8_t>(Op::kPut))
                {
                    const std::uint32_t dim = rp->dim;
                    const std::size_t vec_bytes = static_cast<std::size_t>(dim) * sizeof(float);
                    const std::size_t expect = sizeof(RecordPrefix) + vec_bytes + sizeof(std::uint32_t);
                    if (!encryption_enabled_ && expect != fh.len)
                        return pomai::Status::Corruption("wal put length mismatch");
                    if (encryption_enabled_ && (sizeof(RecordPrefix) + vec_bytes) != payload_len)
                        return pomai::Status::Corruption("wal put length mismatch");

                    const float* vec_ptr = reinterpret_cast<const float*>(payload_ptr + sizeof(RecordPrefix));
                    pomai::Metadata replay_meta;
                    replay_meta.lsn = rp->seq;
                    st = mem.Put(rp->id, pomai::VectorView{vec_ptr, dim}, replay_meta);
                    if (!st.ok())
                        return st;
                }
                else if (rp->op == static_cast<std::uint8_t>(Op::kPutMeta))
                {
                    const std::uint32_t dim = rp->dim;
                    const std::size_t vec_bytes = static_cast<std::size_t>(dim) * sizeof(float);
                    // Minimal check: headers + vec + meta_len(4) + crc(4)
                    if ((!encryption_enabled_ && fh.len < sizeof(RecordPrefix) + vec_bytes + 4 + 4) ||
                        (encryption_enabled_ && payload_len < sizeof(RecordPrefix) + vec_bytes + 4))
                        return pomai::Status::Corruption("wal putmeta too short");

                    const float* vec_ptr = reinterpret_cast<const float*>(payload_ptr + sizeof(RecordPrefix));

                    const std::uint8_t* meta_ptr = payload_ptr + sizeof(RecordPrefix) + vec_bytes;
                    const std::size_t meta_len = payload_len - (sizeof(RecordPrefix) + vec_bytes);
                    pomai::Metadata meta;
                    std::size_t cursor = 0;
                    if (meta_len >= (8 + 8 + 8 + 8 + 8)) {
                        std::memcpy(&meta.src_vid, meta_ptr + cursor, 8); cursor += 8;
                        std::memcpy(&meta.timestamp, meta_ptr + cursor, 8); cursor += 8;
                        std::memcpy(&meta.lsn, meta_ptr + cursor, 8); cursor += 8;
                        std::memcpy(&meta.lat, meta_ptr + cursor, 8); cursor += 8;
                        std::memcpy(&meta.lon, meta_ptr + cursor, 8); cursor += 8;
                        if (!ReadString(meta_ptr, meta_len, &cursor, &meta.tenant) ||
                            !ReadString(meta_ptr, meta_len, &cursor, &meta.device_id) ||
                            !ReadString(meta_ptr, meta_len, &cursor, &meta.location_id) ||
                            !ReadString(meta_ptr, meta_len, &cursor, &meta.text) ||
                            !ReadString(meta_ptr, meta_len, &cursor, &meta.payload)) {
                            return pomai::Status::Corruption("wal putmeta decode failed");
                        }
                    } else {
                        std::size_t c2 = 0;
                        if (!ReadString(meta_ptr, meta_len, &c2, &meta.tenant)) {
                            return pomai::Status::Corruption("wal legacy putmeta decode failed");
                        }
                    }
                    meta.lsn = rp->seq;
                    
                    st = mem.Put(rp->id, pomai::VectorView{vec_ptr, dim}, meta);
                    if (!st.ok()) return st;
                }
                else if (rp->op == static_cast<std::uint8_t>(Op::kDel))
                {
                    if ((!encryption_enabled_ && fh.len != sizeof(RecordPrefix) + sizeof(std::uint32_t)) ||
                        (encryption_enabled_ && payload_len != sizeof(RecordPrefix)))
                        return pomai::Status::Corruption("wal del length mismatch");
                    st = mem.Delete(rp->id);
                    if (!st.ok())
                        return st;
                }
                else if (rp->op == static_cast<std::uint8_t>(Op::kRawKV) ||
                         rp->op == static_cast<std::uint8_t>(Op::kBatchStart) ||
                         rp->op == static_cast<std::uint8_t>(Op::kBatchEnd))
                {
                    // Non-vector records (graph raw KV, batch markers): skip gracefully.
                    off = body_end;
                    if (rp->seq > seq_) seq_ = rp->seq;
                    continue;
                }
                else
                {
                    return pomai::Status::Corruption("wal unknown op");
                }

                if (rp->seq > seq_)
                    seq_ = rp->seq;
                off = body_end;
            }
        }
        return pomai::Status::Ok();
    }

    pomai::Status Wal::ReplayGraphInto(pomai::core::GraphMembraneImpl* graph)
    {
        if (!graph) return pomai::Status::InvalidArgument("graph is null");
        for (std::uint64_t g = 0; env_->FileExists(SegmentPath(g)).ok(); ++g)
        {
            std::unique_ptr<pomai::RandomAccessFile> raf;
            pomai::Status st = env_->NewRandomAccessFile(SegmentPath(g), &raf);
            if (!st.ok() || !raf)
                return st.ok() ? pomai::Status::IOError("NewRandomAccessFile returned null") : st;

            std::uint64_t file_size = 0;
            st = env_->GetFileSize(SegmentPath(g), &file_size);
            if (!st.ok())
                return pomai::Status::IoError("wal GetFileSize failed");

            std::uint64_t off = 0;
            if (file_size >= sizeof(WalFileHeader))
            {
                WalFileHeader hdr{};
                pomai::Slice hdr_slice;
                st = raf->Read(0, sizeof(hdr), &hdr_slice);
                if (!st.ok()) return st;
                if (hdr_slice.size() != sizeof(hdr))
                    return pomai::Status::Corruption("wal short file header");
                std::memcpy(&hdr, hdr_slice.data(), sizeof(hdr));
                if (std::memcmp(hdr.magic, kWalMagic, sizeof(hdr.magic)) == 0)
                {
                    if (hdr.version != kWalVersion)
                        return pomai::Status::Aborted("wal version mismatch");
                    off = sizeof(WalFileHeader);
                }
            }

            while (off + sizeof(FrameHeader) <= file_size)
            {
                FrameHeader fh{};
                pomai::Slice fh_slice;
                st = raf->Read(off, sizeof(fh), &fh_slice);
                if (!st.ok()) return st;
                if (fh_slice.size() != sizeof(fh)) break;
                std::memcpy(&fh, fh_slice.data(), sizeof(fh));

                const std::uint64_t body_off = off + sizeof(FrameHeader);
                const std::uint64_t body_end = body_off + fh.len;
                if (body_end > file_size) break;

                std::vector<std::uint8_t> body(fh.len);
                pomai::Slice body_slice;
                st = raf->Read(body_off, fh.len, &body_slice);
                if (!st.ok()) return st;
                if (body_slice.size() != fh.len) break;
                std::memcpy(body.data(), body_slice.data(), body_slice.size());

                if (fh.len < sizeof(RecordPrefix) + sizeof(std::uint32_t))
                {
                    off = body_end;
                    continue;
                }

                std::uint32_t stored_crc = 0;
                std::memcpy(&stored_crc, body.data() + (fh.len - sizeof(std::uint32_t)), sizeof(stored_crc));
                const std::uint32_t calc_crc = pomai::util::Crc32c(body.data(), fh.len - sizeof(std::uint32_t));
                if (stored_crc != calc_crc)
                    return pomai::Status::Corruption("wal crc mismatch");

                std::vector<std::uint8_t> plain;
                const std::uint8_t* payload_ptr = body.data();
                std::size_t payload_len = fh.len - sizeof(std::uint32_t);
                if (encryption_enabled_) {
                    if (payload_len < 12 + 16) { off = body_end; continue; }
                    std::array<std::uint8_t, 12> nonce{};
                    std::array<std::uint8_t, 16> tag{};
                    std::memcpy(nonce.data(), payload_ptr, nonce.size());
                    const std::size_t cipher_len = payload_len - nonce.size() - tag.size();
                    std::vector<std::uint8_t> cipher(cipher_len);
                    std::memcpy(cipher.data(), payload_ptr + nonce.size(), cipher_len);
                    std::memcpy(tag.data(), payload_ptr + nonce.size() + cipher_len, tag.size());
                    auto dst = pomai::core::AesGcm::Decrypt(encryption_key_, nonce, cipher, tag, &plain);
                    if (!dst.ok()) return dst;
                    payload_ptr = plain.data();
                    payload_len = plain.size();
                }

                if (payload_len < sizeof(RecordPrefix)) { off = body_end; continue; }
                const auto* rp = reinterpret_cast<const RecordPrefix*>(payload_ptr);

                if (rp->op == static_cast<std::uint8_t>(Op::kRawKV))
                {
                    // Layout after RecordPrefix: klen(4) | vlen(4) | key[klen] | value[vlen]
                    const std::size_t min_size = sizeof(RecordPrefix) + 4 + 4;
                    if (payload_len >= min_size)
                    {
                        std::uint32_t klen = 0, vlen = 0;
                        std::memcpy(&klen, payload_ptr + sizeof(RecordPrefix), 4);
                        std::memcpy(&vlen, payload_ptr + sizeof(RecordPrefix) + 4, 4);
                        if (payload_len >= min_size + klen)
                        {
                            pomai::Slice key(reinterpret_cast<const char*>(payload_ptr + min_size), klen);
                            graph->ReplayEntry(key);
                        }
                    }
                }
                // All other ops (kPut, kPutMeta, kDel, kBatchStart, kBatchEnd) are not graph records; skip.

                if (rp->seq > seq_) seq_ = rp->seq;
                off = body_end;
            }
        }
        return pomai::Status::Ok();
    }

    pomai::Status Wal::Reset()
    {
        if (impl_) {
            if (impl_->file) (void)impl_->file->Close();
            impl_->~Impl();
            palloc_free(impl_);
            impl_ = nullptr;
        }

        for (std::uint64_t g = 0; ; ++g) {
            std::string p = SegmentPath(g);
            if (!env_->FileExists(p).ok()) break;
            (void)env_->DeleteFile(p);
        }

        gen_ = 0;
        seq_ = 0;
        file_off_ = 0;
        bytes_in_seg_ = 0;
        bytes_written_total_ = 0;
        return Open();
    }


    pomai::Status Wal::BeginBatch()
    {
        return AppendRawKV(static_cast<std::uint8_t>(Op::kBatchStart), {}, {});
    }

    pomai::Status Wal::EndBatch()
    {
        return AppendRawKV(static_cast<std::uint8_t>(Op::kBatchEnd), {}, {});
    }

} // namespace pomai::storage
