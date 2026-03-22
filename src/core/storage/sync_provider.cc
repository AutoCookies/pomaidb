#include "core/storage/sync_provider.h"
#include "pomai/env.h"
#include "util/crc32c.h"
#include <cstring>
#include <vector>

namespace pomai::core {

#pragma pack(push, 1)
struct FrameHeader {
    uint32_t len;
};

struct WalFileHeader {
    char magic[12];
    uint32_t version;
};

struct RecordPrefix {
    uint64_t seq;
    uint8_t op;
    uint64_t id;
    uint32_t dim;
};
#pragma pack(pop)

constexpr char kWalMagic[] = "pomai.wal.v1";
constexpr uint32_t kWalVersion = 1;

WalStreamer::WalStreamer(const std::string& db_path, uint32_t shard_id)
    : db_path_(db_path), shard_id_(shard_id) {}

std::string WalStreamer::SegmentPath(uint64_t gen) const {
    return db_path_ + "/wal_" + std::to_string(shard_id_) + "_" + std::to_string(gen) + ".log";
}

Status WalStreamer::PushSince(uint64_t last_lsn, SyncReceiver* receiver, uint64_t* new_last_lsn) {
    auto env = Env::Default();
    uint64_t count = 0;
    uint64_t current_lsn = last_lsn;

    for (uint64_t g = 0; env->FileExists(SegmentPath(g)).ok(); ++g) {
        std::unique_ptr<RandomAccessFile> raf;
        auto st = env->NewRandomAccessFile(SegmentPath(g), &raf);
        if (!st.ok()) return st;

        uint64_t file_size = 0;
        st = env->GetFileSize(SegmentPath(g), &file_size);
        if (!st.ok()) return st;

        uint64_t off = 0;
        if (file_size >= sizeof(WalFileHeader)) {
            char hdr_buf[sizeof(WalFileHeader)];
            Slice hdr_slice;
            st = raf->Read(0, sizeof(WalFileHeader), &hdr_slice);
            if (!st.ok()) return st;
            
            if (std::memcmp(hdr_slice.data(), kWalMagic, 12) == 0) {
                off = sizeof(WalFileHeader);
            }
        }

        while (off + sizeof(FrameHeader) <= file_size) {
            FrameHeader fh{};
            Slice fh_slice;
            st = raf->Read(off, sizeof(fh), &fh_slice);
            if (!st.ok()) return st;
            std::memcpy(&fh, fh_slice.data(), sizeof(fh));

            const uint64_t body_off = off + sizeof(FrameHeader);
            if (body_off + fh.len > file_size) break; // truncated

            std::vector<uint8_t> body(fh.len);
            Slice body_slice;
            st = raf->Read(body_off, fh.len, &body_slice);
            if (!st.ok()) return st;
            
            // Slice body_slice now points to internal buffer, so we must COPY it 
            // to our 'body' vector if we want to own it or if we trust the internal buffer until next Read.
            // But since we use 'body' later, let's copy it.
            std::memcpy(body.data(), body_slice.data(), body_slice.size());

            uint32_t stored_crc = 0;
            std::memcpy(&stored_crc, body.data() + (fh.len - 4), 4);
            const uint32_t calc_crc = util::Crc32c(body.data(), fh.len - 4);
            if (stored_crc != calc_crc) return Status::Corruption("WAL CRC mismatch during sync");

            const auto* rp = reinterpret_cast<const RecordPrefix*>(body.data());
            if (rp->seq > current_lsn) {
                WalEntry entry;
                entry.lsn = rp->seq;
                entry.op = rp->op;
                entry.id = rp->id;
                entry.dim = rp->dim;

                if (rp->op == 1 /* kPut */) {
                    entry.vec = std::span<const float>(
                        reinterpret_cast<const float*>(body.data() + sizeof(RecordPrefix)), rp->dim);
                } else if (rp->op == 3 /* kPutMeta */) {
                    const size_t vec_bytes = rp->dim * sizeof(float);
                    const float* vec_ptr = reinterpret_cast<const float*>(body.data() + sizeof(RecordPrefix));
                    entry.vec = std::span<const float>(vec_ptr, rp->dim);

                    const uint8_t* meta_ptr = body.data() + sizeof(RecordPrefix) + vec_bytes;
                    uint32_t meta_len = 0;
                    std::memcpy(&meta_len, meta_ptr, 4);
                    entry.meta.tenant = std::string(reinterpret_cast<const char*>(meta_ptr + 4), meta_len);
                } else if (rp->op == 4 /* kRawKV */) {
                    uint32_t klen, vlen;
                    std::memcpy(&klen, body.data() + sizeof(RecordPrefix), 4);
                    std::memcpy(&vlen, body.data() + sizeof(RecordPrefix) + 4, 4);
                    const char* kdata = reinterpret_cast<const char*>(body.data() + sizeof(RecordPrefix) + 8);
                    const char* vdata = kdata + klen;
                    entry.raw_data = std::string(kdata, klen) + ":" + std::string(vdata, vlen);
                } else if (rp->op == 5 || rp->op == 6) {
                    // BatchStart (5) or BatchEnd (6): No extra payload, just the op marker.
                }

                st = receiver->Receive(entry);
                if (!st.ok()) return st;
                
                current_lsn = rp->seq;
                count++;
            }
            off = body_off + fh.len;
        }
    }

    if (new_last_lsn) *new_last_lsn = current_lsn;
    return Status::Ok();
}

} // namespace pomai::core
