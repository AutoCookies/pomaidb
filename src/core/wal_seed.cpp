#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>
#include <pomai/storage/crc64.h>
#include <pomai/core/posix_compat.h>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace pomai
{

    WalReplayStats Wal::ReplayToSeed(Seed &seed)
    {
        return ReplayToSeed(seed, 0);
    }

    WalReplayStats Wal::ReplayToSeed(Seed &seed, Lsn min_lsn)
    {
        crc64_init();

        if (seed.Dim() != dim_)
            throw std::runtime_error("Replay seed dim mismatch");

        struct stat st;
        if (::stat(wal_path_.c_str(), &st) != 0)
        {
            if (errno == ENOENT)
                return WalReplayStats{};
            Wal::ThrowSys("stat WAL failed");
        }
        if (!S_ISREG(st.st_mode))
            return WalReplayStats{};

        int rfd = ::open(wal_path_.c_str(), O_RDWR | O_CLOEXEC);
        if (rfd < 0)
            Wal::ThrowSys("open WAL for replay failed: " + wal_path_);

        const off_t original_size = st.st_size;
        off_t truncated_bytes = 0;

        Lsn last_lsn = 0;
        std::size_t total_replayed = 0;
        std::size_t total_vectors = 0;

        const std::size_t header_size = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t);
        const std::size_t footer_size = 24;
        const std::size_t max_payload_allowed = MAX_WAL_PAYLOAD_BYTES;

        while (true)
        {
            off_t rec_start = lseek(rfd, 0, SEEK_CUR);
            if (rec_start == -1)
            {
                std::cerr << "[WAL] Error: lseek failed during replay: " << std::strerror(errno) << "\n";
                break;
            }

            off_t rem = original_size - rec_start;
            if (rem == 0)
                break;
            if ((off_t)header_size > rem)
            {
                if (original_size > rec_start)
                {
                    if (ftruncate(rfd, rec_start) != 0)
                        std::cerr << "[WAL] Warning: ftruncate failed while truncating partial header: " << std::strerror(errno) << "\n";
                    else
                    {
                        if (::fdatasync(rfd) != 0)
                            std::cerr << "[WAL] Warning: fdatasync failed after truncate: " << std::strerror(errno) << "\n";
                        int dfd = ::open(wal_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
                        if (dfd >= 0)
                        {
                            if (::fsync(dfd) != 0)
                                std::cerr << "[WAL] Warning: fsync(dir) failed after truncate: " << std::strerror(errno) << "\n";
                            ::close(dfd);
                        }
                    }
                    truncated_bytes = original_size - rec_start;
                }
                break;
            }

            uint8_t head_buf[sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t)];
            if (!ReadExact(rfd, head_buf, sizeof(head_buf)))
            {
                if (original_size > rec_start)
                {
                    if (ftruncate(rfd, rec_start) != 0)
                        std::cerr << "[WAL] Warning: ftruncate failed while truncating missing-header tail: " << std::strerror(errno) << "\n";
                    else
                    {
                        if (::fdatasync(rfd) != 0)
                            std::cerr << "[WAL] Warning: fdatasync failed after truncate: " << std::strerror(errno) << "\n";
                        int dfd = ::open(wal_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
                        if (dfd >= 0)
                        {
                            if (::fsync(dfd) != 0)
                                std::cerr << "[WAL] Warning: fsync(dir) failed after truncate: " << std::strerror(errno) << "\n";
                            ::close(dfd);
                        }
                    }
                    truncated_bytes = original_size - rec_start;
                }
                break;
            }

            const uint8_t *p = head_buf;
            uint64_t lsn_le;
            uint32_t count_le;
            uint16_t dim_le;

            std::memcpy(&lsn_le, p, sizeof(lsn_le));
            p += sizeof(lsn_le);
            std::memcpy(&count_le, p, sizeof(count_le));
            p += sizeof(count_le);
            std::memcpy(&dim_le, p, sizeof(dim_le));
            p += sizeof(dim_le);

#if __BYTE_ORDER == __BIG_ENDIAN
            lsn_le = __builtin_bswap64(lsn_le);
            count_le = __builtin_bswap32(count_le);
            dim_le = __builtin_bswap16(dim_le);
#endif

            uint64_t lsn64 = lsn_le;
            uint32_t count = count_le;
            uint16_t dim16 = dim_le;

            if (dim16 != dim_)
            {
                std::cerr << "[WAL] Error: dimension mismatch in record. Truncating at offset " << rec_start << "\n";
                if (ftruncate(rfd, rec_start) != 0)
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                truncated_bytes = original_size - rec_start;
                break;
            }

            const uint64_t per_entry_bytes = static_cast<uint64_t>(sizeof(uint64_t) + dim_ * sizeof(float));
            if (count != 0 && per_entry_bytes > 0 && count > (UINT64_MAX / per_entry_bytes))
            {
                std::cerr << "[WAL] Error: count overflow in WAL header. Truncating at offset " << rec_start << "\n";
                if (ftruncate(rfd, rec_start) != 0)
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                truncated_bytes = original_size - rec_start;
                break;
            }

            uint64_t rest_payload_bytes_u64 = static_cast<uint64_t>(count) * per_entry_bytes;
            off_t after_header = rec_start + static_cast<off_t>(header_size);
            off_t remaining_after_header = original_size - after_header;
            if (remaining_after_header < 0 ||
                static_cast<uint64_t>(remaining_after_header) < (rest_payload_bytes_u64 + footer_size))
            {
                std::cerr << "[WAL] Error: truncated payload detected during replay. Truncating at offset " << rec_start << "\n";
                if (ftruncate(rfd, rec_start) != 0)
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                truncated_bytes = original_size - rec_start;
                break;
            }

            std::vector<uint8_t> payload_rest;
            payload_rest.resize(static_cast<std::size_t>(rest_payload_bytes_u64));
            if (rest_payload_bytes_u64 > 0)
            {
                if (!ReadExact(rfd, payload_rest.data(), payload_rest.size()))
                {
                    std::cerr << "[WAL] Error: truncated payload during read. Truncating at offset " << rec_start << "\n";
                    if (ftruncate(rfd, rec_start) != 0)
                        std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                    truncated_bytes = original_size - rec_start;
                    break;
                }
            }

            uint8_t footer_buf[24];
            if (!ReadExact(rfd, footer_buf, sizeof(footer_buf)))
            {
                std::cerr << "[WAL] Error: missing footer at end of record. Truncating at offset " << rec_start << "\n";
                if (ftruncate(rfd, rec_start) != 0)
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                truncated_bytes = original_size - rec_start;
                break;
            }

            const uint8_t *f = footer_buf;
            uint32_t payload_size_le;
            uint32_t reserved_le;
            uint64_t crc_le;
            uint64_t magic_le;
            std::memcpy(&payload_size_le, f, 4);
            f += 4;
            std::memcpy(&reserved_le, f, 4);
            f += 4;
            std::memcpy(&crc_le, f, 8);
            f += 8;
            std::memcpy(&magic_le, f, 8);
            f += 8;

#if __BYTE_ORDER == __BIG_ENDIAN
            payload_size_le = __builtin_bswap32(payload_size_le);
            reserved_le = __builtin_bswap32(reserved_le);
            crc_le = __builtin_bswap64(crc_le);
            magic_le = __builtin_bswap64(magic_le);
#endif

            uint32_t payload_size = payload_size_le;
            uint64_t crc = crc_le;
            uint64_t magic = magic_le;

            std::vector<uint8_t> whole_payload;
            whole_payload.reserve(header_size + payload_rest.size());
            whole_payload.insert(whole_payload.end(), head_buf, head_buf + header_size);
            if (!payload_rest.empty())
                whole_payload.insert(whole_payload.end(), payload_rest.begin(), payload_rest.end());

            if (whole_payload.size() != payload_size)
            {
                std::cerr << "[WAL] Error: payload size mismatch. Truncating at offset " << rec_start << "\n";
                if (ftruncate(rfd, rec_start) != 0)
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                truncated_bytes = original_size - rec_start;
                break;
            }

            if (payload_size > max_payload_allowed)
            {
                std::cerr << "[WAL] Error: payload size too large (" << payload_size << "). Truncating at offset " << rec_start << "\n";
                if (ftruncate(rfd, rec_start) != 0)
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                truncated_bytes = original_size - rec_start;
                break;
            }

            if (magic != FOOTER_MAGIC)
            {
                std::cerr << "[WAL] Error: footer magic mismatch. Truncating at offset " << rec_start << "\n";
                if (ftruncate(rfd, rec_start) != 0)
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                truncated_bytes = original_size - rec_start;
                break;
            }

            uint64_t calc = crc64(0, whole_payload.data(), whole_payload.size());
            if (calc != crc)
            {
                std::cerr << "[WAL] Error: CRC mismatch. Truncating at offset " << rec_start << "\n";
                if (ftruncate(rfd, rec_start) != 0)
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                truncated_bytes = original_size - rec_start;
                break;
            }

            const uint8_t *qptr = whole_payload.data();
            qptr += sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t);

            std::vector<UpsertRequest> batch;
            batch.reserve(count);

            for (uint32_t i = 0; i < count; ++i)
            {
                uint64_t id_le2;
                std::memcpy(&id_le2, qptr, sizeof(id_le2));
                qptr += sizeof(id_le2);
#if __BYTE_ORDER == __BIG_ENDIAN
                id_le2 = __builtin_bswap64(id_le2);
#endif
                uint64_t id = id_le2;

                UpsertRequest r;
                r.id = static_cast<Id>(id);
                r.vec.data.resize(dim_);
                std::memcpy(r.vec.data.data(), qptr, dim_ * sizeof(float));
                qptr += dim_ * sizeof(float);
                batch.push_back(std::move(r));
            }

            if (lsn64 > min_lsn)
            {
                try
                {
                    seed.ApplyUpserts(batch);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[WAL] Error applying batch during replay: " << e.what() << "\n";
                    if (ftruncate(rfd, rec_start) != 0)
                        std::cerr << "[WAL] Warning: ftruncate failed after apply error: " << std::strerror(errno) << "\n";
                    truncated_bytes = original_size - rec_start;
                    break;
                }

                total_replayed++;
                total_vectors += count;
            }

            last_lsn = lsn64;

            off_t cur = lseek(rfd, 0, SEEK_CUR);
            if (cur == -1)
            {
                truncated_bytes = 0;
                break;
            }
        }

        ::close(rfd);

        WalReplayStats stats;
        stats.last_lsn = last_lsn;
        stats.records_applied = total_replayed;
        stats.vectors_applied = total_vectors;
        stats.truncated_bytes = static_cast<std::size_t>(truncated_bytes);

        Lsn want = last_lsn + 1;
        Lsn cur = next_lsn_.load(std::memory_order_relaxed);
        if (want > cur)
            next_lsn_.store(want, std::memory_order_relaxed);

        return stats;
    }

} // namespace pomai
