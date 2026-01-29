#include "wal.h"
#include "seed.h"
#include "crc64.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <filesystem>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <algorithm>
#include <endian.h>
#include <mutex>
#include <limits>
#include <memory>

namespace pomai
{

    static constexpr uint64_t FOOTER_MAGIC = UINT64_C(0x706f6d616977616c);

    void Wal::WaitDurable(Lsn lsn)
    {
        if (lsn == 0)
            return;
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]
                 { return !running_.load() || durable_lsn_ >= lsn; });
    }

    static void ThrowSys(const std::string &what)
    {
        throw std::runtime_error(what + ": " + std::string(std::strerror(errno)));
    }

    static void EnsureDirExists(const std::string &dir)
    {
        std::error_code ec;
        if (!std::filesystem::create_directories(dir, ec) && ec)
        {
            throw std::runtime_error("failed to create directory " + dir + ": " + ec.message());
        }
    }

    void Wal::WriteAll(int fd, const void *buf, std::size_t n)
    {
        const auto *p = reinterpret_cast<const uint8_t *>(buf);
        std::size_t off = 0;
        while (off < n)
        {
            ssize_t w = ::write(fd, p + off, n - off);
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                ThrowSys("write failed");
            }
            off += static_cast<std::size_t>(w);
        }
    }

    bool Wal::ReadExact(int fd, void *buf, std::size_t n)
    {
        auto *p = reinterpret_cast<uint8_t *>(buf);
        std::size_t got = 0;
        while (got < n)
        {
            ssize_t r = ::read(fd, p + got, n - got);
            if (r == 0)
                return false;
            if (r < 0)
            {
                if (errno == EINTR)
                    continue;
                ThrowSys("read failed");
            }
            got += static_cast<std::size_t>(r);
        }
        return true;
    }

    Wal::Wal(std::string shard_name, std::string wal_dir, std::size_t dim)
        : shard_name_(std::move(shard_name)),
          wal_dir_(std::move(wal_dir)),
          dim_(dim)
    {
        if (dim_ == 0)
            throw std::runtime_error("Wal dim must be > 0");
        wal_path_ = wal_dir_ + "/" + shard_name_ + ".wal";
    }

    Wal::~Wal()
    {
        Stop();
    }

    void Wal::OpenOrCreateForAppend()
    {
        EnsureDirExists(wal_dir_);

        fd_ = ::open(wal_path_.c_str(),
                     O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC,
                     0644);
        if (fd_ < 0)
            ThrowSys("open WAL failed: " + wal_path_);
    }

    void Wal::CloseFd()
    {
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

    void Wal::Start()
    {
        if (running_.exchange(true))
            return;

        crc64_init();
        OpenOrCreateForAppend();
        writer_th_ = std::thread(&Wal::WriterLoop, this);
        fsync_th_ = std::thread(&Wal::FsyncLoop, this);
    }

    void Wal::Stop()
    {
        if (!running_.exchange(false))
            return;

        cv_.notify_all();
        if (writer_th_.joinable())
            writer_th_.join();
        if (fsync_th_.joinable())
            fsync_th_.join();

        if (fd_ >= 0)
            ::fdatasync(fd_);
        CloseFd();
    }

    Lsn Wal::AppendUpserts(const std::vector<UpsertRequest> &batch, bool wait_durable)
    {
        if (batch.empty())
            return 0;

        if (batch.size() > MAX_BATCH_ROWS)
            throw std::runtime_error("WAL append rejected: batch too large; split into smaller batches");

        const uint64_t per_entry_bytes = static_cast<uint64_t>(sizeof(uint64_t) + dim_ * sizeof(float));
        if (per_entry_bytes == 0)
            throw std::runtime_error("WAL append failed: invalid per-entry size");

        const uint64_t count = static_cast<uint64_t>(batch.size());
        if (count > 0 && per_entry_bytes > 0 && count > (UINT64_MAX / per_entry_bytes))
            throw std::runtime_error("WAL append rejected: batch size would overflow payload calculations");

        const std::size_t payload_size =
            sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t) +
            static_cast<std::size_t>(count * per_entry_bytes);

        if (payload_size > MAX_WAL_PAYLOAD_BYTES)
            throw std::runtime_error("WAL append rejected: payload too large; split batch");

        if (payload_size > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()))
            throw std::runtime_error("WAL append rejected: payload larger than 4GB (unsupported)");

        Lsn lsn = next_lsn_.fetch_add(1, std::memory_order_relaxed);

        const uint32_t count_u32 = static_cast<uint32_t>(batch.size());
        const uint16_t dim16 = static_cast<uint16_t>(dim_);

        std::vector<uint8_t> buf;
        buf.reserve(payload_size + 24);

        auto push_le = [&](const void *ptr, std::size_t n)
        {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(ptr);
            buf.insert(buf.end(), p, p + n);
        };

        uint64_t lsn_le = (uint64_t)lsn;
        uint32_t count_le = count_u32;
        uint16_t dim_le = dim16;

#if __BYTE_ORDER == __BIG_ENDIAN
        lsn_le = __builtin_bswap64(lsn_le);
        count_le = __builtin_bswap32(count_le);
        dim_le = __builtin_bswap16(dim_le);
#endif

        push_le(&lsn_le, sizeof(lsn_le));
        push_le(&count_le, sizeof(count_le));
        push_le(&dim_le, sizeof(dim_le));

        for (const auto &it : batch)
        {
            if (it.vec.data.size() != dim_)
                throw std::runtime_error("WAL dim mismatch in batch write");

            uint64_t id_le = static_cast<uint64_t>(it.id);
#if __BYTE_ORDER == __BIG_ENDIAN
            id_le = __builtin_bswap64(id_le);
#endif
            push_le(&id_le, sizeof(id_le));
            const uint8_t *vf = reinterpret_cast<const uint8_t *>(it.vec.data.data());
            push_le(vf, dim_ * sizeof(float));
        }

        uint64_t crc = crc64(0, buf.data(), buf.size());

        uint32_t payload_size_u32 = static_cast<uint32_t>(buf.size());
        uint32_t reserved = 0;
        uint64_t crc_le = crc;
        uint64_t magic_le = FOOTER_MAGIC;

#if __BYTE_ORDER == __BIG_ENDIAN
        payload_size_u32 = __builtin_bswap32(payload_size_u32);
        reserved = __builtin_bswap32(reserved);
        crc_le = __builtin_bswap64(crc_le);
        magic_le = __builtin_bswap64(magic_le);
#endif

        push_le(&payload_size_u32, sizeof(payload_size_u32));
        push_le(&reserved, sizeof(reserved));
        push_le(&crc_le, sizeof(crc_le));
        push_le(&magic_le, sizeof(magic_le));

        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!running_.load())
                ThrowSys("WAL not started");
            pending_writes_.push_back(Pending{lsn, std::move(buf)});
            cv_.notify_all();
        }

        if (wait_durable)
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (durable_lsn_ >= lsn)
                return lsn;
            lk.unlock();
            // As fallback, attempt immediate fdatasync to speed up durability
            if (fd_ >= 0)
            {
                if (::fdatasync(fd_) != 0)
                    ThrowSys("fdatasync failed in AppendUpserts (wait_durable)");
            }
            std::unique_lock<std::mutex> lk2(mu_);
            cv_.wait(lk2, [&]
                     { return durable_lsn_ >= lsn; });
        }

        return lsn;
    }

    void Wal::WriterLoop()
    {
        while (running_.load())
        {
            std::deque<Pending> to_write;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]
                         { return !running_.load() || !pending_writes_.empty(); });
                if (!running_.load() && pending_writes_.empty())
                    break;
                to_write.swap(pending_writes_);
            }

            for (auto &p : to_write)
            {
                if (fd_ < 0)
                    ThrowSys("WAL writer fd invalid");
                WriteAll(fd_, p.buf.data(), p.buf.size());
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    written_lsn_ = p.lsn;
                    cv_.notify_all();
                }
            }

            // ensure background durability progress: fsync file and advance durable_lsn_
            if (fd_ >= 0)
            {
                if (::fdatasync(fd_) != 0)
                    std::cerr << "[WAL] fdatasync failed in writer: " << std::strerror(errno) << "\n";
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                durable_lsn_ = written_lsn_;
                cv_.notify_all();
            }
        }
    }

    void Wal::FsyncLoop()
    {
        using namespace std::chrono_literals;
        const auto interval = 100ms;

        while (running_.load())
        {
            std::this_thread::sleep_for(interval);
            std::lock_guard<std::mutex> lk(mu_);
            if (written_lsn_ > durable_lsn_)
            {
                if (fd_ >= 0)
                {
                    if (::fdatasync(fd_) != 0)
                        std::cerr << "[WAL] fdatasync failed in fsync loop: " << std::strerror(errno) << "\n";
                }
                durable_lsn_ = written_lsn_;
                cv_.notify_all();
            }
        }
    }

    Lsn Wal::WrittenLsn() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return written_lsn_;
    }

    void Wal::TruncateToZero()
    {
        std::lock_guard<std::mutex> lk(mu_);

        if (fd_ < 0)
            ThrowSys("truncate wal: wal fd invalid");

        if (ftruncate(fd_, 0) != 0)
            ThrowSys("ftruncate failed during checkpoint");

        if (::fdatasync(fd_) != 0)
            ThrowSys("fdatasync failed after truncate");

        int dfd = ::open(wal_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0)
        {
            if (::fsync(dfd) != 0)
            {
                ::close(dfd);
                ThrowSys("fsync(dir) failed after truncate");
            }
            ::close(dfd);
        }
        else
        {
            ThrowSys("open(wal_dir) for fsync after truncate failed");
        }

        next_lsn_.store(1, std::memory_order_relaxed);
        written_lsn_ = 0;
        durable_lsn_ = 0;
        pending_writes_.clear();
        cv_.notify_all();
    }

    WalReplayStats Wal::ReplayToSeed(Seed &seed)
    {
        crc64_init();

        if (seed.Dim() != dim_)
            throw std::runtime_error("Replay seed dim mismatch");

        struct stat st;
        if (::stat(wal_path_.c_str(), &st) != 0)
        {
            if (errno == ENOENT)
                return WalReplayStats{};
            ThrowSys("stat WAL failed");
        }
        if (!S_ISREG(st.st_mode))
            return WalReplayStats{};

        int rfd = ::open(wal_path_.c_str(), O_RDWR | O_CLOEXEC);
        if (rfd < 0)
            ThrowSys("open WAL for replay failed: " + wal_path_);

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

            last_lsn = lsn64;
            total_replayed++;
            total_vectors += count;

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