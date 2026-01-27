// Full wal.cpp with synchronous fdatasync fast-path in AppendUpserts
#include "wal.h"
#include "seed.h"
#include "crc64.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <algorithm>
#include <endian.h> // for byte-order helpers on glibc

namespace pomai
{

    // Footer magic (8 bytes). Chosen mnemonic ASCII 'pomaiwal'
    static constexpr uint64_t FOOTER_MAGIC = UINT64_C(0x706f6d616977616c); // "pomaiwal" little-endian

    static void ThrowSys(const std::string &what)
    {
        throw std::runtime_error(what + ": " + std::string(std::strerror(errno)));
    }

    static void EnsureDirExists(const std::string &dir)
    {
        struct stat st;
        if (::stat(dir.c_str(), &st) == 0)
        {
            if (!S_ISDIR(st.st_mode))
                throw std::runtime_error("WAL dir not a directory: " + dir);
            return;
        }
        if (::mkdir(dir.c_str(), 0755) != 0)
            ThrowSys("mkdir failed for " + dir);
    }

    // Write-all helper (throws on error)
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

    // Read exact n bytes (returns false on EOF)
    bool Wal::ReadExact(int fd, void *buf, std::size_t n)
    {
        auto *p = reinterpret_cast<uint8_t *>(buf);
        std::size_t got = 0;
        while (got < n)
        {
            ssize_t r = ::read(fd, p + got, n - got);
            if (r == 0)
                return false; // EOF
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

    Wal::~Wal() { Stop(); }

    void Wal::OpenOrCreateForAppend()
    {
        EnsureDirExists(wal_dir_);

        // O_APPEND ensures writes append to the end. Use CLOEXEC so children don't inherit fd.
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
        fsync_th_ = std::thread(&Wal::FsyncLoop, this);
    }

    void Wal::Stop()
    {
        if (!running_.exchange(false))
            return;

        cv_.notify_all();
        if (fsync_th_.joinable())
            fsync_th_.join();

        if (fd_ >= 0)
            ::fdatasync(fd_);
        CloseFd();
    }

    // New: synchronous fast-path optional flag
    Lsn Wal::AppendUpserts(const std::vector<UpsertRequest> &batch, bool wait_durable)
    {
        if (batch.empty())
            return 0;

        Lsn lsn = next_lsn_.fetch_add(1, std::memory_order_relaxed);

        const uint32_t count = static_cast<uint32_t>(batch.size());
        const uint16_t dim16 = static_cast<uint16_t>(dim_);

        const std::size_t payload_size =
            sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t) +
            count * (sizeof(uint64_t) + dim_ * sizeof(float));

        std::vector<uint8_t> buf;
        buf.reserve(payload_size + 24);

        auto push_le = [&](const void *ptr, std::size_t n)
        {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(ptr);
            buf.insert(buf.end(), p, p + n);
        };

        uint64_t lsn_le = (uint64_t)lsn;
        uint32_t count_le = (uint32_t)count;
        uint16_t dim_le = (uint16_t)dim16;

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

        // Write buffer (single write loop)
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (fd_ < 0)
                throw std::runtime_error("WAL not started");
            WriteAll(fd_, buf.data(), buf.size());
            // Announce written lsn for background thread and waiters
            written_lsn_ = lsn;
            cv_.notify_all();
        }

        // If caller requested synchronous durability, perform fdatasync now.
        if (wait_durable)
        {
            if (fd_ >= 0)
            {
                if (::fdatasync(fd_) != 0)
                {
                    // Critical: if fdatasync fails, throw so caller knows write wasn't durable.
                    ThrowSys("fdatasync failed in AppendUpserts (wait_durable)");
                }
            }

            // Update durable_lsn_ and notify waiters
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (durable_lsn_ < lsn)
                    durable_lsn_ = lsn;
                cv_.notify_all();
            }
        }

        return lsn;
    }

    void Wal::WaitDurable(Lsn lsn)
    {
        if (lsn == 0)
            return;

        {
            std::unique_lock<std::mutex> lk(mu_);
            if (!running_.load() || durable_lsn_ >= lsn)
                return;
        }

        // If durable_lsn_ hasn't yet caught up, attempt an immediate fdatasync as fallback.
        if (fd_ >= 0)
        {
            if (::fdatasync(fd_) != 0)
                ThrowSys("fdatasync failed in WaitDurable");
        }

        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]
                 { return !running_.load() || durable_lsn_ >= lsn; });
    }

    void Wal::FsyncLoop()
    {
        using namespace std::chrono_literals;
        const auto interval = 5ms;

        std::unique_lock<std::mutex> lk(mu_);
        while (running_.load())
        {
            cv_.wait_for(lk, interval, [&]
                         { return !running_.load() || durable_lsn_ < written_lsn_; });

            if (!running_.load())
                break;

            if (durable_lsn_ < written_lsn_)
            {
                const Lsn target = written_lsn_;
                lk.unlock();
                if (fd_ >= 0)
                {
                    if (::fdatasync(fd_) != 0)
                        std::cerr << "[WAL] fdatasync failed in background: " << std::strerror(errno) << "\n";
                }
                lk.lock();
                durable_lsn_ = target;
                cv_.notify_all();
            }
        }
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

        int rfd = ::open(wal_path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (rfd < 0)
            ThrowSys("open WAL for replay failed: " + wal_path_);

        Lsn last_lsn = 0;
        off_t good_offset = 0;
        std::size_t total_replayed = 0;
        std::size_t total_vectors = 0;
        const off_t original_size = st.st_size;

        while (true)
        {
            uint8_t head_buf[sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t)];
            if (!ReadExact(rfd, head_buf, sizeof(head_buf)))
            {
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
                std::cerr << "[WAL] Error: dimension mismatch in record. Stopping replay.\n";
                break;
            }

            const std::size_t rest_payload_bytes = static_cast<std::size_t>(count) * (sizeof(uint64_t) + dim_ * sizeof(float));

            std::vector<uint8_t> payload_rest;
            payload_rest.resize(rest_payload_bytes);
            if (rest_payload_bytes > 0)
            {
                if (!ReadExact(rfd, payload_rest.data(), rest_payload_bytes))
                {
                    std::cerr << "[WAL] Error: truncated payload detected during replay. Stopping.\n";
                    break;
                }
            }

            uint8_t footer_buf[24];
            if (!ReadExact(rfd, footer_buf, sizeof(footer_buf)))
            {
                std::cerr << "[WAL] Error: missing footer at end of record. Truncation suspected. Stopping.\n";
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
            whole_payload.reserve(sizeof(head_buf) + payload_rest.size());
            whole_payload.insert(whole_payload.end(), head_buf, head_buf + sizeof(head_buf));
            if (!payload_rest.empty())
                whole_payload.insert(whole_payload.end(), payload_rest.begin(), payload_rest.end());

            if (whole_payload.size() != payload_size)
            {
                std::cerr << "[WAL] Error: payload size mismatch. Stopping replay.\n";
                break;
            }

            if (magic != FOOTER_MAGIC)
            {
                std::cerr << "[WAL] Error: footer magic mismatch. Stopping replay.\n";
                break;
            }

            uint64_t calc = crc64(0, whole_payload.data(), whole_payload.size());
            if (calc != crc)
            {
                std::cerr << "[WAL] Error: CRC mismatch. Stopping replay.\n";
                break;
            }

            std::vector<UpsertRequest> batch;
            batch.reserve(count);

            const uint8_t *qptr = whole_payload.data();
            qptr += sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t);

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
                break;
            }

            last_lsn = lsn64;
            total_replayed++;
            total_vectors += count;

            off_t cur = lseek(rfd, 0, SEEK_CUR);
            if (cur == -1)
                good_offset = -1;
            else
                good_offset = cur;
        }

        ::close(rfd);

        off_t truncated = 0;
        if (good_offset > 0)
        {
            int wfd = ::open(wal_path_.c_str(), O_WRONLY | O_CLOEXEC);
            if (wfd >= 0)
            {
                if (ftruncate(wfd, good_offset) != 0)
                {
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                }
                else
                {
                    if (::fdatasync(wfd) != 0)
                    {
                        std::cerr << "[WAL] Warning: fdatasync(wal) after truncate failed: " << std::strerror(errno) << "\n";
                    }

                    int dfd = ::open(wal_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
                    if (dfd >= 0)
                    {
                        if (::fsync(dfd) != 0)
                        {
                            std::cerr << "[WAL] Warning: fsync(dir) after truncate failed: " << std::strerror(errno) << "\n";
                        }
                        ::close(dfd);
                    }
                    else
                    {
                        std::cerr << "[WAL] Warning: open(dir) failed for fsync: " << std::strerror(errno) << "\n";
                    }
                    truncated = original_size > good_offset ? (original_size - good_offset) : 0;
                }
                ::close(wfd);
            }
            else
            {
                std::cerr << "[WAL] Warning: open(wal) for truncate fsync failed: " << std::strerror(errno) << "\n";
            }
        }

        Lsn want = last_lsn + 1;
        Lsn cur = next_lsn_.load(std::memory_order_relaxed);
        if (want > cur)
            next_lsn_.store(want, std::memory_order_relaxed);

        WalReplayStats stats;
        stats.last_lsn = last_lsn;
        stats.records_applied = total_replayed;
        stats.vectors_applied = total_vectors;
        stats.truncated_bytes = truncated;

        return stats;
    }

} // namespace pomai