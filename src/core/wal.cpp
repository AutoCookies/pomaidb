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
#include <endian.h>
#include <mutex>
#include <limits>
#include <filesystem>

namespace pomai
{
    static constexpr uint64_t FOOTER_MAGIC = UINT64_C(0x706f6d616977616c);

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
                ThrowSys("WAL write failed");
            }
            if (w == 0)
                throw std::runtime_error("WAL write returned 0");
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
                ThrowSys("WAL read failed");
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
        fd_ = ::open(wal_path_.c_str(), O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC, 0644);
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

    Lsn Wal::AppendUpserts(const std::vector<UpsertRequest> &batch, bool wait_durable)
    {
        if (batch.empty())
            return 0;

        const uint64_t per_entry_bytes = static_cast<uint64_t>(sizeof(uint64_t) + dim_ * sizeof(float));
        const uint64_t count = static_cast<uint64_t>(batch.size());

        if (count > (UINT64_MAX / per_entry_bytes))
            throw std::runtime_error("WAL batch overflow");

        const std::size_t payload_size = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t) +
                                         static_cast<std::size_t>(count * per_entry_bytes);

        if (batch.size() > MAX_BATCH_ROWS || payload_size > MAX_WAL_PAYLOAD_BYTES)
            throw std::runtime_error("WAL payload limit exceeded");

        Lsn lsn = next_lsn_.fetch_add(1, std::memory_order_relaxed);
        uint64_t lsn_le = (uint64_t)lsn;
        uint32_t count_le = (uint32_t)batch.size();
        uint16_t dim_le = (uint16_t)dim_;

#if __BYTE_ORDER == __BIG_ENDIAN
        lsn_le = __builtin_bswap64(lsn_le);
        count_le = __builtin_bswap32(count_le);
        dim_le = __builtin_bswap16(dim_le);
#endif

        std::vector<uint8_t> buf;
        buf.reserve(payload_size + 24);

        auto push_data = [&](const void *ptr, std::size_t n)
        {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(ptr);
            buf.insert(buf.end(), p, p + n);
        };

        push_data(&lsn_le, sizeof(lsn_le));
        push_data(&count_le, sizeof(count_le));
        push_data(&dim_le, sizeof(dim_le));

        for (const auto &it : batch)
        {
            uint64_t id_le = static_cast<uint64_t>(it.id);
#if __BYTE_ORDER == __BIG_ENDIAN
            id_le = __builtin_bswap64(id_le);
#endif
            push_data(&id_le, sizeof(id_le));
            push_data(it.vec.data.data(), dim_ * sizeof(float));
        }

        uint64_t crc = crc64(0, buf.data(), buf.size());
        uint32_t p_size_u32 = static_cast<uint32_t>(buf.size());
        uint32_t res = 0;
        uint64_t crc_le = crc;
        uint64_t mag_le = FOOTER_MAGIC;

#if __BYTE_ORDER == __BIG_ENDIAN
        p_size_u32 = __builtin_bswap32(p_size_u32);
        res = __builtin_bswap32(res);
        crc_le = __builtin_bswap64(crc_le);
        mag_le = __builtin_bswap64(mag_le);
#endif

        push_data(&p_size_u32, sizeof(p_size_u32));
        push_data(&res, sizeof(res));
        push_data(&crc_le, sizeof(crc_le));
        push_data(&mag_le, sizeof(mag_le));

        {
            std::lock_guard<std::mutex> lk(mu_);
            if (fd_ < 0)
                throw std::runtime_error("WAL FD invalid");
            WriteAll(fd_, buf.data(), buf.size());
            written_lsn_ = lsn;
            cv_.notify_all();
        }

        if (wait_durable)
        {
            if (fd_ >= 0)
            {
                if (::fdatasync(fd_) != 0)
                    ThrowSys("fdatasync failed");
            }
            std::lock_guard<std::mutex> lk(mu_);
            if (durable_lsn_ < lsn)
                durable_lsn_ = lsn;
            cv_.notify_all();
        }

        return lsn;
    }

    void Wal::WaitDurable(Lsn lsn)
    {
        if (lsn == 0)
            return;

        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]
                 { return !running_.load() || durable_lsn_ >= lsn; });
    }

    void Wal::FsyncLoop()
    {
        using namespace std::chrono_literals;
        std::unique_lock<std::mutex> lk(mu_);
        while (running_.load())
        {
            cv_.wait_for(lk, 5ms, [&]
                         { return !running_.load() || durable_lsn_ < written_lsn_; });

            if (!running_.load())
                break;

            if (durable_lsn_ < written_lsn_)
            {
                Lsn target = written_lsn_;
                lk.unlock();
                if (fd_ >= 0)
                {
                    if (::fdatasync(fd_) != 0)
                        std::cerr << "[WAL] background fdatasync failed\n";
                }
                lk.lock();
                durable_lsn_ = target;
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
            ThrowSys("truncate failed: invalid FD");

        if (ftruncate(fd_, 0) != 0)
            ThrowSys("ftruncate failed");

        if (::fdatasync(fd_) != 0)
            ThrowSys("fdatasync failed");

        int dfd = ::open(wal_dir_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd >= 0)
        {
            ::fsync(dfd);
            ::close(dfd);
        }

        next_lsn_.store(1, std::memory_order_relaxed);
        written_lsn_ = 0;
        durable_lsn_ = 0;
        cv_.notify_all();
    }

    WalReplayStats Wal::ReplayToSeed(Seed &seed)
    {
        crc64_init();
        if (seed.Dim() != dim_)
            throw std::runtime_error("replay dimension mismatch");

        struct stat st;
        if (::stat(wal_path_.c_str(), &st) != 0)
        {
            if (errno == ENOENT)
                return WalReplayStats{};
            ThrowSys("stat WAL failed");
        }

        int rfd = ::open(wal_path_.c_str(), O_RDWR | O_CLOEXEC);
        if (rfd < 0)
            ThrowSys("open WAL for replay failed");

        off_t current_size = st.st_size;
        off_t truncated_bytes = 0;
        Lsn last_lsn = 0;
        std::size_t records = 0;
        std::size_t vectors = 0;

        const std::size_t h_size = 14;
        const std::size_t f_size = 24;

        while (true)
        {
            off_t rec_start = lseek(rfd, 0, SEEK_CUR);
            if (rec_start == -1)
                break;

            off_t remaining = current_size - rec_start;
            if (remaining < (off_t)(h_size + f_size))
            {
                if (remaining > 0)
                {
                    ftruncate(rfd, rec_start);
                    truncated_bytes += remaining;
                }
                break;
            }

            uint8_t h_buf[h_size];
            if (!ReadExact(rfd, h_buf, h_size))
                break;

            uint64_t l_le;
            uint32_t c_le;
            uint16_t d_le;
            std::memcpy(&l_le, h_buf, 8);
            std::memcpy(&c_le, h_buf + 8, 4);
            std::memcpy(&d_le, h_buf + 12, 2);

#if __BYTE_ORDER == __BIG_ENDIAN
            l_le = __builtin_bswap64(l_le);
            c_le = __builtin_bswap32(c_le);
            d_le = __builtin_bswap16(d_le);
#endif

            if (d_le != dim_)
            {
                ftruncate(rfd, rec_start);
                truncated_bytes += (current_size - rec_start);
                break;
            }

            uint64_t payload_len = (uint64_t)c_le * (8 + dim_ * 4);
            if (remaining < (off_t)(h_size + payload_len + f_size))
            {
                ftruncate(rfd, rec_start);
                truncated_bytes += remaining;
                break;
            }

            std::vector<uint8_t> p_buf(payload_len);
            if (!ReadExact(rfd, p_buf.data(), payload_len))
                break;

            uint8_t f_buf[f_size];
            if (!ReadExact(rfd, f_buf, f_size))
                break;

            uint32_t p_size_le;
            uint64_t crc_le;
            uint64_t mag_le;
            std::memcpy(&p_size_le, f_buf, 4);
            std::memcpy(&crc_le, f_buf + 8, 8);
            std::memcpy(&mag_le, f_buf + 16, 8);

#if __BYTE_ORDER == __BIG_ENDIAN
            p_size_le = __builtin_bswap32(p_size_le);
            crc_le = __builtin_bswap64(crc_le);
            mag_le = __builtin_bswap64(mag_le);
#endif

            if (mag_le != FOOTER_MAGIC || p_size_le != (h_size + payload_len))
            {
                ftruncate(rfd, rec_start);
                truncated_bytes += (current_size - rec_start);
                break;
            }

            std::vector<uint8_t> check_buf;
            check_buf.reserve(h_size + payload_len);
            check_buf.insert(check_buf.end(), h_buf, h_buf + h_size);
            check_buf.insert(check_buf.end(), p_buf.begin(), p_buf.end());

            if (crc64(0, check_buf.data(), check_buf.size()) != crc_le)
            {
                ftruncate(rfd, rec_start);
                truncated_bytes += (current_size - rec_start);
                break;
            }

            std::vector<UpsertRequest> batch;
            const uint8_t *ptr = p_buf.data();
            for (uint32_t i = 0; i < c_le; ++i)
            {
                uint64_t id_le;
                std::memcpy(&id_le, ptr, 8);
                ptr += 8;
#if __BYTE_ORDER == __BIG_ENDIAN
                id_le = __builtin_bswap64(id_le);
#endif
                UpsertRequest req;
                req.id = static_cast<Id>(id_le);
                req.vec.data.resize(dim_);
                std::memcpy(req.vec.data.data(), ptr, dim_ * 4);
                ptr += dim_ * 4;
                batch.push_back(std::move(req));
            }

            seed.ApplyUpserts(batch);
            last_lsn = l_le;
            records++;
            vectors += c_le;
        }

        ::close(rfd);
        WalReplayStats stats;
        stats.last_lsn = last_lsn;
        stats.records_applied = records;
        stats.vectors_applied = vectors;
        stats.truncated_bytes = (std::size_t)truncated_bytes;

        Lsn next = last_lsn + 1;
        if (next > next_lsn_.load())
            next_lsn_.store(next);

        return stats;
    }
}