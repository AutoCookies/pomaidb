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
#include <endian.h> // for htobe/htole helpers on glibc

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

        // Init CRC table
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

    // Append writes payload+footer in a single contiguous write to reduce syscall window.
    Lsn Wal::AppendUpserts(const std::vector<UpsertRequest> &batch)
    {
        if (batch.empty())
            return 0;

        // Acquire LSN upfront
        Lsn lsn = next_lsn_.fetch_add(1, std::memory_order_relaxed);

        // Serialize payload into a contiguous buffer
        // payload = lsn(u64) + count(u32) + dim(u16) + for each: id(u64) + vec(f32 * dim)
        const uint32_t count = static_cast<uint32_t>(batch.size());
        const uint16_t dim16 = static_cast<uint16_t>(dim_);

        const std::size_t payload_size =
            sizeof(uint64_t) + // lsn
            sizeof(uint32_t) + // count
            sizeof(uint16_t) + // dim
            count * (sizeof(uint64_t) + dim_ * sizeof(float));

        std::vector<uint8_t> buf;
        buf.reserve(payload_size + 24); // footer ~24 bytes

        auto push_le = [&](const void *ptr, std::size_t n)
        {
            const uint8_t *p = reinterpret_cast<const uint8_t *>(ptr);
            buf.insert(buf.end(), p, p + n);
        };

        uint64_t lsn_le = (uint64_t)lsn;
        uint32_t count_le = (uint32_t)count;
        uint16_t dim_le = (uint16_t)dim16;

        // Ensure little-endian on-disk encoding
        // Convert if host is big endian
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
            // vectors are f32 in IEEE754; store raw bytes (we assume same float size)
            const uint8_t *vf = reinterpret_cast<const uint8_t *>(it.vec.data.data());
            push_le(vf, dim_ * sizeof(float));
        }

        // Compute CRC64 over payload
        uint64_t crc = crc64(0, buf.data(), buf.size());

        // Build footer: payload_size(u32 LE) + reserved(u32=0) + crc(u64 LE) + magic(u64 LE)
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

        // Write buffer atomically (as a single write loop) under mutex to keep ordering with written_lsn_
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (fd_ < 0)
                throw std::runtime_error("WAL not started");
            WriteAll(fd_, buf.data(), buf.size());
            written_lsn_ = lsn;
            cv_.notify_all();
        }

        return lsn;
    }

    // Wait durable: if background hasn't fsynced yet, perform a synchronous fdatasync to speed up.
    void Wal::WaitDurable(Lsn lsn)
    {
        if (lsn == 0)
            return;

        {
            std::unique_lock<std::mutex> lk(mu_);
            // If already durable or we're stopping, return
            if (!running_.load() || durable_lsn_ >= lsn)
                return;
        }

        // Fast-path synchronous sync outside of lock to avoid blocking Append writers.
        // This helps callers that require immediate durability (common for small systems).
        if (fd_ >= 0)
        {
            if (::fdatasync(fd_) != 0)
                ThrowSys("fdatasync failed in WaitDurable");
        }

        // Now wait until durable_lsn_ catches up (background thread will update durable_lsn_)
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]
                 { return !running_.load() || durable_lsn_ >= lsn; });
    }

    // Background fsync thread: periodically fdatasync the file if new data is written.
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

    // Replay: read sequentially and verify CRC/footer. Truncate trailing partial/corrupt record if possible.
    Lsn Wal::ReplayToSeed(Seed &seed)
    {
        // Ensure CRC table is ready even if Start() hasn't been called.
        crc64_init();

        if (seed.Dim() != dim_)
            throw std::runtime_error("Replay seed dim mismatch");

        // If no file, nothing to do.
        struct stat st;
        if (::stat(wal_path_.c_str(), &st) != 0)
        {
            if (errno == ENOENT)
                return 0;
            ThrowSys("stat WAL failed");
        }
        if (!S_ISREG(st.st_mode))
            return 0;

        int rfd = ::open(wal_path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (rfd < 0)
            ThrowSys("open WAL for replay failed: " + wal_path_);

        Lsn last_lsn = 0;
        off_t good_offset = 0;
        size_t total_replayed = 0;

        while (true)
        {
            // 1) Read payload header: lsn (8) + count (4) + dim (2)
            uint8_t head_buf[sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t)];
            if (!ReadExact(rfd, head_buf, sizeof(head_buf)))
            {
                // EOF or truncated read -> stop
                break;
            }

            // Decode little-endian
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

            // sanity check dim
            if (dim16 != dim_)
            {
                std::cerr << "[WAL] Error: dimension mismatch in record. Stopping replay.\n";
                break;
            }

            // compute bytes to read for rest of payload (ids + vectors)
            const std::size_t rest_payload_bytes = static_cast<std::size_t>(count) * (sizeof(uint64_t) + dim_ * sizeof(float));

            // read remaining payload
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

            // read footer (payload_size u32 + reserved u32 + crc u64 + magic u64) = 4+4+8+8 = 24 bytes
            uint8_t footer_buf[24];
            if (!ReadExact(rfd, footer_buf, sizeof(footer_buf)))
            {
                std::cerr << "[WAL] Error: missing footer at end of record. Truncation suspected. Stopping.\n";
                break;
            }

            // parse footer
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

            // Reconstruct payload bytes to compute CRC: header bytes + payload_rest
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

            // verify magic
            if (magic != FOOTER_MAGIC)
            {
                std::cerr << "[WAL] Error: footer magic mismatch. Stopping replay.\n";
                break;
            }

            // verify checksum
            uint64_t calc = crc64(0, whole_payload.data(), whole_payload.size());
            if (calc != crc)
            {
                std::cerr << "[WAL] Error: CRC mismatch. Stopping replay.\n";
                break;
            }

            // All good -> deserialize batch and apply
            // parse payload_rest to reconstruct UpsertRequest vector
            std::vector<UpsertRequest> batch;
            batch.reserve(count);

            const uint8_t *qptr = whole_payload.data();
            // skip lsn,u32,count,u16, we've already parsed them (but they are in little-endian)
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
                // copy float bytes
                std::memcpy(r.vec.data.data(), qptr, dim_ * sizeof(float));
                qptr += dim_ * sizeof(float);
                batch.push_back(std::move(r));
            }

            // Apply into seed
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

            // compute good offset: we can compute by ftell relative to start
            off_t cur = lseek(rfd, 0, SEEK_CUR);
            if (cur == -1)
                good_offset = -1;
            else
                good_offset = cur;
        }

        std::cout << "[WAL] Replay finished. Records applied: " << total_replayed << ", Last LSN: " << last_lsn << "\n";

        ::close(rfd);

        // Attempt to truncate file to last good offset (best-effort).
        if (good_offset > 0)
        {
            int wfd = ::open(wal_path_.c_str(), O_WRONLY | O_CLOEXEC);
            if (wfd >= 0)
            {
                if (ftruncate(wfd, good_offset) != 0)
                {
                    std::cerr << "[WAL] Warning: ftruncate failed: " << std::strerror(errno) << "\n";
                }
                ::close(wfd);
            }
        }

        // Ensure next_lsn_ is larger than last_lsn
        Lsn want = last_lsn + 1;
        Lsn cur = next_lsn_.load(std::memory_order_relaxed);
        if (want > cur)
            next_lsn_.store(want, std::memory_order_relaxed);

        return last_lsn;
    }

} // namespace pomai