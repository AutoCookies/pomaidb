/*
 * src/memory/wal_manager.cc
 *
 * Implementation of WalManager.
 * [UPDATED] Uses centralized configuration via Dependency Injection.
 * [IMPROVED]
 *  - Robust write/read helpers that handle partial writes and EINTR.
 *  - Use O_CLOEXEC and set FD_CLOEXEC on open.
 *  - Batched fsync support: when WalConfig.batch_commit_size > 0, fsync is
 *    performed only after accumulated bytes reach the threshold.
 *  - Safer reopen / truncation handling and more logging.
 */

#include "src/memory/wal_manager.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <mutex>
#include <atomic>

namespace pomai::memory
{

    // On-disk file header layout helper values
    static constexpr char WAL_MAGIC[8] = {'P', 'O', 'M', 'A', 'I', 'W', 'A', 'L'};
    static constexpr uint32_t WAL_RECORD_MAGIC = 0xCAFEBABE;

#pragma pack(push, 1)
    struct WalFileHeader
    {
        char magic[8];
        uint32_t version;
        uint32_t header_size;
        uint8_t reserved[16];
    };

    struct WalRecordHeader
    {
        uint32_t magic;   // 4 bytes magic to identify valid start of record
        uint32_t rec_len; // payload length
        uint16_t rec_type;
        uint16_t flags;
        uint64_t seq_no;
    };
#pragma pack(pop)

    static_assert(sizeof(WalFileHeader) == WalManager::WAL_FILE_HEADER_SIZE, "WalFileHeader size mismatch");
    static_assert(sizeof(WalRecordHeader) == 20, "WalRecordHeader expected 20 bytes");

    // CRC32 implementation (table)
    static uint32_t crc32_table[256];
    static bool crc32_table_init = false;

    static void init_crc32_table()
    {
        if (crc32_table_init)
            return;
        const uint32_t poly = 0xEDB88320u;
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t c = i;
            for (size_t j = 0; j < 8; ++j)
            {
                if (c & 1)
                    c = poly ^ (c >> 1);
                else
                    c = c >> 1;
            }
            crc32_table[i] = c;
        }
        crc32_table_init = true;
    }

    uint32_t WalManager::crc32(const uint8_t *buf, size_t len)
    {
        init_crc32_table();
        uint32_t c = 0xFFFFFFFFu;
        for (size_t i = 0; i < len; ++i)
        {
            c = crc32_table[(c ^ buf[i]) & 0xFFu] ^ (c >> 8);
        }
        return c ^ 0xFFFFFFFFu;
    }

    WalManager::WalManager() noexcept = default;

    WalManager::~WalManager()
    {
        close();
    }

    // NOTE:
    // bytes_since_last_fsync is a translation-unit-global accumulator used to
    // implement batch fsync semantics. It's intentionally local to this TU
    // because WalManager header cannot be changed here. In practice there is
    // typically a single WAL per process, so this is acceptable.
    static std::atomic<uint64_t> bytes_since_last_fsync{0};

    // Helper: robust write that handles partial writes and EINTR.
    static bool write_all(int fd, const uint8_t *buf, size_t len)
    {
        size_t written = 0;
        while (written < len)
        {
            ssize_t w = ::write(fd, buf + written, len - written);
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    continue; // should not happen for blocking fd, but be tolerant
                return false;
            }
            written += static_cast<size_t>(w);
        }
        return true;
    }

    // Helper: robust pwrite (used rarely) - loop on EINTR and partial writes.
    static bool pwrite_all(int fd, const uint8_t *buf, size_t len, off_t offset)
    {
        size_t written = 0;
        while (written < len)
        {
            ssize_t w = ::pwrite(fd, buf + written, len - written, offset + static_cast<off_t>(written));
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                return false;
            }
            written += static_cast<size_t>(w);
        }
        return true;
    }

    // Helper: robust fsync that retries on EINTR.
    static bool robust_fsync(int fd)
    {
        while (true)
        {
            if (::fsync(fd) == 0)
                return true;
            if (errno == EINTR)
                continue;
            return false;
        }
    }

    bool WalManager::open(const std::string &path, bool create_if_missing, const WalManager::WalConfig &cfg)
    {
        if (fd_ != -1)
        {
            // already open
            return true;
        }

        cfg_ = cfg; // [UPDATED] Store the injected config
        path_ = path;

        int flags = O_RDWR;
        if (create_if_missing)
            flags |= O_CREAT;
#ifdef O_CLOEXEC
        flags |= O_CLOEXEC;
#endif

        // use 0600 mode
        fd_ = ::open(path_.c_str(), flags, 0600);
        if (fd_ < 0)
        {
            std::cerr << "WalManager::open: open(" << path_ << ") failed: " << strerror(errno) << "\n";
            return false;
        }

        // Ensure FD_CLOEXEC is set (portable guard)
        int fd_flags = fcntl(fd_, F_GETFD);
        if (fd_flags != -1)
        {
            fcntl(fd_, F_SETFD, fd_flags | FD_CLOEXEC);
        }

        // Check file header presence / validity
        struct stat st;
        if (fstat(fd_, &st) != 0)
        {
            std::cerr << "WalManager::open: fstat failed: " << strerror(errno) << "\n";
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        if (st.st_size == 0)
        {
            if (!write_file_header_if_missing())
            {
                ::close(fd_);
                fd_ = -1;
                return false;
            }
        }
        else
        {
            if (!read_file_header_and_validate())
            {
                ::close(fd_);
                fd_ = -1;
                return false;
            }
        }

        // Initialize counters
        uint64_t max_seq = 0;
        auto cb = [&max_seq](uint16_t /*type*/, const void * /*payload*/, uint32_t /*len*/, uint64_t seq) -> bool
        {
            if (seq > max_seq)
                max_seq = seq;
            return true; // continue
        };

        if (!replay(cb))
        {
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        seq_no_.store(max_seq);
        total_bytes_written_.store(0);
        total_records_written_.store(0);
        bytes_since_last_fsync.store(0);

        return true;
    }

    void WalManager::close()
    {
        std::lock_guard<std::mutex> lk(append_mu_);
        if (fd_ != -1)
        {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool WalManager::write_file_header_if_missing()
    {
        if (fd_ < 0)
            return false;

        WalFileHeader h{};
        std::memcpy(h.magic, WAL_MAGIC, sizeof(h.magic));
        h.version = WAL_VERSION;
        h.header_size = static_cast<uint32_t>(sizeof(WalFileHeader));
        std::fill(std::begin(h.reserved), std::end(h.reserved), 0);

        if (lseek(fd_, 0, SEEK_SET) < 0)
        {
            std::cerr << "WalManager::write_file_header_if_missing: lseek failed: " << strerror(errno) << "\n";
            return false;
        }

        if (!write_all(fd_, reinterpret_cast<const uint8_t *>(&h), sizeof(h)))
        {
            std::cerr << "WalManager::write_file_header_if_missing: write failed: " << strerror(errno) << "\n";
            return false;
        }

        // [UPDATED] Use centralized config
        if (cfg_.sync_on_append)
        {
            if (!robust_fsync(fd_))
            {
                std::cerr << "WalManager::write_file_header_if_missing: fsync failed: " << strerror(errno) << "\n";
                return false;
            }
        }

        return true;
    }

    bool WalManager::read_file_header_and_validate()
    {
        if (fd_ < 0)
            return false;

        WalFileHeader h{};
        if (lseek(fd_, 0, SEEK_SET) < 0)
        {
            std::cerr << "WalManager::read_file_header_and_validate: lseek failed: " << strerror(errno) << "\n";
            return false;
        }

        ssize_t r = ::read(fd_, &h, sizeof(h));
        if (r != static_cast<ssize_t>(sizeof(h)))
        {
            std::cerr << "WalManager::read_file_header_and_validate: read header failed: " << strerror(errno) << "\n";
            return false;
        }

        if (std::memcmp(h.magic, WAL_MAGIC, sizeof(h.magic)) != 0)
        {
            std::cerr << "WalManager::read_file_header_and_validate: magic mismatch\n";
            return false;
        }
        if (h.version != WAL_VERSION)
        {
            std::cerr << "WalManager::read_file_header_and_validate: version mismatch\n";
            return false;
        }
        if (h.header_size != sizeof(WalFileHeader))
        {
            std::cerr << "WalManager::read_file_header_and_validate: header_size mismatch\n";
            return false;
        }
        return true;
    }

    std::optional<uint64_t> WalManager::append_record(uint16_t type, const void *payload, uint32_t payload_len)
    {
        if (fd_ < 0)
            return std::nullopt;

        WalRecordHeader rh{};
        rh.magic = WAL_RECORD_MAGIC;
        rh.rec_len = payload_len;
        rh.rec_type = type;
        rh.flags = 0;

        size_t rec_size = sizeof(WalRecordHeader) + payload_len + sizeof(uint32_t);
        std::vector<uint8_t> buf;
        buf.resize(rec_size);

        {
            std::lock_guard<std::mutex> lk(append_mu_);

            uint64_t seq = seq_no_.fetch_add(1) + 1;
            rh.seq_no = seq;

            std::memcpy(buf.data(), &rh, sizeof(WalRecordHeader));
            if (payload_len > 0 && payload)
                std::memcpy(buf.data() + sizeof(WalRecordHeader), payload, payload_len);

            uint32_t c = crc32(buf.data(), sizeof(WalRecordHeader) + payload_len);
            std::memcpy(buf.data() + sizeof(WalRecordHeader) + payload_len, &c, sizeof(c));

            if (lseek(fd_, 0, SEEK_END) < 0)
            {
                std::cerr << "WalManager::append_record: lseek failed: " << strerror(errno) << "\n";
                return std::nullopt;
            }

            if (!write_all(fd_, buf.data(), buf.size()))
            {
                std::cerr << "WalManager::append_record: write failed: " << strerror(errno) << "\n";
                return std::nullopt;
            }

            // update stats
            total_bytes_written_.fetch_add(static_cast<uint64_t>(buf.size()), std::memory_order_relaxed);
            total_records_written_.fetch_add(1, std::memory_order_relaxed);

            // Batched fsync logic:
            // - if batch_commit_size == 0: follow sync_on_append boolean.
            // - if batch_commit_size > 0: accumulate bytes and fsync when threshold reached.
            if (cfg_.batch_commit_size > 0)
            {
                uint64_t acc = bytes_since_last_fsync.fetch_add(static_cast<uint64_t>(buf.size()), std::memory_order_relaxed) + static_cast<uint64_t>(buf.size());
                if (acc >= cfg_.batch_commit_size)
                {
                    if (!robust_fsync(fd_))
                    {
                        std::cerr << "WalManager::append_record: fsync failed after batch threshold: " << strerror(errno) << "\n";
                        return std::nullopt;
                    }
                    bytes_since_last_fsync.store(0, std::memory_order_relaxed);
                }
            }
            else if (cfg_.sync_on_append)
            {
                if (!robust_fsync(fd_))
                {
                    std::cerr << "WalManager::append_record: fsync failed: " << strerror(errno) << "\n";
                    return std::nullopt;
                }
            }

            return rh.seq_no;
        }
    }

    bool WalManager::fsync_log()
    {
        if (fd_ < 0)
            return false;
        std::lock_guard<std::mutex> lk(append_mu_);
        if (!robust_fsync(fd_))
        {
            std::cerr << "WalManager::fsync_log: fsync failed: " << strerror(errno) << "\n";
            return false;
        }
        // reset accumulator on explicit fsync
        bytes_since_last_fsync.store(0, std::memory_order_relaxed);
        return true;
    }

    bool WalManager::replay(const std::function<bool(uint16_t, const void *, uint32_t, uint64_t)> &apply_cb)
    {
        if (fd_ < 0)
            return false;

        std::lock_guard<std::mutex> lk(append_mu_);

        if (lseek(fd_, 0, SEEK_SET) < 0)
        {
            std::cerr << "WalManager::replay: lseek failed: " << strerror(errno) << "\n";
            return false;
        }

        struct stat st;
        if (fstat(fd_, &st) != 0)
        {
            std::cerr << "WalManager::replay: fstat failed: " << strerror(errno) << "\n";
            return false;
        }
        off_t file_size = st.st_size;

        off_t read_off = 0;
        if (file_size < static_cast<off_t>(sizeof(WalFileHeader)))
            return true;
        read_off += sizeof(WalFileHeader);

        off_t last_good_end = read_off;

        while (read_off + static_cast<off_t>(sizeof(WalRecordHeader)) <= file_size)
        {
            WalRecordHeader rh;
            ssize_t rr = pread(fd_, &rh, sizeof(rh), read_off);
            if (rr != static_cast<ssize_t>(sizeof(rh)))
                break;

            if (rh.magic != WAL_RECORD_MAGIC)
            {
                std::cerr << "[WAL] Corruption detected (Bad Magic at offset " << read_off << "). Truncating...\n";
                break;
            }

            off_t payload_off = read_off + static_cast<off_t>(sizeof(WalRecordHeader));
            off_t crc_off = payload_off + static_cast<off_t>(rh.rec_len);
            off_t end_off = crc_off + static_cast<off_t>(sizeof(uint32_t));

            if (crc_off < 0 || end_off < 0 || end_off > file_size)
                break;

            std::vector<uint8_t> payload;
            if (rh.rec_len > 0)
            {
                payload.resize(rh.rec_len);
                ssize_t rp = pread(fd_, payload.data(), rh.rec_len, payload_off);
                if (rp != static_cast<ssize_t>(rh.rec_len))
                    break;
            }

            uint32_t stored_crc = 0;
            ssize_t rc = pread(fd_, &stored_crc, sizeof(stored_crc), crc_off);
            if (rc != static_cast<ssize_t>(sizeof(stored_crc)))
                break;

            std::vector<uint8_t> tmp;
            tmp.resize(sizeof(WalRecordHeader) + rh.rec_len);
            std::memcpy(tmp.data(), &rh, sizeof(WalRecordHeader));
            if (rh.rec_len > 0 && !payload.empty())
                std::memcpy(tmp.data() + sizeof(WalRecordHeader), payload.data(), rh.rec_len);
            uint32_t computed = crc32(tmp.data(), tmp.size());
            if (computed != stored_crc)
            {
                std::cerr << "WalManager::replay: crc mismatch at offset " << read_off << ", truncating\n";
                break;
            }

            bool cont = apply_cb(rh.rec_type, (rh.rec_len > 0) ? payload.data() : nullptr, rh.rec_len, rh.seq_no);
            if (!cont)
                return true;

            last_good_end = end_off;
            read_off = end_off;
        }

        if (last_good_end < file_size)
        {
            if (ftruncate(fd_, last_good_end) != 0)
            {
                std::cerr << "WalManager::replay: ftruncate failed: " << strerror(errno) << "\n";
                return false;
            }
            if (!robust_fsync(fd_))
            {
                std::cerr << "WalManager::replay: fsync after truncate failed: " << strerror(errno) << "\n";
                return false;
            }
        }

        return true;
    }

    bool WalManager::truncate_to_zero()
    {
        std::lock_guard<std::mutex> lk(append_mu_);
        if (fd_ < 0)
            return false;

        if (ftruncate(fd_, 0) != 0)
        {
            std::cerr << "WalManager::truncate_to_zero: ftruncate failed: " << strerror(errno) << "\n";
            return false;
        }
        // Reset accumulator and counters
        bytes_since_last_fsync.store(0, std::memory_order_relaxed);
        seq_no_.store(0);
        total_bytes_written_.store(0);
        total_records_written_.store(0);

        if (!write_file_header_if_missing())
            return false;

        return true;
    }

} // namespace pomai::memory