/*
 * src/memory/wal_manager.cc
 *
 * Implementation of WalManager.
 *
 * Notes:
 * - Uses POSIX file descriptor I/O (open/write/pread/ftruncate/fsync) for precise control.
 * - Implements crc32 for record validation.
 * - Replay is robust to partial trailing records: it truncates WAL to last valid offset.
 * - Append/fsync/replay/truncate are serialized in-process using append_mu_ to make
 * append_record thread-safe for multiple threads in the same process.
 * - [FIXED] Added Magic Marker per record to detect corruption/partial writes accurately.
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

namespace pomai::memory
{

    // On-disk file header layout helper values
    static constexpr char WAL_MAGIC[8] = {'P', 'O', 'M', 'A', 'I', 'W', 'A', 'L'};
    static constexpr uint32_t WAL_RECORD_MAGIC = 0xCAFEBABE; // [NEW] Magic per record

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
        uint32_t magic;   // [NEW] 4 bytes magic to identify valid start of record
        uint32_t rec_len; // payload length
        uint16_t rec_type;
        uint16_t flags;
        uint64_t seq_no;
    };
#pragma pack(pop)

    static_assert(sizeof(WalFileHeader) == WalManager::WAL_FILE_HEADER_SIZE, "WalFileHeader size mismatch");
    // [UPDATED] Size tăng từ 16 lên 20 bytes do thêm magic (uint32_t)
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

    bool WalManager::open(const std::string &path, bool create_if_missing, const WalConfig &cfg)
    {
        if (fd_ != -1)
        {
            // already open
            return true;
        }

        cfg_ = cfg;
        path_ = path;

        int flags = O_RDWR;
        if (create_if_missing)
            flags |= O_CREAT;
        // use 0600 mode
        fd_ = ::open(path_.c_str(), flags, 0600);
        if (fd_ < 0)
        {
            std::cerr << "WalManager::open: open(" << path_ << ") failed: " << strerror(errno) << "\n";
            return false;
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

        // Initialize counters and seq_no_ by scanning the WAL for the last sequence number.
        // We call replay(...) which will be protected by append_mu_ to ensure no concurrent append.
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

        // write at offset 0
        if (lseek(fd_, 0, SEEK_SET) < 0)
        {
            std::cerr << "WalManager::write_file_header_if_missing: lseek failed: " << strerror(errno) << "\n";
            return false;
        }

        ssize_t w = write(fd_, &h, sizeof(h));
        if (w != static_cast<ssize_t>(sizeof(h)))
        {
            std::cerr << "WalManager::write_file_header_if_missing: write failed: " << strerror(errno) << "\n";
            return false;
        }

        if (cfg_.sync_on_append)
        {
            if (::fsync(fd_) != 0)
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

        ssize_t r = read(fd_, &h, sizeof(h));
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
            std::cerr << "WalManager::read_file_header_and_validate: version mismatch (got " << h.version << " expected " << WAL_VERSION << ")\n";
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

        // Build header and buffer. We'll serialize actual write under append_mu_.
        WalRecordHeader rh{};
        rh.magic = WAL_RECORD_MAGIC; // [FIX] Set Magic
        rh.rec_len = payload_len;
        rh.rec_type = type;
        rh.flags = 0;

        size_t rec_size = sizeof(WalRecordHeader) + payload_len + sizeof(uint32_t);
        std::vector<uint8_t> buf;
        buf.resize(rec_size);

        // We will fill the header (including seq) inside the critical section to avoid races.
        {
            std::lock_guard<std::mutex> lk(append_mu_);

            // Acquire a fresh seq number
            uint64_t seq = seq_no_.fetch_add(1) + 1;
            rh.seq_no = seq;

            // Copy header + payload into buffer
            std::memcpy(buf.data(), &rh, sizeof(WalRecordHeader));
            if (payload_len > 0 && payload)
                std::memcpy(buf.data() + sizeof(WalRecordHeader), payload, payload_len);

            // Compute crc over header + payload (including seq)
            uint32_t c = crc32(buf.data(), sizeof(WalRecordHeader) + payload_len);
            std::memcpy(buf.data() + sizeof(WalRecordHeader) + payload_len, &c, sizeof(c));

            // Seek to end and write (serialized by mutex)
            if (lseek(fd_, 0, SEEK_END) < 0)
            {
                std::cerr << "WalManager::append_record: lseek(SEEK_END) failed: " << strerror(errno) << "\n";
                return std::nullopt;
            }
            ssize_t w = write(fd_, buf.data(), buf.size());
            if (w != static_cast<ssize_t>(buf.size()))
            {
                std::cerr << "WalManager::append_record: write failed: " << strerror(errno) << "\n";
                return std::nullopt;
            }

            if (cfg_.sync_on_append)
            {
                if (::fsync(fd_) != 0)
                {
                    std::cerr << "WalManager::append_record: fsync failed: " << strerror(errno) << "\n";
                    return std::nullopt;
                }
            }

            total_bytes_written_.fetch_add(static_cast<uint64_t>(buf.size()));
            total_records_written_.fetch_add(1);

            // Return the sequence number we assigned
            return rh.seq_no;
        }
    }

    bool WalManager::fsync_log()
    {
        if (fd_ < 0)
            return false;
        std::lock_guard<std::mutex> lk(append_mu_);
        if (::fsync(fd_) != 0)
        {
            std::cerr << "WalManager::fsync_log: fsync failed: " << strerror(errno) << "\n";
            return false;
        }
        return true;
    }

    bool WalManager::replay(const std::function<bool(uint16_t, const void *, uint32_t, uint64_t)> &apply_cb)
    {
        if (fd_ < 0)
            return false;

        // Prevent concurrent appends/truncate while replaying
        std::lock_guard<std::mutex> lk(append_mu_);

        // Read entire file sequentially
        if (lseek(fd_, 0, SEEK_SET) < 0)
        {
            std::cerr << "WalManager::replay: lseek failed: " << strerror(errno) << "\n";
            return false;
        }

        // Determine file size
        struct stat st;
        if (fstat(fd_, &st) != 0)
        {
            std::cerr << "WalManager::replay: fstat failed: " << strerror(errno) << "\n";
            return false;
        }
        off_t file_size = st.st_size;

        off_t read_off = 0;
        // skip file header
        if (file_size < static_cast<off_t>(sizeof(WalFileHeader)))
            return true; // nothing to replay
        read_off += sizeof(WalFileHeader);

        // keep offset of last good record end; start at header
        off_t last_good_end = read_off;

        while (read_off + static_cast<off_t>(sizeof(WalRecordHeader)) <= file_size)
        {
            // read record header
            WalRecordHeader rh;
            ssize_t rr = pread(fd_, &rh, sizeof(rh), read_off);
            if (rr != static_cast<ssize_t>(sizeof(rh)))
            {
                // partial header -> truncate
                break;
            }

            // [FIX] Validate Magic Marker
            if (rh.magic != WAL_RECORD_MAGIC)
            {
                std::cerr << "[WAL] Corruption detected (Bad Magic at offset " << read_off << "). Truncating...\n";
                break; // Stop replay and truncate here
            }

            // boundary checks
            off_t payload_off = read_off + static_cast<off_t>(sizeof(WalRecordHeader));
            off_t crc_off = payload_off + static_cast<off_t>(rh.rec_len);
            off_t end_off = crc_off + static_cast<off_t>(sizeof(uint32_t));

            if (crc_off < 0 || end_off < 0 || end_off > file_size)
            {
                // partial/truncated payload -> break and truncate
                break;
            }

            // read payload
            std::vector<uint8_t> payload;
            if (rh.rec_len > 0)
            {
                payload.resize(rh.rec_len);
                ssize_t rp = pread(fd_, payload.data(), rh.rec_len, payload_off);
                if (rp != static_cast<ssize_t>(rh.rec_len))
                {
                    // truncated
                    break;
                }
            }

            // read crc
            uint32_t stored_crc = 0;
            ssize_t rc = pread(fd_, &stored_crc, sizeof(stored_crc), crc_off);
            if (rc != static_cast<ssize_t>(sizeof(stored_crc)))
            {
                // truncated
                break;
            }

            // validate crc
            // compute crc over header + payload
            std::vector<uint8_t> tmp;
            tmp.resize(sizeof(WalRecordHeader) + rh.rec_len);
            std::memcpy(tmp.data(), &rh, sizeof(WalRecordHeader));
            if (rh.rec_len > 0 && !payload.empty())
                std::memcpy(tmp.data() + sizeof(WalRecordHeader), payload.data(), rh.rec_len);
            uint32_t computed = crc32(tmp.data(), tmp.size());
            if (computed != stored_crc)
            {
                std::cerr << "WalManager::replay: crc mismatch at offset " << read_off << " (stored=" << stored_crc << " computed=" << computed << "), truncating\n";
                // corrupted trailing record -> truncate it
                break;
            }

            // Call apply callback
            bool cont = apply_cb(rh.rec_type, (rh.rec_len > 0) ? payload.data() : nullptr, rh.rec_len, rh.seq_no);
            if (!cont)
            {
                // callback requested abort; leave WAL as-is
                return true;
            }

            last_good_end = end_off;
            read_off = end_off;
        }

        // Truncate file to last_good_end (if not already)
        if (last_good_end < file_size)
        {
            if (ftruncate(fd_, last_good_end) != 0)
            {
                std::cerr << "WalManager::replay: ftruncate failed: " << strerror(errno) << "\n";
                return false;
            }
            // ensure truncation durable
            if (::fsync(fd_) != 0)
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
        // Recreate header: write header at offset 0
        if (!write_file_header_if_missing())
            return false;

        // reset seq counter
        seq_no_.store(0);

        return true;
    }

} // namespace pomai::memory