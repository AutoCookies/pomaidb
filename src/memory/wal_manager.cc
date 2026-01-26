#include "src/memory/wal_manager.h"

extern "C"
{
#include "src/external/crc64.h"
}

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <utility>
#include <filesystem>

namespace pomai::memory
{
    static constexpr char WAL_MAGIC[8] = {'P', 'O', 'M', 'A', 'I', 'W', 'A', 'L'};
    static constexpr uint32_t WAL_RECORD_MAGIC = 0xCAFEBABE;

    static std::once_flag g_crc64_init_flag;

    static void ensure_crc64_init()
    {
        std::call_once(g_crc64_init_flag, []()
                       { crc64_init(); });
    }

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
        uint32_t magic;
        uint32_t rec_len;
        uint64_t checksum;
        uint64_t seq_no;
        uint16_t rec_type;
        uint16_t flags;
    };
#pragma pack(pop)

    static_assert(sizeof(WalFileHeader) == WalManager::WAL_FILE_HEADER_SIZE, "WalFileHeader size mismatch");

    WalManager::WalManager() noexcept
    {
        ensure_crc64_init();
    }

    WalManager::~WalManager()
    {
        close();
    }

    bool WalManager::open(const std::string &path, const WalConfig &cfg)
    {
        std::lock_guard<std::mutex> lk(append_mu_);
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }

        path_ = path;
        cfg_ = cfg;

        try
        {
            std::filesystem::path p(path_);
            std::filesystem::path parent = p.parent_path();
            if (!parent.empty() && !std::filesystem::exists(parent))
            {
                std::filesystem::create_directories(parent);
            }
        }
        catch (...)
        {
        }

        fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd_ < 0)
        {
            std::cerr << "[WAL] Open failed: " << strerror(errno) << "\n";
            return false;
        }

        struct stat st;
        if (fstat(fd_, &st) != 0)
        {
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

        if (::lseek(fd_, 0, SEEK_END) < 0)
        {
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        flush_running_ = true;
        flush_thread_ = std::thread(&WalManager::flush_worker_loop, this);
        return true;
    }

    void WalManager::close()
    {
        flush_running_ = false;
        flush_cv_.notify_all();
        if (flush_thread_.joinable())
            flush_thread_.join();

        std::lock_guard<std::mutex> lk(append_mu_);
        if (fd_ >= 0)
        {
            robust_fsync(fd_);
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool WalManager::write_file_header_if_missing()
    {
        WalFileHeader h;
        std::memset(&h, 0, sizeof(h));
        std::memcpy(h.magic, WAL_MAGIC, 8);
        h.version = WAL_VERSION;
        h.header_size = sizeof(WalFileHeader);
        if (::write(fd_, &h, sizeof(h)) != sizeof(h))
            return false;
        return robust_fsync(fd_);
    }

    bool WalManager::read_file_header_and_validate()
    {
        if (::lseek(fd_, 0, SEEK_SET) < 0)
            return false;
        WalFileHeader h;
        if (::read(fd_, &h, sizeof(h)) != sizeof(h))
            return false;

        if (std::memcmp(h.magic, WAL_MAGIC, 8) != 0)
            return false;
        if (h.version != WAL_VERSION)
        {
            std::cerr << "[WAL] Version mismatch. Expected " << WAL_VERSION << ", got " << h.version << "\n";
            return false;
        }
        return true;
    }

    uint64_t WalManager::append(uint16_t type, const void *data, size_t len, uint16_t flags)
    {
        if (fd_ < 0)
            return 0;

        std::lock_guard<std::mutex> lk(append_mu_);

        WalRecordHeader rh;
        rh.magic = WAL_RECORD_MAGIC;
        rh.rec_len = static_cast<uint32_t>(len);
        rh.rec_type = type;
        rh.flags = flags;
        rh.seq_no = ++seq_no_;
        rh.checksum = crc64(0, static_cast<const unsigned char *>(data), len);

        struct iovec iov[2];
        iov[0].iov_base = &rh;
        iov[0].iov_len = sizeof(rh);
        iov[1].iov_base = const_cast<void *>(data);
        iov[1].iov_len = len;

        ssize_t written = ::writev(fd_, iov, 2);
        if (written != static_cast<ssize_t>(sizeof(rh) + len))
        {
            std::cerr << "[WAL] Append failed: " << strerror(errno) << "\n";
            return 0;
        }

        uint64_t w = static_cast<uint64_t>(written);
        total_bytes_written_.fetch_add(w, std::memory_order_relaxed);
        total_records_written_.fetch_add(1, std::memory_order_relaxed);
        bytes_since_last_fsync_.fetch_add(w, std::memory_order_relaxed);

        if (bytes_since_last_fsync_.load(std::memory_order_relaxed) >= cfg_.batch_commit_size)
        {
            flush_cv_.notify_one();
        }

        return rh.seq_no;
    }

    bool WalManager::truncate_to_zero()
    {
        std::lock_guard<std::mutex> lk(append_mu_);
        if (fd_ < 0)
            return false;

        if (::ftruncate(fd_, 0) != 0)
            return false;

        bytes_since_last_fsync_.store(0);
        seq_no_.store(0);
        total_bytes_written_.store(0);
        total_records_written_.store(0);

        if (::lseek(fd_, 0, SEEK_SET) < 0)
            return false;
        return write_file_header_if_missing();
    }

    void WalManager::flush_worker_loop()
    {
        while (flush_running_)
        {
            std::unique_lock<std::mutex> lk(append_mu_);
            flush_cv_.wait_for(lk, std::chrono::milliseconds(10), [this]
                               { return !flush_running_ || bytes_since_last_fsync_.load() >= cfg_.batch_commit_size; });

            if (bytes_since_last_fsync_.load() > 0 && fd_ >= 0)
            {
                bytes_since_last_fsync_.store(0);
                if (!robust_fsync(fd_))
                    std::cerr << "[WAL] Fsync failed\n";
            }
        }
    }

    bool WalManager::robust_fsync(int fd)
    {
        int ret;
        do
        {
            ret = ::fsync(fd);
        } while (ret == -1 && errno == EINTR);
        return ret == 0;
    }

    void WalManager::recover(const std::function<void(uint64_t seq, uint16_t type, const std::vector<uint8_t> &data)> &cb)
    {
        std::lock_guard<std::mutex> lk(append_mu_);
        if (fd_ < 0)
            return;

        ::lseek(fd_, sizeof(WalFileHeader), SEEK_SET);

        std::vector<uint8_t> buf;
        while (true)
        {
            WalRecordHeader rh;
            ssize_t n = ::read(fd_, &rh, sizeof(rh));
            if (n == 0)
                break;
            if (n != sizeof(rh))
            {
                std::cerr << "[WAL] Corrupt header at offset " << lseek(fd_, 0, SEEK_CUR) << "\n";
                break;
            }

            if (rh.magic != WAL_RECORD_MAGIC)
            {
                std::cerr << "[WAL] Invalid magic. Stop recovery.\n";
                break;
            }

            buf.resize(rh.rec_len);
            if (rh.rec_len > 0)
            {
                n = ::read(fd_, buf.data(), rh.rec_len);
                if (n != static_cast<ssize_t>(rh.rec_len))
                {
                    std::cerr << "[WAL] Truncated payload. Stop recovery.\n";
                    break;
                }

                uint64_t calc = crc64(0, buf.data(), rh.rec_len);
                if (calc != rh.checksum)
                {
                    std::cerr << "[WAL] Checksum mismatch! Seq=" << rh.seq_no << ". Stop recovery.\n";
                    break;
                }
            }
            else
            {
                if (rh.checksum != 0)
                {
                    std::cerr << "[WAL] Checksum mismatch for empty payload.\n";
                    break;
                }
            }

            if (rh.seq_no > seq_no_)
                seq_no_ = rh.seq_no;
            cb(rh.seq_no, rh.rec_type, buf);
        }
        ::lseek(fd_, 0, SEEK_END);
    }
}