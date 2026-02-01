#include "pomai/core/wal.h"
#include "pomai/util/crc32c.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace pomai::core
{
    static pomai::Status ErrnoStatus(const char *what)
    {
        return pomai::Status::IO(std::string(what) + ": " + std::strerror(errno));
    }

    std::filesystem::path WalFilePath(const std::filesystem::path &wal_dir, std::uint64_t wal_id)
    {
        return wal_dir / (std::string("wal_") + std::to_string(wal_id) + ".log");
    }

    pomai::Status ListWalFileIds(const std::filesystem::path &wal_dir, std::vector<std::uint64_t> &out_ids)
    {
        out_ids.clear();
        std::error_code ec;
        if (!std::filesystem::exists(wal_dir, ec))
            return pomai::Status::OK();

        for (const auto &ent : std::filesystem::directory_iterator(wal_dir, ec))
        {
            if (ec)
                break;
            if (!ent.is_regular_file())
                continue;

            const auto name = ent.path().filename().string();
            // wal_<id>.log
            if (name.size() < 9) // minimal: wal_0.log
                continue;
            if (name.rfind("wal_", 0) != 0)
                continue;
            if (name.size() < 4 || name.substr(name.size() - 4) != ".log")
                continue;

            const auto mid = name.substr(4, name.size() - 4 - 4);
            if (mid.empty())
                continue;

            std::uint64_t id = 0;
            try
            {
                std::size_t pos = 0;
                id = std::stoull(mid, &pos, 10);
                if (pos != mid.size())
                    continue;
            }
            catch (...)
            {
                continue;
            }
            out_ids.push_back(id);
        }

        std::sort(out_ids.begin(), out_ids.end());
        out_ids.erase(std::unique(out_ids.begin(), out_ids.end()), out_ids.end());
        return pomai::Status::OK();
    }

    pomai::Status WalWriter::Open(const std::filesystem::path &path, FsyncPolicy policy)
    {
        path_ = path;
        policy_ = policy;
        bytes_written_.store(0, std::memory_order_release);

#if defined(__linux__) || defined(__APPLE__)
        fd_ = ::open(path_.c_str(), O_CREAT | O_APPEND | O_WRONLY, 0644);
        if (fd_ < 0)
            return ErrnoStatus("open wal failed");

        // Initialize bytes_written_ with current file size (so scheduling uses real WAL size).
        struct stat st;
        if (::fstat(fd_, &st) == 0 && st.st_size > 0)
        {
            bytes_written_.store(static_cast<std::uint64_t>(st.st_size), std::memory_order_release);
        }
        return pomai::Status::OK();
#else
        return pomai::Status::Internal("WalWriter only implemented for POSIX systems");
#endif
    }

    pomai::Status WalWriter::Append(const void *data, std::size_t size)
    {
#if defined(__linux__) || defined(__APPLE__)
        if (fd_ < 0)
            return pomai::Status::Internal("wal not open");

        const char *ptr = static_cast<const char *>(data);
        std::size_t remain = size;

        while (remain > 0)
        {
            ssize_t written = ::write(fd_, ptr, remain);
            if (written < 0)
            {
                if (errno == EINTR)
                    continue;
                return ErrnoStatus("write wal failed");
            }

            ptr += written;
            remain -= static_cast<std::size_t>(written);
        }

        bytes_written_.fetch_add(static_cast<std::uint64_t>(size), std::memory_order_release);

        if (policy_ == FsyncPolicy::EveryWrite)
            return Flush();

        // GroupCommit reserved.
        return pomai::Status::OK();
#else
        (void)data;
        (void)size;
        return pomai::Status::Internal("not implemented");
#endif
    }

    pomai::Status WalWriter::Flush()
    {
#if defined(__linux__) || defined(__APPLE__)
        if (fd_ < 0)
            return pomai::Status::Internal("wal not open");

        auto t0 = std::chrono::steady_clock::now();

#if defined(__APPLE__)
        if (::fcntl(fd_, F_FULLFSYNC) != 0)
            return ErrnoStatus("fsync wal failed");
#else
        if (::fsync(fd_) != 0)
            return ErrnoStatus("fsync wal failed");
#endif

        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        fsync_lat_us_.Add(static_cast<std::uint64_t>(us));

        return pomai::Status::OK();
#else
        return pomai::Status::Internal("not implemented");
#endif
    }

    void WalWriter::Close()
    {
#if defined(__linux__) || defined(__APPLE__)
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
#endif
    }

    WalReader::WalReader(std::filesystem::path path) : path_(std::move(path)) {}

    WalReader::~WalReader()
    {
#if defined(__linux__) || defined(__APPLE__)
        if (fd_ >= 0)
            ::close(fd_);
#endif
    }

    pomai::Status WalReader::Open()
    {
#if defined(__linux__) || defined(__APPLE__)
        fd_ = ::open(path_.c_str(), O_RDONLY);
        if (fd_ < 0)
        {
            if (errno == ENOENT)
                return pomai::Status::NotFound("wal file not found");
            return ErrnoStatus("open wal for read");
        }
        return pomai::Status::OK();
#else
        return pomai::Status::Internal("posix only");
#endif
    }

    pomai::Status WalReader::ReadNext(std::vector<std::byte> &out)
    {
#if defined(__linux__) || defined(__APPLE__)
        if (fd_ < 0)
            return pomai::Status::Internal("reader not open");

        std::uint32_t payload_len = 0;
        ssize_t n = ::read(fd_, &payload_len, sizeof(payload_len));

        if (n == 0)
            return pomai::Status::NotFound("eof");
        if (n < 0)
            return ErrnoStatus("read len");
        if (n < static_cast<ssize_t>(sizeof(payload_len)))
            return pomai::Status::IO("truncated record length");

        std::uint32_t expected_crc = 0;
        n = ::read(fd_, &expected_crc, sizeof(expected_crc));
        if (n < static_cast<ssize_t>(sizeof(expected_crc)))
            return pomai::Status::IO("truncated record crc");

        out.resize(payload_len);

        std::size_t remaining = payload_len;
        char *ptr = reinterpret_cast<char *>(out.data());
        while (remaining > 0)
        {
            ssize_t r = ::read(fd_, ptr, remaining);
            if (r < 0)
            {
                if (errno == EINTR)
                    continue;
                return ErrnoStatus("read payload");
            }
            if (r == 0)
                return pomai::Status::IO("unexpected eof in payload");

            ptr += r;
            remaining -= static_cast<std::size_t>(r);
        }

        std::uint32_t actual_crc = pomai::util::Crc32c(out.data(), out.size());
        if (actual_crc != expected_crc)
            return pomai::Status::IO("crc mismatch: corruption detected");

        return pomai::Status::OK();
#else
        return pomai::Status::Internal("not implemented");
#endif
    }

} // namespace pomai::core
