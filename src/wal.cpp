#include "pomai/core/wal.h"
#include "pomai/util/crc32c.h"

#include <cerrno>
#include <cstring>
#include <chrono>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace pomai::core
{
    // Helper để tạo lỗi kèm message từ errno
    static pomai::Status ErrnoStatus(const char *what)
    {
        // strerror không thread-safe tuyệt đối nhưng đủ tốt cho single-threaded shard logic
        return pomai::Status::IO(std::string(what) + ": " + std::strerror(errno));
    }

    pomai::Status WalWriter::Open(const std::filesystem::path &path, FsyncPolicy policy)
    {
        path_ = path;
        policy_ = policy;

#if defined(__linux__) || defined(__APPLE__)
        // O_CREAT: Tạo nếu chưa có
        // O_APPEND: Luôn ghi vào cuối file (atomic trên nhiều OS hiện đại)
        // O_WRONLY: Chỉ ghi
        // 0644: Permission rw-r--r--
        fd_ = ::open(path_.c_str(), O_CREAT | O_APPEND | O_WRONLY, 0644);
        if (fd_ < 0)
        {
            return ErrnoStatus("open wal failed");
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
            // ::write trả về số byte thực sự ghi được
            ssize_t written = ::write(fd_, ptr, remain);

            if (written < 0)
            {
                // EINTR: Ghi bị ngắt bởi signal (rất quan trọng trong hệ thống production)
                if (errno == EINTR)
                {
                    continue;
                }
                return ErrnoStatus("write wal failed");
            }

            ptr += written;
            remain -= static_cast<std::size_t>(written);
        }

        // Chính sách Fsync: EveryWrite (An toàn nhất nhưng chậm nhất)
        if (policy_ == FsyncPolicy::EveryWrite)
        {
            return Flush();
        }

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

        // fsync đẩy dữ liệu từ OS buffer xuống đĩa vật lý
#if defined(__APPLE__)
        // MacOS fsync không đảm bảo flush phần cứng, dùng fcntl
        if (::fcntl(fd_, F_FULLFSYNC) != 0)
            return ErrnoStatus("fsync wal failed");
#else
        // Linux standard
        if (::fsync(fd_) != 0)
            return ErrnoStatus("fsync wal failed");
#endif

        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // Ghi lại latency để monitoring
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

    // --- WAL READER IMPLEMENTATION ---

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
        // O_RDONLY: Chỉ đọc
        fd_ = ::open(path_.c_str(), O_RDONLY);
        if (fd_ < 0)
        {
            if (errno == ENOENT) // File chưa tồn tại -> Coi như rỗng
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

        // 1. Đọc Header (Length: 4B)
        std::uint32_t payload_len = 0;
        ssize_t n = ::read(fd_, &payload_len, sizeof(payload_len));

        if (n == 0)
            return pomai::Status::NotFound("eof"); // Clean EOF
        if (n < 0)
            return ErrnoStatus("read len");
        if (n < static_cast<ssize_t>(sizeof(payload_len)))
            return pomai::Status::IO("truncated record length");

        // 2. Đọc CRC (4B)
        std::uint32_t expected_crc = 0;
        n = ::read(fd_, &expected_crc, sizeof(expected_crc));
        if (n < static_cast<ssize_t>(sizeof(expected_crc)))
            return pomai::Status::IO("truncated record crc");

        // 3. Đọc Payload
        // Resize buffer để nhận dữ liệu
        out.resize(payload_len);

        // Loop read để đảm bảo đọc đủ payload_len byte
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

            // FIX: Ép kiểu sang size_t vì ta đã đảm bảo r >= 0 ở trên
            remaining -= static_cast<std::size_t>(r);
        }

        // 4. Verify CRC (Quan trọng nhất)
        // Tính CRC thực tế của payload vừa đọc và so sánh với CRC lưu trong file
        std::uint32_t actual_crc = pomai::util::Crc32c(out.data(), out.size());

        if (actual_crc != expected_crc)
        {
            // BigTech Standard: Không bao giờ chấp nhận dữ liệu sai.
            return pomai::Status::IO("crc mismatch: corruption detected");
        }

        return pomai::Status::OK();
#else
        return pomai::Status::Internal("not implemented");
#endif
    }

} // namespace pomai::core