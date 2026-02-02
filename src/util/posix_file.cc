#include "util/posix_file.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace pomai::util
{

    static pomai::Status Err(std::string_view what)
    {
        return pomai::Status::IoError(std::string(what) + ": " + std::strerror(errno));
    }

    PosixFile::PosixFile(PosixFile &&other) noexcept : fd_(other.fd_)
    {
        other.fd_ = -1;
    }

    PosixFile &PosixFile::operator=(PosixFile &&other) noexcept
    {
        if (this != &other)
        {
            Close();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    PosixFile::~PosixFile() { (void)Close(); }

    pomai::Status PosixFile::OpenAppend(const std::string &path, PosixFile *out)
    {
        int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
        if (fd < 0)
            return Err("open");
        *out = PosixFile(fd);
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::OpenRead(const std::string &path, PosixFile *out)
    {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            return Err("open");
        *out = PosixFile(fd);
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::CreateTrunc(const std::string &path, PosixFile *out)
    {
        int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
        if (fd < 0)
            return Err("open");
        *out = PosixFile(fd);
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::PWrite(std::uint64_t off, const void *data, std::size_t n)
    {
        const std::uint8_t *p = static_cast<const std::uint8_t *>(data);
        std::size_t done = 0;
        while (done < n)
        {
            ssize_t w = ::pwrite(fd_, p + done, n - done, static_cast<off_t>(off + done));
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                return Err("pwrite");
            }
            done += static_cast<std::size_t>(w);
        }
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::ReadAt(std::uint64_t off, void *data, std::size_t n, std::size_t *out_read)
    {
        std::uint8_t *p = static_cast<std::uint8_t *>(data);
        std::size_t done = 0;
        while (done < n)
        {
            ssize_t r = ::pread(fd_, p + done, n - done, static_cast<off_t>(off + done));
            if (r < 0)
            {
                if (errno == EINTR)
                    continue;
                return Err("pread");
            }
            if (r == 0)
                break;
            done += static_cast<std::size_t>(r);
        }
        *out_read = done;
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::Flush() { return pomai::Status::Ok(); }

    pomai::Status PosixFile::SyncData()
    {
        if (::fdatasync(fd_) != 0)
            return Err("fdatasync");
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::SyncAll()
    {
        if (::fsync(fd_) != 0)
            return Err("fsync");
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::Close()
    {
        if (fd_ >= 0)
        {
            int r;
            do
            {
                r = ::close(fd_);
            } while (r != 0 && errno == EINTR);
            fd_ = -1;
            if (r != 0)
                return Err("close");
        }
        return pomai::Status::Ok();
    }

    pomai::Status FsyncDir(const std::string &dir_path)
    {
        int fd = ::open(dir_path.c_str(), O_DIRECTORY | O_RDONLY);
        if (fd < 0)
            return Err("open dir");
        int r;
        do
        {
            r = ::fsync(fd);
        } while (r != 0 && errno == EINTR);
        int saved = errno;
        ::close(fd);
        errno = saved;
        if (r != 0)
            return Err("fsync dir");
        return pomai::Status::Ok();
    }

} // namespace pomai::util