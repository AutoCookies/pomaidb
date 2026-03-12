// util/posix_env.cc — POSIX-backed Env implementation.
// All OS-specific includes and calls are confined to this file.

#include "util/posix_env.h"

#include <cerrno>
#include <cstring>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
// Windows stubs; full implementation would use CreateFile, ReadFile, etc.
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace pomai {

#if !defined(_WIN32) && !defined(_WIN64)

namespace detail {

static Status ErrnoStatus(const char* op) {
  return Status::IOError(std::string(op) + ": " + std::strerror(errno));
}

// -----------------------------------------------------------------------------
// PosixSequentialFile
// -----------------------------------------------------------------------------
class PosixSequentialFile : public SequentialFile {
 public:
  explicit PosixSequentialFile(int fd) : fd_(fd) {}
  ~PosixSequentialFile() override { (void)Close(); }

  Status Read(size_t n, Slice* result) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    buf_.resize(n);
    ssize_t r = ::read(fd_, buf_.data(), n);
    if (r < 0) {
      if (errno == EINTR) return Read(n, result);
      return ErrnoStatus("read");
    }
    buf_.resize(static_cast<size_t>(r));
    *result = Slice(buf_.data(), buf_.size());
    return Status::Ok();
  }

  Status Skip(uint64_t n) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    if (::lseek(fd_, static_cast<off_t>(n), SEEK_CUR) == static_cast<off_t>(-1))
      return ErrnoStatus("lseek");
    return Status::Ok();
  }

  Status Close() override {
    if (fd_ < 0) return Status::Ok();
    int r;
    do {
      r = ::close(fd_);
    } while (r != 0 && errno == EINTR);
    fd_ = -1;
    return r == 0 ? Status::Ok() : ErrnoStatus("close");
  }

 private:
  int fd_ = -1;
  std::vector<char> buf_;
};

// -----------------------------------------------------------------------------
// PosixRandomAccessFile
// -----------------------------------------------------------------------------
class PosixRandomAccessFile : public RandomAccessFile {
 public:
  explicit PosixRandomAccessFile(int fd) : fd_(fd) {}
  ~PosixRandomAccessFile() override { (void)Close(); }

  Status Read(uint64_t offset, size_t n, Slice* result) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    buf_.resize(n);
    ssize_t r = ::pread(fd_, buf_.data(), n, static_cast<off_t>(offset));
    if (r < 0) {
      if (errno == EINTR) return Read(offset, n, result);
      return ErrnoStatus("pread");
    }
    buf_.resize(static_cast<size_t>(r));
    *result = Slice(buf_.data(), buf_.size());
    return Status::Ok();
  }

  Status Close() override {
    if (fd_ < 0) return Status::Ok();
    int r;
    do {
      r = ::close(fd_);
    } while (r != 0 && errno == EINTR);
    fd_ = -1;
    return r == 0 ? Status::Ok() : ErrnoStatus("close");
  }

 private:
  int fd_ = -1;
  mutable std::vector<char> buf_;
};

// -----------------------------------------------------------------------------
// PosixWritableFile
// -----------------------------------------------------------------------------
class PosixWritableFile : public WritableFile {
 public:
  PosixWritableFile(int fd, uint64_t start_offset)
      : fd_(fd), offset_(start_offset) {}
  ~PosixWritableFile() override { (void)Close(); }

  Status Append(Slice data) override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    const char* p = reinterpret_cast<const char*>(data.data());
    size_t rem = data.size();
    while (rem > 0) {
      ssize_t w = ::pwrite(fd_, p, rem, static_cast<off_t>(offset_));
      if (w < 0) {
        if (errno == EINTR) continue;
        return ErrnoStatus("pwrite");
      }
      p += static_cast<size_t>(w);
      rem -= static_cast<size_t>(w);
      offset_ += static_cast<uint64_t>(w);
    }
    return Status::Ok();
  }

  Status Flush() override { return Status::Ok(); }

  Status Sync() override {
    if (fd_ < 0) return Status::InvalidArgument("file closed");
    if (::fdatasync(fd_) != 0) return ErrnoStatus("fdatasync");
    return Status::Ok();
  }

  Status Close() override {
    if (fd_ < 0) return Status::Ok();
    int r;
    do {
      r = ::close(fd_);
    } while (r != 0 && errno == EINTR);
    fd_ = -1;
    return r == 0 ? Status::Ok() : ErrnoStatus("close");
  }

 private:
  int fd_ = -1;
  uint64_t offset_ = 0;
};

// -----------------------------------------------------------------------------
// PosixFileMapping
// -----------------------------------------------------------------------------
class PosixFileMapping : public FileMapping {
 public:
  PosixFileMapping(void* addr, size_t size) : data_(static_cast<const uint8_t*>(addr)), size_(size) {}
  ~PosixFileMapping() override {
    if (data_ && size_ > 0) {
      ::munmap(const_cast<uint8_t*>(data_), size_);
    }
  }
  const void* Data() const override { return data_; }
  size_t Size() const override { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace detail

// -----------------------------------------------------------------------------
// PosixEnv
// -----------------------------------------------------------------------------
Status PosixEnv::NewSequentialFile(const std::string& path,
                                   std::unique_ptr<SequentialFile>* result) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return detail::ErrnoStatus("open");
  result->reset(new detail::PosixSequentialFile(fd));
  return Status::Ok();
}

Status PosixEnv::NewRandomAccessFile(const std::string& path,
                                     std::unique_ptr<RandomAccessFile>* result) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return detail::ErrnoStatus("open");
  result->reset(new detail::PosixRandomAccessFile(fd));
  return Status::Ok();
}

Status PosixEnv::NewWritableFile(const std::string& path,
                                 std::unique_ptr<WritableFile>* result) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) return detail::ErrnoStatus("open");
  result->reset(new detail::PosixWritableFile(fd, 0));
  return Status::Ok();
}

Status PosixEnv::NewAppendableFile(const std::string& path,
                                  std::unique_ptr<WritableFile>* result) {
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return detail::ErrnoStatus("open");
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return detail::ErrnoStatus("fstat");
  }
  uint64_t start = static_cast<uint64_t>(st.st_size);
  result->reset(new detail::PosixWritableFile(fd, start));
  return Status::Ok();
}

Status PosixEnv::NewFileMapping(const std::string& path,
                                std::unique_ptr<FileMapping>* result) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return detail::ErrnoStatus("open");
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    return detail::ErrnoStatus("fstat");
  }
  size_t size = static_cast<size_t>(st.st_size);
  if (size == 0) {
    ::close(fd);
    result->reset(new detail::PosixFileMapping(nullptr, 0));
    return Status::Ok();
  }
  void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (addr == MAP_FAILED) return detail::ErrnoStatus("mmap");
  result->reset(new detail::PosixFileMapping(addr, size));
  return Status::Ok();
}

Status PosixEnv::FileExists(const std::string& path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) {
    if (errno == ENOENT) return Status::NotFound("file does not exist");
    return detail::ErrnoStatus("stat");
  }
  return Status::Ok();
}

Status PosixEnv::GetFileSize(const std::string& path, uint64_t* size) {
  if (!size) return Status::InvalidArgument("size is null");
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) return detail::ErrnoStatus("stat");
  *size = static_cast<uint64_t>(st.st_size);
  return Status::Ok();
}

Status PosixEnv::DeleteFile(const std::string& path) {
  if (::unlink(path.c_str()) != 0) return detail::ErrnoStatus("unlink");
  return Status::Ok();
}

Status PosixEnv::RenameFile(const std::string& src, const std::string& dst) {
  if (::rename(src.c_str(), dst.c_str()) != 0) return detail::ErrnoStatus("rename");
  return Status::Ok();
}

Status PosixEnv::CreateDirIfMissing(const std::string& path) {
  if (path.empty()) return Status::Ok();
  std::string p;
  for (size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      if (!p.empty()) {
        if (::mkdir(p.c_str(), 0755) != 0 && errno != EEXIST)
          return detail::ErrnoStatus("mkdir");
      }
    }
    if (i < path.size()) p += path[i];
  }
  return Status::Ok();
}

Status PosixEnv::SyncDir(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) return detail::ErrnoStatus("open dir");
  int r;
  do {
    r = ::fsync(fd);
  } while (r != 0 && errno == EINTR);
  int saved = errno;
  ::close(fd);
  if (r != 0) return Status::IOError(std::string("fsync dir: ") + std::strerror(saved));
  return Status::Ok();
}

#else  // Windows

Status PosixEnv::NewSequentialFile(const std::string&, std::unique_ptr<SequentialFile>*) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::NewRandomAccessFile(const std::string&, std::unique_ptr<RandomAccessFile>*) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::NewWritableFile(const std::string&, std::unique_ptr<WritableFile>*) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::NewAppendableFile(const std::string&, std::unique_ptr<WritableFile>*) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::NewFileMapping(const std::string&, std::unique_ptr<FileMapping>*) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::FileExists(const std::string&) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::GetFileSize(const std::string&, uint64_t*) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::DeleteFile(const std::string&) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::RenameFile(const std::string&, const std::string&) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::CreateDirIfMissing(const std::string&) {
  return Status::IOError("PosixEnv not implemented on Windows");
}
Status PosixEnv::SyncDir(const std::string&) {
  return Status::IOError("PosixEnv not implemented on Windows");
}

#endif

}  // namespace pomai
