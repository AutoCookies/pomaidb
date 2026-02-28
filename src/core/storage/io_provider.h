#pragma once

#include <filesystem>
#include <memory>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::storage {

/**
 * SequentialFile: Optimized for streaming reads (e.g. WAL replay, Scan).
 */
class SequentialFile {
 public:
  virtual ~SequentialFile() = default;
  virtual Status Read(size_t n, Slice* result, char* scratch) = 0;
  virtual Status Skip(uint64_t n) = 0;
};

/**
 * RandomAccessFile: Optimized for point lookups (e.g. Vector Search).
 */
class RandomAccessFile {
 public:
  virtual ~RandomAccessFile() = default;
  virtual Status Read(uint64_t offset, size_t n, Slice* result, char* scratch) const = 0;
  
  // Hint that data will be needed soon (POSIX_FADV_WILLNEED)
  virtual void Prefetch(uint64_t offset, size_t n) const = 0;
};

/**
 * WritableFile: Optimized for append-only writing with explicit sync gates.
 */
class WritableFile {
 public:
  virtual ~WritableFile() = default;
  virtual Status Append(Slice data) = 0;
  virtual Status Flush() = 0;
  virtual Status Sync() = 0;
  virtual Status Close() = 0;
};

/**
 * MemoryMappedFile: Interface for zero-copy file access.
 */
class MemoryMappedFile {
public:
    virtual ~MemoryMappedFile() = default;
    virtual const uint8_t* Data() const = 0;
    virtual size_t Size() const = 0;
};

class PosixMemoryMappedFile : public MemoryMappedFile {
public:
    PosixMemoryMappedFile(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    ~PosixMemoryMappedFile() override {
        if (data_) munmap(const_cast<uint8_t*>(data_), size_);
    }
    const uint8_t* Data() const override { return data_; }
    size_t Size() const override { return size_; }
private:
    const uint8_t* data_;
    size_t size_;
};

/**
 * PosixIOProvider: Lean, zero-copy implementation using modern syscalls.
 */
class PosixIOProvider {
 public:
  static Status NewSequentialFile(const std::filesystem::path& path, std::unique_ptr<SequentialFile>* result);
  static Status NewRandomAccessFile(const std::filesystem::path& path, std::unique_ptr<RandomAccessFile>* result);
  static Status NewWritableFile(const std::filesystem::path& path, std::unique_ptr<WritableFile>* result);
  static Status NewMemoryMappedFile(const std::filesystem::path& path, std::unique_ptr<MemoryMappedFile>* result);
};

// --- In-place implementations for speed & zero-dependency ---

class PosixSequentialFile : public SequentialFile {
 public:
  explicit PosixSequentialFile(int fd) : fd_(fd) {
    ::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
  }
  ~PosixSequentialFile() override { ::close(fd_); }

  Status Read(size_t n, Slice* result, char* scratch) override {
    ssize_t r = ::read(fd_, scratch, n);
    if (r < 0) return Status::IOError("Sequential Read failed");
    *result = Slice(scratch, static_cast<size_t>(r));
    return Status::Ok();
  }

  Status Skip(uint64_t n) override {
    if (::lseek(fd_, static_cast<off_t>(n), SEEK_CUR) == -1) {
      return Status::IOError("Skip failed");
    }
    return Status::Ok();
  }

 private:
  int fd_;
};

class PosixRandomAccessFile : public RandomAccessFile {
 public:
  explicit PosixRandomAccessFile(int fd) : fd_(fd) {}
  ~PosixRandomAccessFile() override { ::close(fd_); }

  Status Read(uint64_t offset, size_t n, Slice* result, char* scratch) const override {
    ssize_t r = ::pread(fd_, scratch, n, static_cast<off_t>(offset));
    if (r < 0) return Status::IOError("Random Read failed");
    *result = Slice(scratch, static_cast<size_t>(r));
    return Status::Ok();
  }

  void Prefetch(uint64_t offset, size_t n) const override {
    ::posix_fadvise(fd_, static_cast<off_t>(offset), static_cast<off_t>(n), POSIX_FADV_WILLNEED);
  }

 private:
  int fd_;
};

class PosixWritableFile : public WritableFile {
 public:
  explicit PosixWritableFile(int fd) : fd_(fd) {}
  ~PosixWritableFile() override { if (fd_ >= 0) ::close(fd_); }

  Status Append(Slice data) override {
    ssize_t w = ::write(fd_, data.data(), data.size());
    if (w < 0) return Status::IOError("Append failed");
    return Status::Ok();
  }

  Status Flush() override { return Status::Ok(); }
  Status Sync() override { return ::fdatasync(fd_) == 0 ? Status::Ok() : Status::IOError("Sync failed"); }
  Status Close() override {
    int res = ::close(fd_);
    fd_ = -1;
    return res == 0 ? Status::Ok() : Status::IOError("Close failed");
  }

 protected:
  int fd_;
};

/**
 * SectorAlignedWritableFile: Pads writes to the hardware sector boundary (4KB).
 * Distilled from SQLite's sector-aligned VFS.
 */
class SectorAlignedWritableFile : public PosixWritableFile {
 public:
  static constexpr size_t SECTOR_SIZE = 4096;
  
  explicit SectorAlignedWritableFile(int fd) : PosixWritableFile(fd), offset_(0) {}

  Status Append(Slice data) override {
    Status s = PosixWritableFile::Append(data);
    if (s.ok()) offset_ += data.size();
    return s;
  }

  /// Pad to sector boundary and sync.
  Status Sync() override {
    size_t padding = SECTOR_SIZE - (offset_ % SECTOR_SIZE);
    if (padding != SECTOR_SIZE) {
      std::string pad_data(padding, '\0');
      Status s = Append(Slice(pad_data));
      if (!s.ok()) return s;
    }
    return PosixWritableFile::Sync();
  }

 private:
  size_t offset_;
};

// --- Factory Implementations ---

inline Status PosixIOProvider::NewSequentialFile(const std::filesystem::path& path, std::unique_ptr<SequentialFile>* result) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<PosixSequentialFile>(fd);
    return Status::Ok();
}

inline Status PosixIOProvider::NewRandomAccessFile(const std::filesystem::path& path, std::unique_ptr<RandomAccessFile>* result) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<PosixRandomAccessFile>(fd);
    return Status::Ok();
}

inline Status PosixIOProvider::NewWritableFile(const std::filesystem::path& path, std::unique_ptr<WritableFile>* result) {
    int fd = ::open(path.c_str(), O_TRUNC | O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return Status::IOError("open failed: " + path.string());
    *result = std::make_unique<PosixWritableFile>(fd);
    return Status::Ok();
}

inline Status PosixIOProvider::NewMemoryMappedFile(const std::filesystem::path& path, std::unique_ptr<MemoryMappedFile>* result) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return Status::IOError("open failed: " + path.string());
    struct stat st;
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      return Status::IOError("fstat failed: " + path.string());
    }
    size_t size = st.st_size;
    void* data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (data == MAP_FAILED) return Status::IOError("mmap failed: " + path.string());
    *result = std::make_unique<PosixMemoryMappedFile>(static_cast<const uint8_t*>(data), size);
    return Status::Ok();
}

} // namespace pomai::storage
