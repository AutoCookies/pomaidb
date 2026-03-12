// pomai/env.h — Virtual File System (VFS) abstraction for PomaiDB.
//
// OS-agnostic interfaces for file and environment operations. Enables builds
// on non-POSIX targets (custom edge OSes, bare-metal, bootloaders). No
// <unistd.h>, <fcntl.h>, <sys/mman.h>, or <windows.h> in this header.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include "pomai/slice.h"
#include "pomai/status.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace pomai {

// =============================================================================
// File handle abstractions (implementation-owned buffers where noted)
// =============================================================================

/**
 * SequentialFile: Streaming read-only access.
 * Result of Read() may point to an internal buffer; valid until next Read,
 * Skip, or Close.
 */
class SequentialFile {
 public:
  virtual ~SequentialFile() = default;
  virtual Status Read(size_t n, Slice* result) = 0;
  virtual Status Skip(uint64_t n) = 0;
  virtual Status Close() = 0;
};

/**
 * RandomAccessFile: Position-based read-only access.
 * Result of Read() may point to an internal buffer; valid until next Read
 * or Close.
 */
class RandomAccessFile {
 public:
  virtual ~RandomAccessFile() = default;
  virtual Status Read(uint64_t offset, size_t n, Slice* result) = 0;
  virtual Status Close() = 0;
};

/**
 * WritableFile: Append-only write access with explicit flush/sync.
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
 * FileMapping: Read-only view of a file (e.g. mmap or in-memory buffer).
 * On platforms without virtual memory, implementations may use internal
 * buffers or streaming; Data()/Size() remain valid until unmapping or
 * destruction.
 */
class FileMapping {
 public:
  virtual ~FileMapping() = default;
  virtual const void* Data() const = 0;
  virtual size_t Size() const = 0;
};

// =============================================================================
// Environment: factory and filesystem operations
// =============================================================================

class Env {
 public:
  virtual ~Env() = default;

  // File creation
  virtual Status NewSequentialFile(const std::string& path,
                                   std::unique_ptr<SequentialFile>* result) = 0;
  virtual Status NewRandomAccessFile(const std::string& path,
                                    std::unique_ptr<RandomAccessFile>* result) = 0;
  virtual Status NewWritableFile(const std::string& path,
                                 std::unique_ptr<WritableFile>* result) = 0;
  /** Open existing file for append (e.g. WAL); create if missing. */
  virtual Status NewAppendableFile(const std::string& path,
                                  std::unique_ptr<WritableFile>* result) = 0;
  /** Optional: may return NotSupported on platforms without virtual memory. */
  virtual Status NewFileMapping(const std::string& path,
                                std::unique_ptr<FileMapping>* result) = 0;

  // Filesystem operations
  virtual Status FileExists(const std::string& path) = 0;
  virtual Status GetFileSize(const std::string& path, uint64_t* size) = 0;
  virtual Status DeleteFile(const std::string& path) = 0;
  virtual Status RenameFile(const std::string& src, const std::string& dst) = 0;
  virtual Status CreateDirIfMissing(const std::string& path) = 0;
  /** Sync directory metadata to durable storage (e.g. fsync(dir_fd)). */
  virtual Status SyncDir(const std::string& path) = 0;

  /** Process-wide default environment (e.g. PosixEnv on POSIX builds). */
  static Env* Default();
  /** In-memory environment for tests; no disk I/O. */
  static std::unique_ptr<Env> NewInMemory();
};

}  // namespace pomai
