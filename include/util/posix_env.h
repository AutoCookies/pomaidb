// util/posix_env.h — POSIX-backed Env implementation (declaration only).
// Implementation in src/util/posix_env.cc; no OS headers here.

#pragma once

#include "pomai/env.h"

namespace pomai {

class PosixEnv : public Env {
 public:
  PosixEnv() = default;
  ~PosixEnv() override = default;

  Status NewSequentialFile(const std::string& path,
                          std::unique_ptr<SequentialFile>* result) override;
  Status NewRandomAccessFile(const std::string& path,
                             std::unique_ptr<RandomAccessFile>* result) override;
  Status NewWritableFile(const std::string& path,
                         std::unique_ptr<WritableFile>* result) override;
  Status NewAppendableFile(const std::string& path,
                          std::unique_ptr<WritableFile>* result) override;
  Status NewFileMapping(const std::string& path,
                       std::unique_ptr<FileMapping>* result) override;

  Status FileExists(const std::string& path) override;
  Status GetFileSize(const std::string& path, uint64_t* size) override;
  Status DeleteFile(const std::string& path) override;
  Status RenameFile(const std::string& src, const std::string& dst) override;
  Status CreateDirIfMissing(const std::string& path) override;
  Status SyncDir(const std::string& path) override;
};

}  // namespace pomai
