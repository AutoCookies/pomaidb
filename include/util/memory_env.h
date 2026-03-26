// util/memory_env.h — In-memory Env implementation (declaration only).
// For testing; no disk I/O.

#pragma once

#include "pomai/env.h"

#include <unordered_map>

namespace pomai {

class InMemoryEnv : public Env {
 public:
  InMemoryEnv() = default;
  ~InMemoryEnv() override = default;

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

 private:
  std::unordered_map<std::string, std::string> files_;
};

}  // namespace pomai
