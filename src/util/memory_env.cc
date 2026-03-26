// util/memory_env.cc — In-memory Env implementation for testing.

#include "util/memory_env.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace pomai {

namespace {

// -----------------------------------------------------------------------------
// In-memory file storage (shared between readers and writers)
// -----------------------------------------------------------------------------
using FileMap = std::unordered_map<std::string, std::string>;

// -----------------------------------------------------------------------------
// InMemorySequentialFile
// -----------------------------------------------------------------------------
class InMemorySequentialFile : public SequentialFile {
 public:
  explicit InMemorySequentialFile(std::string data)
      : data_(std::move(data)), pos_(0) {}
  ~InMemorySequentialFile() override = default;

  Status Read(size_t n, Slice* result) override {
    size_t avail = (pos_ < data_.size()) ? (data_.size() - pos_) : 0;
    size_t to_read = std::min(n, avail);
    if (to_read == 0) {
      *result = Slice();
      return Status::Ok();
    }
    buf_.assign(data_.data() + pos_, data_.data() + pos_ + to_read);
    pos_ += to_read;
    *result = Slice(buf_.data(), buf_.size());
    return Status::Ok();
  }

  Status Skip(uint64_t n) override {
    uint64_t skip = std::min<uint64_t>(n, data_.size() - pos_);
    pos_ += static_cast<size_t>(skip);
    return Status::Ok();
  }

  Status Close() override { return Status::Ok(); }

 private:
  std::string data_;
  size_t pos_ = 0;
  std::string buf_;  // last Read result
};

// -----------------------------------------------------------------------------
// InMemoryRandomAccessFile
// -----------------------------------------------------------------------------
class InMemoryRandomAccessFile : public RandomAccessFile {
 public:
  explicit InMemoryRandomAccessFile(std::string data) : data_(std::move(data)) {}
  ~InMemoryRandomAccessFile() override = default;

  Status Read(uint64_t offset, size_t n, Slice* result) override {
    if (offset >= data_.size()) {
      *result = Slice();
      return Status::Ok();
    }
    size_t avail = data_.size() - static_cast<size_t>(offset);
    size_t to_read = std::min(n, avail);
    buf_.assign(data_.data() + offset, data_.data() + offset + to_read);
    *result = Slice(buf_.data(), buf_.size());
    return Status::Ok();
  }

  Status Close() override { return Status::Ok(); }

 private:
  std::string data_;
  mutable std::string buf_;
};

// -----------------------------------------------------------------------------
// InMemoryWritableFile
// -----------------------------------------------------------------------------
class InMemoryWritableFile : public WritableFile {
 public:
  InMemoryWritableFile(std::string path, FileMap* files, bool append)
      : path_(std::move(path)), files_(files), append_(append) {
    if (append_) {
      auto it = files_->find(path_);
      if (it != files_->end()) content_ = it->second;
    }
  }
  ~InMemoryWritableFile() override { (void)Close(); }

  Status Append(Slice data) override {
    const char* p = reinterpret_cast<const char*>(data.data());
    content_.append(p, data.size());
    bytes_written_ += static_cast<uint64_t>(data.size());
    return Status::Ok();
  }
  uint64_t BytesWritten() const override { return bytes_written_; }

  Status Flush() override { return Status::Ok(); }
  Status Sync() override { return Status::Ok(); }

  Status Close() override {
    if (closed_) return Status::Ok();
    closed_ = true;
    (*files_)[path_] = std::move(content_);
    return Status::Ok();
  }

 private:
  std::string path_;
  FileMap* files_ = nullptr;
  bool append_ = false;
  bool closed_ = false;
  std::string content_;
  uint64_t bytes_written_ = 0;
};

// -----------------------------------------------------------------------------
// InMemoryFileMapping
// -----------------------------------------------------------------------------
class InMemoryFileMapping : public FileMapping {
 public:
  explicit InMemoryFileMapping(std::string data) : data_(std::move(data)) {}
  const void* Data() const override {
    return data_.empty() ? nullptr : data_.data();
  }
  size_t Size() const override { return data_.size(); }

 private:
  std::string data_;
};

}  // namespace

// -----------------------------------------------------------------------------
// InMemoryEnv
// -----------------------------------------------------------------------------
Status InMemoryEnv::NewSequentialFile(const std::string& path,
                                     std::unique_ptr<SequentialFile>* result) {
  auto it = files_.find(path);
  if (it == files_.end()) return Status::NotFound("file does not exist");
  *result = std::make_unique<InMemorySequentialFile>(it->second);
  return Status::Ok();
}

Status InMemoryEnv::NewRandomAccessFile(const std::string& path,
                                        std::unique_ptr<RandomAccessFile>* result) {
  auto it = files_.find(path);
  if (it == files_.end()) return Status::NotFound("file does not exist");
  *result = std::make_unique<InMemoryRandomAccessFile>(it->second);
  return Status::Ok();
}

Status InMemoryEnv::NewWritableFile(const std::string& path,
                                    std::unique_ptr<WritableFile>* result) {
  *result = std::make_unique<InMemoryWritableFile>(path, &files_, false);
  return Status::Ok();
}

Status InMemoryEnv::NewAppendableFile(const std::string& path,
                                     std::unique_ptr<WritableFile>* result) {
  *result = std::make_unique<InMemoryWritableFile>(path, &files_, true);
  return Status::Ok();
}

Status InMemoryEnv::NewFileMapping(const std::string& path,
                                   std::unique_ptr<FileMapping>* result) {
  auto it = files_.find(path);
  if (it == files_.end()) return Status::NotFound("file does not exist");
  *result = std::make_unique<InMemoryFileMapping>(it->second);
  return Status::Ok();
}

Status InMemoryEnv::FileExists(const std::string& path) {
  return files_.count(path) ? Status::Ok() : Status::NotFound("file does not exist");
}

Status InMemoryEnv::GetFileSize(const std::string& path, uint64_t* size) {
  if (!size) return Status::InvalidArgument("size is null");
  auto it = files_.find(path);
  if (it == files_.end()) return Status::NotFound("file does not exist");
  *size = static_cast<uint64_t>(it->second.size());
  return Status::Ok();
}

Status InMemoryEnv::DeleteFile(const std::string& path) {
  files_.erase(path);
  return Status::Ok();
}

Status InMemoryEnv::RenameFile(const std::string& src, const std::string& dst) {
  auto it = files_.find(src);
  if (it == files_.end()) return Status::NotFound("file does not exist");
  std::string content = std::move(it->second);
  files_.erase(it);
  files_[dst] = std::move(content);
  return Status::Ok();
}

Status InMemoryEnv::CreateDirIfMissing(const std::string&) {
  return Status::Ok();
}

Status InMemoryEnv::SyncDir(const std::string&) {
  return Status::Ok();
}

}  // namespace pomai
