#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "types.h"

namespace pomai {

class Wal {
 public:
  Wal(std::string shard_name, std::string wal_dir, std::size_t dim);
  ~Wal();

  Wal(const Wal&) = delete;
  Wal& operator=(const Wal&) = delete;

  void Start();
  void Stop();

  Lsn AppendUpserts(const std::vector<UpsertRequest>& batch);

  void WaitDurable(Lsn lsn);

  Lsn ReplayToSeed(class Seed& seed);

 private:
  void OpenOrCreateForAppend();
  void CloseFd();
  void FsyncLoop();

  void WriteRecordLocked(Lsn lsn, const std::vector<UpsertRequest>& batch);

  std::string shard_name_;
  std::string wal_dir_;
  std::string wal_path_;
  std::size_t dim_{0};

  int fd_{-1};

  std::atomic<Lsn> next_lsn_{1};

  std::mutex mu_;
  std::condition_variable cv_;
  std::thread fsync_th_;
  std::atomic<bool> running_{false};

  // protected by mu_
  Lsn written_lsn_{0};
  Lsn durable_lsn_{0};
};

}  // namespace pomai
