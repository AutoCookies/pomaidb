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

/*
  Production-ready WAL (append-only, checksummed, replay-safe).

  AppendUpserts now supports an optional per-append synchronous durability flag.
  ReplayToSeed returns WalReplayStats describing the replay operation.
*/

namespace pomai
{

  struct WalReplayStats
  {
    Lsn last_lsn{0};                // last applied LSN
    std::size_t records_applied{0}; // number of WAL records (append batches) applied
    std::size_t vectors_applied{0}; // number of vectors applied into seed
    off_t truncated_bytes{0};       // bytes truncated off the WAL (0 if no truncation)
  };

  class Wal
  {
  public:
    Wal(std::string shard_name, std::string wal_dir, std::size_t dim);
    ~Wal();

    Wal(const Wal &) = delete;
    Wal &operator=(const Wal &) = delete;

    void Start();
    void Stop();

    // AppendUpserts writes the batch to the WAL and returns the assigned LSN.
    // If wait_durable==true, AppendUpserts will synchronously fdatasync the WAL
    // before returning and update durable_lsn_ so the append is durable on return.
    Lsn AppendUpserts(const std::vector<UpsertRequest> &batch, bool wait_durable = false);

    // Wait until the given LSN is durable (fsynced). This may perform an fdatasync
    // if the background thread hasn't yet advanced durable_lsn_.
    void WaitDurable(Lsn lsn);

    // Replay WAL into the provided seed and return replay statistics.
    WalReplayStats ReplayToSeed(class Seed &seed);

  private:
    void OpenOrCreateForAppend();
    void CloseFd();
    void FsyncLoop();

    static void WriteAll(int fd, const void *buf, std::size_t n);
    static bool ReadExact(int fd, void *buf, std::size_t n);

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

} // namespace pomai