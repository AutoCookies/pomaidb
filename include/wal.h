#ifndef WAL_H
#define WAL_H

#include "types.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <memory>

namespace pomai
{

  using Lsn = std::uint64_t;

  constexpr std::size_t MAX_WAL_PAYLOAD_BYTES = 64 * 1024 * 1024;
  constexpr std::size_t MAX_BATCH_ROWS = 50'000;

  struct WalReplayStats
  {
    Lsn last_lsn{0};
    std::size_t records_applied{0};
    std::size_t vectors_applied{0};
    off_t truncated_bytes{0};
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

    Lsn AppendUpserts(const std::vector<UpsertRequest> &batch, bool wait_durable = false);

    void WaitDurable(Lsn lsn);

    WalReplayStats ReplayToSeed(class Seed &seed);

    Lsn WrittenLsn() const;

    void TruncateToZero();

  private:
    void OpenOrCreateForAppend();
    void CloseFd();
    void WriterLoop(); // writer thread: flush pending buffers and fsync
    void FsyncLoop();  // kept for backward-compatible periodic fsync behavior (now writer also writes)

    static void WriteAll(int fd, const void *buf, std::size_t n);
    static bool ReadExact(int fd, void *buf, std::size_t n);

    struct Pending
    {
      Lsn lsn;
      std::vector<uint8_t> buf;
    };

    std::string shard_name_;
    std::string wal_dir_;
    std::string wal_path_;
    std::size_t dim_{0};

    int fd_{-1};

    std::atomic<Lsn> next_lsn_{1};

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::thread writer_th_;
    std::thread fsync_th_;
    std::atomic<bool> running_{false};

    // protected by mu_
    Lsn written_lsn_{0};
    Lsn durable_lsn_{0};

    std::deque<Pending> pending_writes_;
  };

} // namespace pomai

#endif // WAL_H