#ifndef WAL_H
#define WAL_H

#include <pomai/core/types.h>
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

#ifdef __linux__
#include <pomai/storage/wal_uring.h>
#endif

namespace pomai
{

  using Lsn = std::uint64_t;

  constexpr std::size_t MAX_WAL_PAYLOAD_BYTES = 64 * 1024 * 1024;
  constexpr std::size_t MAX_BATCH_ROWS = 50'000;
  inline constexpr std::uint64_t FOOTER_MAGIC = UINT64_C(0x706f6d616977616c);

  struct WalReplayStats
  {
    Lsn last_lsn{0};
    std::size_t records_applied{0};
    std::size_t vectors_applied{0};
    off_t truncated_bytes{0};
  };

  struct WalOptions
  {
    std::size_t max_queued_bytes{64 * 1024 * 1024};
    std::size_t max_pending_records{4096};
    std::size_t max_batch_bytes{4 * 1024 * 1024};
    std::size_t max_iovecs{128};
    unsigned uring_entries{256};
    std::size_t max_inflight_batches{128};
    bool enable_sqpoll{false};
    bool prefer_uring{true};
  };

  class Wal
  {
  public:
    Wal(std::string shard_name, std::string wal_dir, std::size_t dim, WalOptions options = {});
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

    struct Pending
    {
      Lsn lsn;
      std::shared_ptr<std::vector<uint8_t>> buf;
      std::size_t bytes;
    };

  private:
    // Durability: an LSN is durable once fdatasync completes after its write submission.
    // On WAL creation, fsync the directory once; on truncate/rotate, fsync the directory
    // to persist metadata updates.
    static void ThrowSys(const std::string &what);
    static void EnsureDirExists(const std::string &dir);
    static void FsyncDir(const std::string &dir);

    void OpenOrCreateForAppend();
    void CloseFd();
    void WriterLoop(); // writer thread: drain pending buffers and flush
    void CompletionLoop();
    void SubmitBatchUring(std::deque<Pending> &&batch);
    void SubmitBatchSync(std::deque<Pending> &&batch);

    static void WriteAll(int fd, const void *buf, std::size_t n);
    static bool ReadExact(int fd, void *buf, std::size_t n);

    std::string shard_name_;
    std::string wal_dir_;
    std::string wal_path_;
    std::size_t dim_{0};
    WalOptions options_{};

    int fd_{-1};

    std::atomic<Lsn> next_lsn_{1};

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::thread writer_th_;
    std::thread completion_th_;
    std::atomic<bool> running_{false};
    bool stop_requested_{false};
    std::atomic<bool> wal_error_{false};
    std::string wal_error_msg_;

    // protected by mu_
    Lsn written_lsn_{0};
    Lsn durable_lsn_{0};
    std::size_t pending_bytes_{0};
    std::size_t inflight_batches_{0};

    std::deque<Pending> pending_writes_;

    struct InflightBatch
    {
      Lsn start_lsn{0};
      Lsn end_lsn{0};
      bool write_done{false};
      bool sync_done{false};
      std::vector<std::shared_ptr<std::vector<uint8_t>>> buffers;
#ifdef __linux__
      std::vector<iovec> iovecs;
#endif
    };

    std::deque<std::unique_ptr<InflightBatch>> inflight_;

#ifdef __linux__
    WalUring uring_;
    bool uring_enabled_{false};
#else
    bool uring_enabled_{false};
#endif
  };

} // namespace pomai

#endif // WAL_H
