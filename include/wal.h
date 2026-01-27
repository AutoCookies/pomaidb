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

  On-disk record layout (little-endian, portable fields):
    [payload]
    [footer]

  payload:
    LSN (u64)
    count (u32)
    dim   (u16)
    for each record:
      id  (u64)
      vec (dim * f32)

  footer (fixed size, written right after payload):
    payload_size (u32)    -- number of bytes in payload
    reserved      (u32)    -- reserved for future use (alignment)
    crc64         (u64)    -- CRC64 of payload (not including footer)
    magic         (u64)    -- fixed magic marker to identify footer / integrity

  Notes:
  - Replay scans sequentially by reading payload header (lsn/count/dim) then the
    rest of the payload and then the footer. If any read fails, or checksum/magic
    mismatch, replay stops at the last good record and returns that LSN.
  - Append writes payload+footer together in a single contiguous write to reduce
    syscall window. Durability is provided asynchronously by a background fdatasync
    thread; however WaitDurable() will perform a synchronous fdatasync if required.
  - All multithreading access is protected by mutexes/atomics.
*/

namespace pomai
{

  class Wal
  {
  public:
    Wal(std::string shard_name, std::string wal_dir, std::size_t dim);
    ~Wal();

    Wal(const Wal &) = delete;
    Wal &operator=(const Wal &) = delete;

    // Start background threads and open file for append.
    void Start();
    // Stop background thread, ensure durable, close file.
    void Stop();

    // Append a batch to WAL. Returns LSN assigned (0 if empty).
    // Append writes to the OS (append) but does not guarantee durability unless WaitDurable is used.
    Lsn AppendUpserts(const std::vector<UpsertRequest> &batch);

    // Wait until given LSN is durable on disk. Performs synchronous fdatasync if needed.
    void WaitDurable(Lsn lsn);

    // Replay WAL into the provided seed. Returns last applied LSN.
    // On replay, any trailing partial/corrupt record is ignored and the file may be truncated
    // to the last good offset (attempted, best-effort).
    Lsn ReplayToSeed(class Seed &seed);

  private:
    void OpenOrCreateForAppend();
    void CloseFd();
    void FsyncLoop();

    // low level helpers
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