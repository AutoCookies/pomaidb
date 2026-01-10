/*
 * src/ai/soa_ids_manager.h
 *
 * SoA IDs/offsets manager with atomic update semantics and optional write-ahead log (WAL)
 * for durability.
 *
 * Responsibilities:
 *  - Map an on-disk ids block (uint64_t entries) via MmapFileManager.
 *  - Provide safe, atomic updates to individual uint64_t entries (std::atomic_ref used
 *    where available).
 *  - Optional durable updates via a small append-only WAL. Durable update sequence:
 *      1) append WalEntry{idx,value} to WAL and fsync WAL
 *      2) atomic_store value into mapped ids array
 *      3) msync mapped range (optional durable sync)
 *      4) truncate WAL (or leave applied entries; on startup recovery will replay)
 *  - Recover/Replay WAL on open to ensure consistency after crash.
 *
 * Notes:
 *  - This utility is designed for single-process access (the process that created the
 *    mapping). If multiple processes concurrently modify the same files, extra
 *    coordination is required.
 *  - WAL format is intentionally simple: sequence of fixed-size records of two uint64_t:
 *      struct WalEntry { uint64_t idx; uint64_t value; };
 *    All values are stored in native little-endian host order (consistent within a single host).
 *
 * API:
 *  - bool open(const std::string &ids_path, size_t num_entries, bool create_if_missing)
 *  - const uint64_t *ids_ptr() const noexcept; // direct pointer to mmap'd ids (read-only pointer)
 *  - bool atomic_update(size_t idx, uint64_t value, bool durable = false);
 *  - bool replay_wal_and_truncate(); // invoked on open
 *
 * Crash-recovery semantics:
 *  - On open(), if a WAL exists and contains entries, replay_wal_and_truncate() will
 *    apply all entries (atomic store + msync) then truncate the WAL to zero.
 *
 * Thread-safety:
 *  - atomic_update serializes WAL writes/ftsync/truncate under a mutex so concurrent updates
 *    from multiple threads are safe.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <mutex>
#include <vector>
#include <memory>

#include "src/memory/mmap_file_manager.h"
#include "src/memory/wal_manager.h"

namespace pomai::ai::soa
{

    class SoaIdsManager
    {
    public:
        SoaIdsManager() noexcept;
        ~SoaIdsManager();

        SoaIdsManager(const SoaIdsManager &) = delete;
        SoaIdsManager &operator=(const SoaIdsManager &) = delete;

        SoaIdsManager(SoaIdsManager &&) = delete;
        SoaIdsManager &operator=(SoaIdsManager &&) = delete;

        // Open or create the ids file (num_entries uint64_t values).
        // ids_path: path to ids file
        // wal_path: path to WAL file (if empty, default is ids_path + ".wal")
        // create_if_missing: if true will create and preallocate ids file
        // Returns true on success.
        bool open(const std::string &ids_path, size_t num_entries, bool create_if_missing = true,
                  const std::string &wal_path = std::string());

        // Close files and unmap.
        void close();

        // Pointer to mmap'd ids array (const). Valid until close().
        const uint64_t *ids_ptr() const noexcept;

        // Number of entries
        size_t num_entries() const noexcept;

        // Atomically update entry at index to value.
        // If durable==true, the update is recorded to WAL and msynced before returning.
        // Returns true on success.
        bool atomic_update(size_t idx, uint64_t value, bool durable = false);

        // Replay WAL (apply pending entries) and truncate WAL. Safe to call multiple times.
        // Called automatically on open. Returns true on success.
        bool replay_wal_and_truncate();

        // Optional: enable structured WalManager and use it for replay/append.
        // If called, wal_manager_ will be created and used instead of raw wal_fd_.
        // Returns true on success.
        bool enable_wal_manager(const std::string &wal_path, const pomai::memory::WalConfig &cfg = {});

    private:
        struct WalEntry
        {
            uint64_t idx;
            uint64_t value;
        };

        // internal helpers
        bool ensure_mapped_ids(size_t num_entries, bool create_if_missing);
        bool open_wal_file_if_needed();

        // members
        pomai::memory::MmapFileManager ids_mmap_;
        std::string ids_path_;
        size_t num_entries_ = 0;

        // Existing raw WAL file descriptor & path (preserve legacy behavior)
        int wal_fd_ = -1;
        std::string wal_path_;

        // Optional structured WalManager (new, preferred). When present, it is used
        // for replay and append; the raw wal_fd_ path is kept for compatibility.
        std::unique_ptr<pomai::memory::WalManager> wal_manager_;
        std::mutex wal_mu_; // protects WAL operations (both raw fd and wal_manager)
    };

} // namespace pomai::ai::soa