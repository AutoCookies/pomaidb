/* src/memory/wal_manager.h */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <atomic>
#include <optional>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "src/core/config.h"

namespace pomai::memory
{
    // Record types
    enum WalRecordType : uint16_t
    {
        WAL_REC_IDS_UPDATE = 1,
        WAL_REC_CREATE_MEMBRANCE = 10,
        WAL_REC_DROP_MEMBRANCE = 11,

        // [THÊM MỚI] Dùng cho vector data batch. Payload khớp với PomaiOrbit.
        WAL_REC_INSERT_BATCH = 20,

        WAL_REC_CHECKPOINT = 100
    };

    class WalManager
    {
    public:
        // Provide nested alias so callers/tests can refer to WalManager::WalConfig
        using WalConfig = pomai::config::WalConfig;

        WalManager() noexcept;
        ~WalManager();

        // Non-copyable
        WalManager(const WalManager &) = delete;
        WalManager &operator=(const WalManager &) = delete;

        // Open/create WAL file at path. If create_if_missing==true, file will be created.
        // Returns true on success.
        bool open(const std::string &path, bool create_if_missing, const WalConfig &cfg);

        // Close the WAL file.
        void close();

        // Append a typed record with payload (payload_len bytes).
        // If sync_on_append is true in config, append will fsync before returning.
        // Returns sequence number (monotonic) on success, or std::nullopt on error.
        std::optional<uint64_t> append_record(uint16_t type, const void *payload, uint32_t payload_len);

        // Explicitly fsync WAL to disk (useful when grouping appends)
        // Returns true on success.
        bool fsync_log();

        // Replay WAL from start and invoke apply_cb for each valid record:
        //   apply_cb(type, payload_ptr, payload_len, seq_no) -> return true to continue, false to abort replay.
        // If replay detects a partial/truncated trailing record it will truncate the WAL
        // to the last valid offset before returning true.
        // Returns true on success, false on fatal error (e.g., I/O error).
        bool replay(const std::function<bool(uint16_t, const void *, uint32_t, uint64_t)> &apply_cb);

        // Truncate WAL file to zero length. Safe to call after successful replay / snapshot.
        bool truncate_to_zero();

        // Path accessors for tests/observability
        std::string path() const noexcept { return path_; }
        uint64_t last_seq_no() const noexcept { return seq_no_.load(); }

        // Basic stats
        uint64_t total_bytes_written() const noexcept { return total_bytes_written_.load(); }
        uint64_t total_records_written() const noexcept { return total_records_written_.load(); }

        // Expose internal header sizes for tests / external tools
        static constexpr size_t WAL_FILE_HEADER_SIZE = 8 + 4 + 4 + 16; // magic + version + header_size + reserved
        static constexpr uint32_t WAL_VERSION = 1;

    private:
        bool write_file_header_if_missing();
        bool read_file_header_and_validate();

        // Compute CRC32 over provided buffer (simple table-driven)
        static uint32_t crc32(const uint8_t *buf, size_t len);

        std::string path_;
        int fd_ = -1;
        WalConfig cfg_;

        std::atomic<uint64_t> seq_no_{0};
        std::atomic<uint64_t> total_bytes_written_{0};
        std::atomic<uint64_t> total_records_written_{0};

        // [SỬA LỖI QUAN TRỌNG] Biến thành viên để tracking batch fsync cho từng instance riêng biệt
        std::atomic<uint64_t> bytes_since_last_fsync_{0};

        // Serialize append/fsync/truncate/replay operations in-process
        std::mutex append_mu_;

        void flush_worker_loop();
        std::thread flush_thread_;
        std::atomic<bool> flush_running_{false};
        std::condition_variable flush_cv_;
    };
}