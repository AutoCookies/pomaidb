#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "src/core/config.h"

namespace pomai::memory
{
    enum WalRecordType : uint16_t
    {
        WAL_REC_IDS_UPDATE = 1,
        WAL_REC_CREATE_MEMBRANCE = 10,
        WAL_REC_DROP_MEMBRANCE = 11,
        WAL_REC_INSERT_BATCH = 20,
        WAL_REC_DELETE_LABEL = 21,
        WAL_REC_CHECKPOINT = 100
    };

    class WalManager
    {
    public:
        using WalConfig = pomai::config::WalConfig;

        WalManager() noexcept;
        ~WalManager();

        WalManager(const WalManager &) = delete;
        WalManager &operator=(const WalManager &) = delete;

        bool open(const std::string &path, const WalConfig &cfg);
        void close();

        uint64_t append(uint16_t type, const void *data, size_t len, uint16_t flags = 0);
        void recover(const std::function<void(uint64_t seq, uint16_t type, const std::vector<uint8_t> &data)> &cb);
        bool truncate_to_zero();

        std::string path() const noexcept { return path_; }
        uint64_t last_seq_no() const noexcept { return seq_no_.load(); }
        uint64_t total_bytes_written() const noexcept { return total_bytes_written_.load(); }
        uint64_t total_records_written() const noexcept { return total_records_written_.load(); }

        static constexpr size_t WAL_FILE_HEADER_SIZE = 32;
        static constexpr uint32_t WAL_VERSION = 2;

    private:
        bool write_file_header_if_missing();
        bool read_file_header_and_validate();
        void flush_worker_loop();
        static bool robust_fsync(int fd);

        std::string path_;
        int fd_ = -1;
        WalConfig cfg_;

        std::atomic<uint64_t> seq_no_{0};
        std::atomic<uint64_t> total_bytes_written_{0};
        std::atomic<uint64_t> total_records_written_{0};
        std::atomic<uint64_t> bytes_since_last_fsync_{0};

        std::mutex append_mu_;
        std::thread flush_thread_;
        std::atomic<bool> flush_running_{false};
        std::condition_variable flush_cv_;
    };
}