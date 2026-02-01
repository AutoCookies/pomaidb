#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "pomai/status.h"
#include "pomai/core/stats.h"

namespace pomai::core
{
    enum class FsyncPolicy : std::uint8_t
    {
        Never = 0,       // OS tự lo (nhanh nhất, rủi ro cao nhất)
        EveryWrite = 1,  // Fsync sau mỗi lần append (chậm, an toàn tuyệt đối)
        GroupCommit = 2, // reserved (chưa implement)
    };

    // Helpers for manifest-based WAL rotation.
    // WAL directory contains files: wal_<id>.log where <id> is uint64.
    std::filesystem::path WalFilePath(const std::filesystem::path &wal_dir, std::uint64_t wal_id);
    pomai::Status ListWalFileIds(const std::filesystem::path &wal_dir, std::vector<std::uint64_t> &out_ids);

    class WalWriter final
    {
    public:
        WalWriter() = default;

        WalWriter(const WalWriter &) = delete;
        WalWriter &operator=(const WalWriter &) = delete;

        pomai::Status Open(const std::filesystem::path &path, FsyncPolicy policy);

        pomai::Status Append(const void *data, std::size_t size);

        pomai::Status Flush();
        void Close();

        std::uint64_t BytesWritten() const { return bytes_written_.load(std::memory_order_acquire); }
        const LatencyWindow &FsyncLatencyWindow() const { return fsync_lat_us_; }

    private:
        std::filesystem::path path_;
        FsyncPolicy policy_{FsyncPolicy::Never};
        int fd_{-1};

        std::atomic<std::uint64_t> bytes_written_{0};
        LatencyWindow fsync_lat_us_{2048};
    };

    class WalReader final
    {
    public:
        explicit WalReader(std::filesystem::path path);
        ~WalReader();

        pomai::Status Open();

        // Trả OK khi đọc được record; NotFound("eof") khi EOF clean.
        pomai::Status ReadNext(std::vector<std::byte> &out);

    private:
        std::filesystem::path path_;
        int fd_{-1};
    };

} // namespace pomai::core
