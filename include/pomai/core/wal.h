#pragma once
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
        GroupCommit = 2, // (Chưa implement) Gom nhiều write rồi fsync 1 lần
    };

    class WalWriter final
    {
    public:
        WalWriter() = default;

        // Không cho phép copy để tránh tranh chấp file descriptor
        WalWriter(const WalWriter &) = delete;
        WalWriter &operator=(const WalWriter &) = delete;

        pomai::Status Open(const std::filesystem::path &path, FsyncPolicy policy);

        // Zero-copy interface: Nhận pointer raw và size
        pomai::Status Append(const void *data, std::size_t size);

        pomai::Status Flush();
        void Close();

        const LatencyWindow &FsyncLatencyWindow() const { return fsync_lat_us_; }

    private:
        std::filesystem::path path_;
        FsyncPolicy policy_{FsyncPolicy::Never};
        int fd_{-1};

        LatencyWindow fsync_lat_us_{2048};
    };

    class WalReader final
    {
    public:
        explicit WalReader(std::filesystem::path path);
        ~WalReader();

        // Mở file để đọc
        pomai::Status Open();

        // Đọc record tiếp theo.
        // - Trả về OK: Đọc thành công, data nằm trong `out`.
        // - Trả về OUT_OF_RANGE: Hết file (EOF clean).
        // - Trả về DATA_LOSS: Checksum sai hoặc file bị cắt cụt bất thường.
        pomai::Status ReadNext(std::vector<std::byte> &out);

    private:
        std::filesystem::path path_;
        int fd_{-1};
    };

} // namespace pomai::core