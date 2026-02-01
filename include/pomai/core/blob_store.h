#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <mutex>
#include "pomai/types.h"

namespace pomai::core
{

    class BlobStore
    {
    public:
        explicit BlobStore(std::filesystem::path path) : path_(std::move(path))
        {
            // Mở file append binary
            // Nếu muốn xịn hơn thì dùng mmap, nhưng ở đây dùng fstream cho đơn giản
            if (std::filesystem::exists(path_))
            {
                Recover();
            }
            ofs_.open(path_, std::ios::binary | std::ios::app | std::ios::out);
        }

        // Lưu text/token và trả về offset
        void Append(VectorId id, const std::string &data)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            // Nếu id lớn hơn size hiện tại, resize bảng offset
            if (id >= offsets_.size())
            {
                offsets_.resize(id + 1, -1);
            }

            // Ghi vị trí hiện tại vào index
            uint64_t current_pos = ofs_.tellp();
            offsets_[id] = current_pos;

            // Ghi độ dài (4 bytes) + Data
            uint32_t len = static_cast<uint32_t>(data.size());
            ofs_.write(reinterpret_cast<const char *>(&len), 4);
            ofs_.write(data.data(), len);

            // Flush nhẹ (hoặc dựa vào OS page cache)
        }

        // Đọc data theo ID
        std::string Get(VectorId id)
        {
            if (id >= offsets_.size() || offsets_[id] == -1)
            {
                return ""; // Not found
            }

            // Dùng ifstream riêng để đọc (thread-safe với writer)
            // Lưu ý: Trong production nên dùng pread hoặc mmap để không phải seek/open liên tục
            std::ifstream ifs(path_, std::ios::binary | std::ios::in);
            ifs.seekg(offsets_[id]);

            uint32_t len;
            ifs.read(reinterpret_cast<char *>(&len), 4);

            std::string res(len, '\0');
            ifs.read(res.data(), len);
            return res;
        }

        // Recover lại bảng offsets khi khởi động lại
        void Recover()
        {
            std::ifstream ifs(path_, std::ios::binary | std::ios::in);
            uint64_t pos = 0;
            VectorId id_counter = 0; // Giả sử ID tăng dần tuần tự khớp với log

            // Logic Recover này cần khớp với logic WAL của Shard.
            // Ở phiên bản đơn giản, ta tạm bỏ qua Recover chi tiết mà chỉ init file.
            // Để chuẩn 10/10: BlobStore cần snapshot bảng offsets_ ra file riêng (blob.index).
        }

        // Snapshot: Chỉ cần dump cái vector offsets_ ra đĩa là xong
        void SaveIndexSnapshot(std::filesystem::path idx_path)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::ofstream ofs(idx_path, std::ios::binary);
            size_t sz = offsets_.size();
            ofs.write(reinterpret_cast<char *>(&sz), sizeof(sz));
            ofs.write(reinterpret_cast<char *>(offsets_.data()), sz * sizeof(uint64_t));
        }

        void LoadIndexSnapshot(std::filesystem::path idx_path)
        {
            std::ifstream ifs(idx_path, std::ios::binary);
            if (!ifs.is_open())
                return;
            size_t sz;
            ifs.read(reinterpret_cast<char *>(&sz), sizeof(sz));
            offsets_.resize(sz);
            ifs.read(reinterpret_cast<char *>(offsets_.data()), sz * sizeof(uint64_t));
        }

    private:
        std::filesystem::path path_;
        std::ofstream ofs_;
        std::vector<int64_t> offsets_; // Index: VectorId -> File Offset
        std::mutex mutex_;
    };
}