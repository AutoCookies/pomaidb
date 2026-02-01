#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "pomai/types.h"

namespace pomai::core
{
    class BlobStore
    {
    public:
        explicit BlobStore(std::filesystem::path path)
            : path_(std::move(path))
        {
            if (std::filesystem::exists(path_))
            {
                Recover();
            }
            ofs_.open(path_, std::ios::binary | std::ios::app | std::ios::out);
        }

        void Append(pomai::VectorId id, const std::string &data)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (id >= offsets_.size())
                offsets_.resize(static_cast<std::size_t>(id) + 1, kNotFound);

            const std::int64_t current_pos = static_cast<std::int64_t>(ofs_.tellp());
            offsets_[static_cast<std::size_t>(id)] = current_pos;

            const std::uint32_t len = static_cast<std::uint32_t>(data.size());
            ofs_.write(reinterpret_cast<const char *>(&len), sizeof(len));
            ofs_.write(data.data(), static_cast<std::streamsize>(len));
        }

        std::string Get(pomai::VectorId id) const
        {
            const std::size_t idx = static_cast<std::size_t>(id);
            if (idx >= offsets_.size() || offsets_[idx] == kNotFound)
                return "";

            std::ifstream ifs(path_, std::ios::binary | std::ios::in);
            if (!ifs.is_open())
                return "";

            ifs.seekg(static_cast<std::streamoff>(offsets_[idx]));

            std::uint32_t len = 0;
            ifs.read(reinterpret_cast<char *>(&len), sizeof(len));
            if (!ifs.good())
                return "";

            std::string res(len, '\0');
            ifs.read(res.data(), static_cast<std::streamsize>(len));
            if (!ifs.good())
                return "";

            return res;
        }

        // (Placeholder) Recover offsets_ if you implement a real on-disk index.
        // In current design, offsets_ is persisted via SaveIndexSnapshot().
        void Recover()
        {
            // Intentionally empty in this simplified design.
            // Production: rebuild offsets_ by scanning file or load snapshot.
        }

        // Snapshot offsets_ -> file (atomicity handled by caller if needed)
        void SaveIndexSnapshot(const std::filesystem::path &idx_path)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            std::ofstream ofs(idx_path, std::ios::binary | std::ios::trunc);
            if (!ofs.is_open())
                return;

            const std::uint64_t sz = static_cast<std::uint64_t>(offsets_.size());
            ofs.write(reinterpret_cast<const char *>(&sz), sizeof(sz));

            if (!offsets_.empty())
            {
                ofs.write(reinterpret_cast<const char *>(offsets_.data()),
                          static_cast<std::streamsize>(offsets_.size() * sizeof(std::int64_t)));
            }
        }

        void LoadIndexSnapshot(const std::filesystem::path &idx_path)
        {
            std::ifstream ifs(idx_path, std::ios::binary | std::ios::in);
            if (!ifs.is_open())
                return;

            std::uint64_t sz = 0;
            ifs.read(reinterpret_cast<char *>(&sz), sizeof(sz));
            if (!ifs.good())
                return;

            offsets_.resize(static_cast<std::size_t>(sz), kNotFound);

            if (sz > 0)
            {
                ifs.read(reinterpret_cast<char *>(offsets_.data()),
                         static_cast<std::streamsize>(offsets_.size() * sizeof(std::int64_t)));
            }
        }

    private:
        static constexpr std::int64_t kNotFound = -1;

        std::filesystem::path path_;
        mutable std::mutex mutex_;

        std::ofstream ofs_;
        std::vector<std::int64_t> offsets_; // VectorId -> file offset
    };

} // namespace pomai::core