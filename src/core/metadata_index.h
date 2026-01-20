#pragma once
/*
 * src/core/metadata_index.h
 *
 * Inverted index for metadata filtering (tag-based search).
 * [FIXED] Integrated with PomaiConfig for capacity & delimiter control.
 * [NEW] Added get_groups() to support Stratified Split by tag key.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <shared_mutex>
#include <algorithm>
#include "src/core/config.h"

namespace pomai::core
{
    struct Tag
    {
        std::string key;
        std::string val;
    };

    class MetadataIndex
    {
    public:
        // Constructor nhận config tập trung
        explicit MetadataIndex(const pomai::config::MetadataConfig &cfg)
            : cfg_(cfg)
        {
            // Cấp phát trước bộ nhớ dựa trên cấu hình để tối ưu Hot Path
            index_.reserve(cfg_.initial_capacity);
        }

        // Thêm tags cho một vector ID
        void add_tags(uint64_t label, const std::vector<Tag> &tags)
        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            for (const auto &t : tags)
            {
                // normalize key (case-insensitive matching for keys)
                std::string key_norm = t.key;
                std::transform(key_norm.begin(), key_norm.end(), key_norm.begin(),
                               [](unsigned char c)
                               { return static_cast<char>(std::tolower(c)); });

                // Optionally normalize value as well to avoid case mismatches in queries.
                std::string val_norm = t.val;
                std::transform(val_norm.begin(), val_norm.end(), val_norm.begin(),
                               [](unsigned char c)
                               { return static_cast<char>(std::tolower(c)); });

                // Sử dụng delimiter từ config (mặc định là "=")
                std::string composite = key_norm + cfg_.delimiter + val_norm;
                index_[composite].push_back(label);
            }
        }

        // Tìm kiếm vector theo tag (Exact Match)
        std::vector<uint64_t> filter(const std::string &key, const std::string &val) const
        {
            std::shared_lock<std::shared_mutex> lock(mu_);
            std::string key_norm = key;
            std::transform(key_norm.begin(), key_norm.end(), key_norm.begin(),
                           [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });
            std::string val_norm = val;
            std::transform(val_norm.begin(), val_norm.end(), val_norm.begin(),
                           [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });

            std::string composite = key_norm + cfg_.delimiter + val_norm;
            auto it = index_.find(composite);
            if (it != index_.end())
                return it->second;
            return {};
        }

        // [NEW] Lấy toàn bộ các nhóm ID dựa trên một key cụ thể (ví dụ: "class")
        // Dùng cho Stratified Split để biết ID nào thuộc class nào.
        // Trả về: Map<TagValue, List<VectorID>>
        // Ví dụ: {"dog": [1, 2], "cat": [3, 4]}
        std::map<std::string, std::vector<uint64_t>> get_groups(const std::string &tag_key) const
        {
            std::shared_lock<std::shared_mutex> lock(mu_);
            std::map<std::string, std::vector<uint64_t>> groups;

            // Normalize the requested key to lowercase to match stored composites
            std::string key_norm = tag_key;
            std::transform(key_norm.begin(), key_norm.end(), key_norm.begin(),
                           [](unsigned char c)
                           { return static_cast<char>(std::tolower(c)); });

            // Prefix cần tìm: "key=" (ví dụ "class=")
            std::string prefix = key_norm + cfg_.delimiter;

            for (const auto &kv : index_)
            {
                const std::string &composite = kv.first;
                // Kiểm tra xem composite key có bắt đầu bằng prefix không
                if (composite.rfind(prefix, 0) == 0)
                {
                    // Tách value: "key=value" -> "value"
                    std::string val = composite.substr(prefix.size());
                    const auto &ids = kv.second;

                    // Copy IDs vào nhóm tương ứng
                    auto &group_list = groups[val];
                    group_list.insert(group_list.end(), ids.begin(), ids.end());
                }
            }
            return groups;
        }

        size_t size() const
        {
            std::shared_lock<std::shared_mutex> lock(mu_);
            return index_.size();
        }

    private:
        pomai::config::MetadataConfig cfg_; // Lưu trữ config cục bộ
        std::unordered_map<std::string, std::vector<uint64_t>> index_;
        mutable std::shared_mutex mu_;
    };
}