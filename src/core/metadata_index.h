#pragma once
/*
 * src/core/metadata_index.h
 *
 * Inverted index for metadata filtering (tag-based search).
 * [FIXED] Integrated with PomaiConfig for capacity & delimiter control.
 */

#include <string>
#include <vector>
#include <unordered_map>
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

        void add_tags(uint64_t label, const std::vector<Tag> &tags)
        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            for (const auto &t : tags)
            {
                // Sử dụng delimiter từ config (mặc định là "=")
                std::string composite = t.key + cfg_.delimiter + t.val;
                index_[composite].push_back(label);
            }
        }

        std::vector<uint64_t> filter(const std::string &key, const std::string &val) const
        {
            std::shared_lock<std::shared_mutex> lock(mu_);
            std::string composite = key + cfg_.delimiter + val;
            auto it = index_.find(composite);
            if (it != index_.end())
                return it->second;
            return {};
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