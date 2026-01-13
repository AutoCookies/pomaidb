#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <algorithm>

namespace pomai::core
{

    struct Tag
    {
        std::string key;
        std::string value;
        std::string to_string() const { return key + ":" + value; }
    };

    class MetadataIndex
    {
    public:
        MetadataIndex() = default;

        // Thêm metadata cho một vector
        void add_tags(uint64_t label, const std::vector<Tag> &tags)
        {
            std::unique_lock<std::shared_mutex> lock(mu_);
            for (const auto &tag : tags)
            {
                // Key format: "category:shoes"
                inverted_index_[tag.to_string()].push_back(label);
            }
        }

        // Lọc: Trả về danh sách Label ID thỏa mãn điều kiện
        // Trả về copy để đảm bảo thread-safety cho người gọi (Orbit) dùng thoải mái
        std::vector<uint64_t> filter(const std::string &key, const std::string &value) const
        {
            std::shared_lock<std::shared_mutex> lock(mu_);
            std::string token = key + ":" + value;

            auto it = inverted_index_.find(token);
            if (it != inverted_index_.end())
            {
                return it->second;
            }
            return {}; // Không tìm thấy
        }

        // Xóa (Optional - Soft delete chỉ cần xóa trong Orbit,
        // Metadata cứ để đó cũng được, vì search ra ID mà Orbit thấy đã xóa thì sẽ bỏ qua)
        // Tuy nhiên, để sạch sẽ thì nên implement lazy cleanup sau.

    private:
        // Cấu trúc: "color:red" -> [1, 5, 10, ...]
        std::unordered_map<std::string, std::vector<uint64_t>> inverted_index_;
        mutable std::shared_mutex mu_;
    };
}