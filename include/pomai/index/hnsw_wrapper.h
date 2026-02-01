#pragma once

// Tắt warning của hnswlib nếu cần
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "hnswlib/hnswlib.h"
#pragma GCC diagnostic pop

#include "pomai/types.h"
#include <memory>
#include <mutex>
#include <vector>

namespace pomai::index
{
    class HnswIndex
    {
    public:
        // Config giống như lúc nãy
        struct Config
        {
            std::size_t max_elements{100000}; // Sức chứa tối đa (phải khai báo trước với hnswlib)
            std::size_t m{16};
            std::size_t ef_construction{200};
            std::uint32_t dim{128};
        };

        explicit HnswIndex(Config cfg) : cfg_(cfg)
        {
            // Chọn không gian L2 (Euclidean)
            space_ = std::make_unique<hnswlib::L2Space>(cfg_.dim);

            // Khởi tạo thuật toán
            alg_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                space_.get(),
                cfg_.max_elements,
                cfg_.m,
                cfg_.ef_construction);
        }

        // Insert: Thread-safe nhờ hnswlib tự quản lý lock bên trong (tuy nhiên ta vẫn cần mutex nếu resize)
        void AddPoint(const std::vector<float> &vec, pomai::VectorId id)
        {
            // Lưu ý: hnswlib dùng id kiểu size_t. VectorId của ta là uint64_t (khớp nhau).
            try
            {
                alg_->addPoint(vec.data(), static_cast<hnswlib::labeltype>(id));
            }
            catch (const std::runtime_error &e)
            {
                // Nếu full, hnswlib sẽ ném exception.
                // Ở bản pro, ta phải handle resizeIndex ở đây.
                // Tạm thời log error hoặc throw tiếp.
                throw;
            }
        }

        // Search
        std::vector<std::pair<float, pomai::VectorId>> Search(const std::vector<float> &query, std::size_t k)
        {
            // Kết quả trả về là Priority Queue
            auto result_pq = alg_->searchKnn(query.data(), k);

            std::vector<std::pair<float, pomai::VectorId>> hits;
            hits.reserve(result_pq.size());

            // PQ của hnswlib trả về từ xa nhất -> gần nhất (Top là max distance).
            // Ta cần reverse lại hoặc cứ lấy ra rồi reverse sau.
            while (!result_pq.empty())
            {
                auto item = result_pq.top();
                result_pq.pop();
                hits.push_back({item.first, static_cast<pomai::VectorId>(item.second)});
            }

            // Reverse để được thứ tự Gần -> Xa (Best match first)
            std::reverse(hits.begin(), hits.end());
            return hits;
        }

        // Save/Load index riêng của HNSW
        void Save(const std::string &path)
        {
            alg_->saveIndex(path);
        }

        void Load(const std::string &path)
        {
            alg_->loadIndex(path, space_.get(), cfg_.max_elements);
        }

        std::size_t Size() const { return alg_->cur_element_count; }

    private:
        Config cfg_;
        std::unique_ptr<hnswlib::L2Space> space_;
        std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_;
    };
}