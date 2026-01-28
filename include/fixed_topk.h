#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "types.h"

namespace pomai
{

    class FixedTopK
    {
    public:
        struct Node
        {
            float score;
            Id id;
        };

        explicit FixedTopK(std::size_t k)
            : k_(k)
        {
            if (k_ <= kStackCapacity)
            {
                data_ = stack_.data();
            }
            else
            {
                heap_.reset(new Node[k_]);
                data_ = heap_.get();
            }
        }

        std::size_t Size() const noexcept { return size_; }

        void Push(float score, Id id)
        {
            if (k_ == 0)
                return;
            if (size_ < k_)
            {
                data_[size_++] = Node{score, id};
                if (size_ == k_)
                    RebuildMin();
                return;
            }

            if (score <= min_score_)
                return;
            data_[min_index_] = Node{score, id};
            RebuildMin();
        }

        void FillSorted(std::vector<SearchResultItem> &out) const
        {
            out.clear();
            if (size_ == 0)
                return;

            out.reserve(size_);
            for (std::size_t i = 0; i < size_; ++i)
                out.push_back(SearchResultItem{data_[i].id, data_[i].score});

            std::sort(out.begin(), out.end(), [](const SearchResultItem &a, const SearchResultItem &b)
                      { return a.score > b.score; });
        }

    private:
        void RebuildMin()
        {
            min_index_ = 0;
            min_score_ = data_[0].score;
            for (std::size_t i = 1; i < size_; ++i)
            {
                if (data_[i].score < min_score_)
                {
                    min_score_ = data_[i].score;
                    min_index_ = i;
                }
            }
        }

        static constexpr std::size_t kStackCapacity = 64;

        std::size_t k_{0};
        std::size_t size_{0};
        std::size_t min_index_{0};
        float min_score_{0.0f};

        std::array<Node, kStackCapacity> stack_{};
        std::unique_ptr<Node[]> heap_;
        Node *data_{nullptr};
    };

} // namespace pomai
