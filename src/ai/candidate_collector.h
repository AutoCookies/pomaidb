/*
 * src/ai/candidate_collector.h
 *
 * CandidateCollector - small fixed-size heap to maintain top-N candidates by score.
 *
 * Purpose
 * -------
 * Provide a lightweight utility used in the search pipeline to keep the best
 * K candidates seen so far. The collector maintains a max-heap of size <= K
 * where the top element is the worst (largest) score among the current best K.
 * On insertion, if the heap is full and the new score is better (smaller for
 * distance), the worst element is popped and the new one inserted.
 *
 * API (clean, documented, thread-unsafe by default):
 *  - CandidateCollector(size_t K)
 *  - void add(size_t id, float score)
 *  - float worst_score() const           // +inf if empty
 *  - size_t size() const
 *  - bool full() const
 *  - std::vector<std::pair<size_t,float>> topk()   // returns ascending-by-score list
 *  - void clear()
 *  - void merge_from(const CandidateCollector &other) // incorporate other's items
 *
 * Notes
 * -----
 * - This implementation assumes lower score == better (i.e., distances). If you
 *   want to maintain top-K by descending scores, invert comparison at call sites.
 * - Not thread-safe. If you need concurrent updates, wrap calls with external mutex
 *   or use the optional ThreadSafeCandidateCollector wrapper (not provided here).
 *
 * Complexity
 * ----------
 * - add(): O(log K) for push/pop
 * - topk(): O(K log K) due to extraction + sorting (but K is small, e.g., 100..1000).
 *
 * Design goal: clean, readable, production-quality comments and API.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <queue>
#include <utility>
#include <limits>
#include <algorithm>

namespace pomai::ai
{

    // CandidateCollector keeps the best (smallest) K candidates by score.
    // Stored pair: (score, id) internally to make comparator simpler.
    class CandidateCollector
    {
    public:
        // Construct collector that keeps up to k best candidates.
        explicit CandidateCollector(size_t k) : k_(k) {}

        // Add a candidate (id, score). If collector is not full, candidate is pushed.
        // If full and score is strictly better (smaller) than current worst, replace worst.
        void add(size_t id, float score)
        {
            if (k_ == 0)
                return;
            if (heap_.size() < k_)
            {
                heap_.emplace(Item{score, id});
                return;
            }
            // If new score is better than worst (heap top), replace
            if (score < heap_.top().score)
            {
                heap_.pop();
                heap_.emplace(Item{score, id});
            }
        }

        // Merge items from another collector into this one.
        // Complexity: O(min(K, other.size()) * log K)
        void merge_from(const CandidateCollector &other)
        {
            // Extract other's elements into a temporary vector (non-destructive)
            std::vector<Item> tmp;
            tmp.reserve(other.heap_.size());
            // std::priority_queue has no iterator; copy by making a local copy
            auto copy = other.heap_;
            while (!copy.empty())
            {
                tmp.push_back(copy.top());
                copy.pop();
            }
            // tmp holds elements in heap order (worst..best). We can add them.
            for (const auto &it : tmp)
                add(it.id, it.score);
        }

        // Return current worst score (largest among kept), or +inf if empty
        float worst_score() const
        {
            if (heap_.empty())
                return std::numeric_limits<float>::infinity();
            return heap_.top().score;
        }

        // Number of candidates currently stored (<= k)
        size_t size() const noexcept { return heap_.size(); }

        // Is collector full (stored exactly k elements)?
        bool full() const noexcept { return heap_.size() >= k_; }

        // Clear all stored candidates
        void clear()
        {
            while (!heap_.empty())
                heap_.pop();
        }

        // Return top-K list sorted ascending by score (best first).
        // The returned vector element is pair<id, score>.
        // This operation destructively extracts the internal heap into a local vector,
        // but does not modify the original heap (we operate on a copy).
        std::vector<std::pair<size_t, float>> topk() const
        {
            std::vector<std::pair<size_t, float>> out;
            out.reserve(heap_.size());
            auto copy = heap_;
            // copy is a max-heap by score (worst at top). Extract into out, then reverse.
            while (!copy.empty())
            {
                const Item &it = copy.top();
                out.emplace_back(it.id, it.score);
                copy.pop();
            }
            std::reverse(out.begin(), out.end()); // now best (smallest score) first
            return out;
        }

    private:
        struct Item
        {
            float score;
            size_t id;
        };

        // comparator for max-heap by score (largest score on top)
        struct Cmp
        {
            bool operator()(Item const &a, Item const &b) const { return a.score < b.score; }
        };

        size_t k_{0};
        std::priority_queue<Item, std::vector<Item>, Cmp> heap_;
    };

} // namespace pomai::ai