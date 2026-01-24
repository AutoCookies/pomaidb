#pragma once
#include <vector>
#include <cstdint>
#include <future>
#include <chrono>
#include <optional>
#include <string>

namespace pomai::core
{

    struct ShardSearchResult
    {
        std::vector<std::pair<uint64_t, float>> hits;
    };

    struct Work
    {
        enum class Kind : uint8_t
        {
            INSERT,
            SEARCH,
            STOP
        } kind;

        uint64_t label = 0;
        std::vector<float> vec;
        size_t k = 0;
        std::promise<bool> prom_insert;
        std::promise<ShardSearchResult> prom_search;
        uint64_t req_id = 0;
        std::chrono::steady_clock::time_point enqueue_ts;
        std::string membrance;

        Work() = default;

        static Work make_insert(const std::string &mem, uint64_t lbl, std::vector<float> &&v, uint64_t req = 0)
        {
            Work w;
            w.kind = Kind::INSERT;
            w.label = lbl;
            w.vec = std::move(v);
            w.req_id = req;
            w.enqueue_ts = std::chrono::steady_clock::now();
            w.membrance = mem;
            return w;
        }

        static Work make_search(const std::string &mem, std::vector<float> &&qvec, size_t topk, uint64_t req = 0)
        {
            Work w;
            w.kind = Kind::SEARCH;
            w.vec = std::move(qvec);
            w.k = topk;
            w.req_id = req;
            w.enqueue_ts = std::chrono::steady_clock::now();
            w.membrance = mem;
            return w;
        }

        static Work make_stop()
        {
            Work w;
            w.kind = Kind::STOP;
            w.enqueue_ts = std::chrono::steady_clock::now();
            return w;
        }
    };

    inline uint64_t splitmix64(uint64_t x) noexcept
    {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
    }

    inline size_t shard_for_label(uint64_t label, size_t nshards) noexcept
    {
        if (nshards == 0)
            return 0;
        return static_cast<size_t>(splitmix64(label) % nshards);
    }

} // namespace pomai::core