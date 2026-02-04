#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/iterator.h"
#include "util/thread_pool.h"

namespace pomai::core
{

    class Shard;

    class Engine
    {
    public:
        explicit Engine(pomai::DBOptions opt);
        ~Engine();

        Engine(const Engine &) = delete;
        Engine &operator=(const Engine &) = delete;

        Status Open();
        Status Close();

        Status Put(VectorId id, std::span<const float> vec);
        Status Get(VectorId id, std::vector<float> *out);
        Status Exists(VectorId id, bool *exists);
        Status Delete(VectorId id);
        Status Flush();
        Status Freeze();
        Status Compact();
        Status NewIterator(std::unique_ptr<pomai::SnapshotIterator> *out);

        Status Search(std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);

        const pomai::DBOptions &options() const { return opt_; }

    private:
        Status OpenLocked();
        static std::uint32_t ShardOf(VectorId id, std::uint32_t shard_count);

        pomai::DBOptions opt_;
        bool opened_ = false;

        std::vector<std::unique_ptr<Shard>> shards_;
        std::unique_ptr<util::ThreadPool> search_pool_;
    };

} // namespace pomai::core
