#pragma once
#include <memory>
#include <span>
#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai
{
    struct SearchResult
    {
        std::span<const SearchHit> hits;
    };

    class DB
    {
    public:
        virtual ~DB() = default;

        virtual Status Put(VectorId id, std::span<const float> vec) = 0;
        virtual Status Delete(VectorId id) = 0;

        virtual Status Search(std::span<const float> query, std::uint32_t topk, SearchResult *out) = 0;

        virtual Status Flush() = 0;
        virtual Status Close() = 0;

        static Status Open(const DBOptions &options, std::unique_ptr<DB> *out);
    };

} // namespace pomai
