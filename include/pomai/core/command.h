#pragma once
#include <future>
#include <variant>
#include <vector>

#include "pomai/types.h"
#include "pomai/status.h"

namespace pomai::core
{
    enum class SearchMode : std::uint8_t
    {
        Latency = 0,
        Quality = 1
    };

    struct SearchRequest
    {
        pomai::VectorData query;
        std::uint32_t topk{10};
        SearchMode mode{SearchMode::Latency};
        std::uint64_t deadline_ns{0};
        std::uint32_t candidate_budget{0};
    };

    struct SearchReply
    {
        pomai::Status status{pomai::Status::OK()};
        std::vector<pomai::SearchHit> hits;
    };

    struct CmdUpsert
    {
        std::vector<pomai::UpsertItem> items;
        std::promise<pomai::Status> prom;
    };

    struct CmdSearch
    {
        SearchRequest req;
        std::promise<SearchReply> prom;
    };

    struct CmdFlush
    {
        std::promise<pomai::Status> prom;
    };

    struct CmdStop
    {
    };

    struct Command
    {
        std::variant<CmdUpsert, CmdSearch, CmdFlush, CmdStop> payload;
    };

} // namespace pomai::core