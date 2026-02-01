#pragma once
#include <future>
#include <string>
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

        // Step1/2: keep shard search hot-path IO-free.
        // If true, Router may fetch payload after merging final topK (via CmdGetPayloadBatch).
        bool include_payload{false};

        std::uint64_t deadline_ns{0};
        std::uint32_t candidate_budget{0};
    };

    struct SearchReply
    {
        pomai::Status status{pomai::Status::OK()};
        std::vector<pomai::SearchHit> hits;
    };

    struct PayloadBatchReply
    {
        pomai::Status status{pomai::Status::OK()};
        std::vector<std::string> payloads; // aligned with request ids order
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

    // Router -> shard runtime: fetch payload for final topK only.
    struct CmdGetPayloadBatch
    {
        std::vector<pomai::VectorId> ids;
        std::promise<PayloadBatchReply> prom;
    };

    // MaintenanceScheduler -> shard runtime: request checkpoint on shard thread.
    struct CmdCheckpoint
    {
        // Non-blocking by default: runtime may ignore promise if not needed.
        std::promise<pomai::Status> prom;
        bool has_promise{false};
    };

    struct CmdStop
    {
    };

    struct Command
    {
        std::variant<
            CmdUpsert,
            CmdSearch,
            CmdFlush,
            CmdGetPayloadBatch,
            CmdCheckpoint,
            CmdStop>
            payload;
    };

} // namespace pomai::core
