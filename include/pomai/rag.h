#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "metadata.h"
#include "types.h"

namespace pomai
{
    using ChunkId = std::uint64_t;
    using DocId = std::uint64_t;
    using TokenId = std::uint32_t;
    using TokenBlob = std::vector<TokenId>;
    using TokenOffsets = std::vector<std::uint32_t>;

    struct RagChunk
    {
        ChunkId chunk_id = 0;
        DocId doc_id = 0;
        TokenBlob tokens; // REQUIRED
        std::optional<VectorView> vec; // OPTIONAL; caller retains ownership of vec memory
        TokenOffsets offsets; // OPTIONAL
        Metadata meta;
    };

    struct RagQuery
    {
        std::span<const TokenId> tokens;
        std::optional<VectorView> vec;
        std::uint32_t topk = 10;
    };

    struct RagSearchOptions
    {
        std::uint32_t candidate_budget = 200;
        std::uint32_t token_budget = 0; // 0 => no budget
        bool enable_vector_rerank = true;
    };

    struct RagSearchExplain
    {
        std::uint32_t shards_visited = 0;
        std::uint32_t lexical_candidates = 0;
        std::uint32_t vector_candidates = 0;
        std::uint64_t lexical_time_us = 0;
        std::uint64_t vector_time_us = 0;
        std::uint64_t total_time_us = 0;
        std::string prune_reason;
    };

    struct RagSearchHit
    {
        ChunkId chunk_id = 0;
        DocId doc_id = 0;
        float score = 0.0f;
        std::uint32_t token_matches = 0;
        TokenOffsets offsets;
    };

    struct RagSearchResult
    {
        std::vector<RagSearchHit> hits;
        RagSearchExplain explain;

        void Clear()
        {
            hits.clear();
            explain = RagSearchExplain{};
        }
    };
} // namespace pomai
