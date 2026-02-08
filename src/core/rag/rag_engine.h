#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "pomai/options.h"
#include "pomai/rag.h"
#include "pomai/status.h"

namespace pomai::core
{
    class RagEngine
    {
    public:
        RagEngine(pomai::DBOptions opt, pomai::MembraneSpec spec);
        ~RagEngine();

        RagEngine(const RagEngine &) = delete;
        RagEngine &operator=(const RagEngine &) = delete;

        Status Open();
        Status Close();

        Status PutChunk(const pomai::RagChunk& chunk);
        Status Search(const pomai::RagQuery& query,
                      const pomai::RagSearchOptions& opts,
                      pomai::RagSearchResult* out) const;

        const pomai::MembraneSpec& spec() const { return spec_; }

    private:
        struct RagRecord
        {
            pomai::ChunkId chunk_id = 0;
            pomai::DocId doc_id = 0;
            pomai::TokenBlob tokens;
            pomai::TokenOffsets offsets;
            std::vector<float> embedding;
            bool has_vector = false;
            pomai::Metadata meta;
        };

        struct RagShard
        {
            mutable std::mutex mu;
            std::unordered_map<pomai::ChunkId, RagRecord> chunks;
            std::unordered_map<pomai::TokenId, std::vector<pomai::ChunkId>> postings;
        };

        static std::uint32_t ShardOf(pomai::ChunkId id, std::uint32_t shard_count);
        RagShard& ShardFor(pomai::ChunkId id);
        const RagShard& ShardFor(pomai::ChunkId id) const;

        pomai::DBOptions opt_;
        pomai::MembraneSpec spec_;
        bool opened_ = false;
        std::vector<std::unique_ptr<RagShard>> shards_;
    };
} // namespace pomai::core
