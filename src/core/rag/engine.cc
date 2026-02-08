#include "core/rag/engine.h"

#include <algorithm>
#include <chrono>
#include <unordered_set>

#include "core/distance.h"

namespace pomai::core
{
    namespace
    {
        struct Candidate
        {
            pomai::ChunkId chunk_id = 0;
            std::uint32_t token_matches = 0;
        };

        bool BetterCandidate(const Candidate& a, const Candidate& b)
        {
            if (a.token_matches != b.token_matches) {
                return a.token_matches > b.token_matches;
            }
            return a.chunk_id < b.chunk_id;
        }
    } // namespace

    RagEngine::RagEngine(pomai::DBOptions opt, pomai::MembraneSpec spec)
        : opt_(std::move(opt)), spec_(std::move(spec))
    {
    }

    RagEngine::~RagEngine() = default;

    std::uint32_t RagEngine::ShardOf(pomai::ChunkId id, std::uint32_t shard_count)
    {
        return shard_count == 0 ? 0u : static_cast<std::uint32_t>(id % shard_count);
    }

    RagEngine::RagShard& RagEngine::ShardFor(pomai::ChunkId id)
    {
        return *shards_[ShardOf(id, static_cast<std::uint32_t>(shards_.size()))];
    }

    const RagEngine::RagShard& RagEngine::ShardFor(pomai::ChunkId id) const
    {
        return *shards_[ShardOf(id, static_cast<std::uint32_t>(shards_.size()))];
    }

    Status RagEngine::Open()
    {
        if (opened_) return Status::Ok();
        if (spec_.kind != pomai::MembraneKind::kRag) {
            return Status::InvalidArgument("RAG engine requires RAG membrane kind");
        }
        if (spec_.shard_count == 0) return Status::InvalidArgument("shard_count must be > 0");
        if (spec_.dim == 0) return Status::InvalidArgument("dim must be > 0");
        shards_.clear();
        shards_.reserve(spec_.shard_count);
        for (std::uint32_t i = 0; i < spec_.shard_count; ++i) {
            shards_.push_back(std::make_unique<RagShard>());
        }
        opened_ = true;
        return Status::Ok();
    }

    Status RagEngine::Close()
    {
        shards_.clear();
        opened_ = false;
        return Status::Ok();
    }

    Status RagEngine::PutChunk(const pomai::RagChunk& chunk)
    {
        if (!opened_) return Status::InvalidArgument("rag engine not opened");
        if (chunk.tokens.empty()) {
            return Status::InvalidArgument("RAG membrane requires token_blob; vector-only payload rejected");
        }
        if (chunk.vec.has_value() && chunk.vec->dim != spec_.dim) {
            return Status::InvalidArgument("embedding dim mismatch");
        }

        auto& shard = ShardFor(chunk.chunk_id);
        std::lock_guard<std::mutex> lock(shard.mu);

        auto existing = shard.chunks.find(chunk.chunk_id);
        if (existing != shard.chunks.end()) {
            for (auto token : existing->second.tokens) {
                auto it = shard.postings.find(token);
                if (it == shard.postings.end()) continue;
                auto& postings = it->second;
                postings.erase(std::remove(postings.begin(), postings.end(), chunk.chunk_id), postings.end());
            }
        }

        RagRecord record;
        record.chunk_id = chunk.chunk_id;
        record.doc_id = chunk.doc_id;
        record.tokens = chunk.tokens;
        record.offsets = chunk.offsets;
        record.meta = chunk.meta;
        if (chunk.vec.has_value()) {
            record.embedding.assign(chunk.vec->data, chunk.vec->data + chunk.vec->dim);
            record.has_vector = true;
        }

        for (auto token : record.tokens) {
            shard.postings[token].push_back(record.chunk_id);
        }
        shard.chunks[record.chunk_id] = std::move(record);

        return Status::Ok();
    }

    Status RagEngine::Search(const pomai::RagQuery& query,
                             const pomai::RagSearchOptions& opts,
                             pomai::RagSearchResult* out) const
    {
        if (!out) return Status::InvalidArgument("out is null");
        out->Clear();
        if (query.tokens.empty() && !query.vec.has_value()) {
            return Status::InvalidArgument("query requires tokens or vector");
        }

        const auto start = std::chrono::steady_clock::now();
        std::unordered_map<pomai::ChunkId, std::uint32_t> match_counts;

        for (const auto& shard_ptr : shards_) {
            const auto& shard = *shard_ptr;
            std::lock_guard<std::mutex> lock(shard.mu);
            for (auto token : query.tokens) {
                auto it = shard.postings.find(token);
                if (it == shard.postings.end()) continue;
                for (auto chunk_id : it->second) {
                    ++match_counts[chunk_id];
                }
            }
        }

        const auto lexical_done = std::chrono::steady_clock::now();

        std::vector<Candidate> candidates;
        candidates.reserve(match_counts.size());
        for (const auto& kv : match_counts) {
            candidates.push_back({kv.first, kv.second});
        }

        std::string prune_reason;
        if (opts.candidate_budget > 0 && candidates.size() > opts.candidate_budget) {
            std::nth_element(candidates.begin(),
                             candidates.begin() + static_cast<std::ptrdiff_t>(opts.candidate_budget),
                             candidates.end(), BetterCandidate);
            candidates.resize(opts.candidate_budget);
            prune_reason = "candidate_budget";
        }

        std::sort(candidates.begin(), candidates.end(), BetterCandidate);

        std::vector<pomai::RagSearchHit> hits;
        hits.reserve(std::min<std::size_t>(query.topk, candidates.size()));

        std::uint64_t token_budget_used = 0;
        std::uint32_t vector_candidates = 0;

        const auto vector_start = std::chrono::steady_clock::now();
        for (const auto& candidate : candidates) {
            if (hits.size() >= query.topk) break;
            const auto& shard = ShardFor(candidate.chunk_id);
            std::lock_guard<std::mutex> lock(shard.mu);
            auto it = shard.chunks.find(candidate.chunk_id);
            if (it == shard.chunks.end()) continue;
            const RagRecord& record = it->second;

            float score = static_cast<float>(candidate.token_matches);
            if (opts.enable_vector_rerank && query.vec.has_value() && record.has_vector) {
                ++vector_candidates;
                score += pomai::core::Dot(query.vec->span(),
                                          std::span<const float>(record.embedding.data(),
                                                                 record.embedding.size()));
            }

            const std::uint64_t token_count = record.tokens.size();
            if (opts.token_budget > 0 && (token_budget_used + token_count) > opts.token_budget) {
                if (prune_reason.empty()) {
                    prune_reason = "token_budget";
                }
                continue;
            }

            pomai::RagSearchHit hit;
            hit.chunk_id = record.chunk_id;
            hit.doc_id = record.doc_id;
            hit.score = score;
            hit.token_matches = candidate.token_matches;
            hit.offsets = record.offsets;
            hits.push_back(std::move(hit));
            token_budget_used += token_count;
        }
        const auto vector_done = std::chrono::steady_clock::now();

        std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.chunk_id < b.chunk_id;
        });

        out->hits = std::move(hits);
        out->explain.shards_visited = static_cast<std::uint32_t>(shards_.size());
        out->explain.lexical_candidates = static_cast<std::uint32_t>(match_counts.size());
        out->explain.vector_candidates = vector_candidates;
        out->explain.lexical_time_us = std::chrono::duration_cast<std::chrono::microseconds>(lexical_done - start).count();
        out->explain.vector_time_us = std::chrono::duration_cast<std::chrono::microseconds>(vector_done - vector_start).count();
        out->explain.total_time_us = std::chrono::duration_cast<std::chrono::microseconds>(vector_done - start).count();
        out->explain.prune_reason = std::move(prune_reason);

        return Status::Ok();
    }
} // namespace pomai::core
