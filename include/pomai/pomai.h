#pragma once
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "metadata.h"
#include "options.h"
#include "rag.h"
#include "search.h"
#include "status.h"
#include "types.h"
#include "snapshot.h"

namespace pomai
{

    class DB
    {
    public:
        virtual ~DB() = default;

        // DB lifetime
        virtual Status Flush() = 0;
        virtual Status Close() = 0;

        // Default membrane (optional semantic; can map to "default")
        virtual Status Put(VectorId id, std::span<const float> vec) = 0;
        virtual Status Put(VectorId id, std::span<const float> vec, const Metadata& meta) = 0; // Added
        virtual Status PutVector(VectorId id, std::span<const float> vec) = 0;
        virtual Status PutVector(VectorId id, std::span<const float> vec, const Metadata& meta) = 0;
        virtual Status PutChunk(const RagChunk& chunk) = 0;

        // ... existing PutBatch ...
        // Batch upsert (5-10x faster than sequential Put for large batches)
        // ids.size() must equal vectors.size()
        // All vectors must have dimension matching DBOptions.dim
        virtual Status PutBatch(const std::vector<VectorId>& ids,
                                const std::vector<std::span<const float>>& vectors) = 0;
        virtual Status Get(VectorId id, std::vector<float> *out) = 0;
        virtual Status Get(VectorId id, std::vector<float> *out, Metadata* out_meta) = 0; // Added
        virtual Status Exists(VectorId id, bool *exists) = 0;
        virtual Status Delete(VectorId id) = 0;
        virtual Status Search(std::span<const float> query, uint32_t topk,
                              SearchResult *out) = 0;

        virtual Status Search(std::span<const float> query, uint32_t topk,
                              const SearchOptions& opts, SearchResult *out) = 0;
        virtual Status SearchVector(std::span<const float> query, uint32_t topk,
                                    SearchResult *out) = 0;
        virtual Status SearchVector(std::span<const float> query, uint32_t topk,
                                    const SearchOptions& opts, SearchResult *out) = 0;
        
        // Batch Search (runs concurrently across multiple queries)
        virtual Status SearchBatch(std::span<const float> queries, uint32_t num_queries, 
                                   uint32_t topk, std::vector<SearchResult>* out) = 0;
        virtual Status SearchBatch(std::span<const float> queries, uint32_t num_queries, 
                                   uint32_t topk, const SearchOptions& opts, std::vector<SearchResult>* out) = 0;
        virtual Status SearchRag(const RagQuery& query, const RagSearchOptions& opts, RagSearchResult *out) = 0;

        // Membrane API
        virtual Status CreateMembrane(const MembraneSpec &spec) = 0;
        virtual Status DropMembrane(std::string_view name) = 0;
        virtual Status OpenMembrane(std::string_view name) = 0;
        virtual Status CloseMembrane(std::string_view name) = 0;
        virtual Status ListMembranes(std::vector<std::string> *out) const = 0;

        virtual Status Put(std::string_view membrane, VectorId id,
                           std::span<const float> vec) = 0;
        virtual Status Put(std::string_view membrane, VectorId id,
                           std::span<const float> vec, const Metadata& meta) = 0; // Added
        virtual Status PutVector(std::string_view membrane, VectorId id,
                                 std::span<const float> vec) = 0;
        virtual Status PutVector(std::string_view membrane, VectorId id,
                                 std::span<const float> vec, const Metadata& meta) = 0;
        virtual Status PutChunk(std::string_view membrane, const RagChunk& chunk) = 0;
        virtual Status Get(std::string_view membrane, VectorId id,
                           std::vector<float> *out) = 0;
        virtual Status Get(std::string_view membrane, VectorId id,
                           std::vector<float> *out, Metadata* out_meta) = 0; // Added
        virtual Status Exists(std::string_view membrane, VectorId id,
                              bool *exists) = 0;
        virtual Status Delete(std::string_view membrane, VectorId id) = 0;
        virtual Status Search(std::string_view membrane, std::span<const float> query,
                              uint32_t topk, SearchResult *out) = 0;

        // Search with filtering options
        virtual Status Search(std::string_view membrane, std::span<const float> query,
                              uint32_t topk, const SearchOptions& opts, SearchResult *out) = 0;
        virtual Status SearchVector(std::string_view membrane, std::span<const float> query,
                                    uint32_t topk, SearchResult *out) = 0;
        virtual Status SearchVector(std::string_view membrane, std::span<const float> query,
                                    uint32_t topk, const SearchOptions& opts, SearchResult *out) = 0;
        virtual Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries, 
                                   uint32_t topk, std::vector<SearchResult>* out) = 0;
        virtual Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries, 
                                   uint32_t topk, const SearchOptions& opts, std::vector<SearchResult>* out) = 0;
        virtual Status SearchRag(std::string_view membrane, const RagQuery& query,
                                 const RagSearchOptions& opts, RagSearchResult *out) = 0;

        virtual Status Freeze(std::string_view membrane) = 0;
        virtual Status Compact(std::string_view membrane) = 0;

        // Iterator API: Full-scan access to all live vectors
        virtual Status NewIterator(std::string_view membrane,
                                  std::unique_ptr<class SnapshotIterator> *out) = 0;

        // Snapshot API
        virtual Status GetSnapshot(std::string_view membrane, std::shared_ptr<Snapshot>* out) = 0;
        virtual Status NewIterator(std::string_view membrane, const std::shared_ptr<Snapshot>& snap,
                                   std::unique_ptr<class SnapshotIterator> *out) = 0;

        static Status Open(const DBOptions &options, std::unique_ptr<DB> *out);
    };

} // namespace pomai
