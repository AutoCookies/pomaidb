#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/iterator.h"
#include "pomai/iterator.h"
#include "pomai/metadata.h"
#include "pomai/snapshot.h"
#include "pomai/rag.h"

namespace pomai::core
{

    class VectorEngine;
    class RagEngine;

    class MembraneManager
    {
    public:
        explicit MembraneManager(pomai::DBOptions base);
        ~MembraneManager();

        MembraneManager(const MembraneManager &) = delete;
        MembraneManager &operator=(const MembraneManager &) = delete;

        Status Open();
        Status Close();

        Status FlushAll();
        Status CloseAll();

        Status CreateMembrane(const pomai::MembraneSpec &spec);
        Status DropMembrane(std::string_view name);
        Status OpenMembrane(std::string_view name);
        Status CloseMembrane(std::string_view name);

        Status ListMembranes(std::vector<std::string> *out) const;

        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec);
        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta); // Overload
        Status PutVector(std::string_view membrane, VectorId id, std::span<const float> vec);
        Status PutVector(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta);
        Status PutChunk(std::string_view membrane, const pomai::RagChunk& chunk);
        Status PutBatch(std::string_view membrane,
                        const std::vector<VectorId>& ids,
                        const std::vector<std::span<const float>>& vectors);
        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out);
        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out, Metadata* out_meta); // Added
        Status Exists(std::string_view membrane, VectorId id, bool *exists);
        Status Delete(std::string_view membrane, VectorId id);
        Status Search(std::string_view membrane, std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);
        Status Search(std::string_view membrane, std::span<const float> query, std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out);
        Status SearchVector(std::string_view membrane, std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);
        Status SearchVector(std::string_view membrane, std::span<const float> query, std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out);
        Status SearchRag(std::string_view membrane, const pomai::RagQuery& query, const pomai::RagSearchOptions& opts, pomai::RagSearchResult *out);

        Status Freeze(std::string_view membrane);
        Status Compact(std::string_view membrane);
        Status NewIterator(std::string_view membrane, std::unique_ptr<pomai::SnapshotIterator> *out);
        Status GetSnapshot(std::string_view membrane, std::shared_ptr<pomai::Snapshot>* out);
        Status NewIterator(std::string_view membrane, const std::shared_ptr<pomai::Snapshot>& snap, std::unique_ptr<pomai::SnapshotIterator> *out);

        // Default membrane convenience: use name "__default__"
        static constexpr std::string_view kDefaultMembrane = "__default__";

    private:
        struct MembraneState
        {
            pomai::MembraneSpec spec;
            std::unique_ptr<VectorEngine> vector_engine;
            std::unique_ptr<RagEngine> rag_engine;
        };

        MembraneState *GetMembraneOrNull(std::string_view name);
        const MembraneState *GetMembraneOrNull(std::string_view name) const;

        pomai::DBOptions base_;
        bool opened_ = false;

        // For now: keep engines in-memory; later you can add lazy-open by manifest.
        std::unordered_map<std::string, MembraneState> membranes_;
    };

} // namespace pomai::core
