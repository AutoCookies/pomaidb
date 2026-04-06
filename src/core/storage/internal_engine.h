#pragma once
#include <memory>
#include <optional>
#include <vector>
#include <span>
#include <string_view>

#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/metadata.h"
#include "pomai/search.h"
#include "pomai/options.h"
#include "pomai/snapshot.h"
#include "pomai/hooks.h"
#include "core/shard/runtime.h"
#include "core/graph/graph_membrane_impl.h"
#include "core/concurrency/scheduler.h"
#include "core/query/query_planner.h"
#include "core/kernel/micro_kernel.h"

namespace pomai {
namespace core {
struct LinkedObject;
}

class StorageEngine : public core::IQueryEngine {
public:
    Status Open(const EmbeddedOptions& options);
    Status SearchMultiModal(std::string_view membrane, const MultiModalQuery& query, SearchResult* out);
    void Close();

    Status Flush();
    Status Freeze();

    Status Append(VectorId id, std::span<const float> vec);
    Status Append(VectorId id, std::span<const float> vec, const Metadata& meta);
    Status AppendBatch(const std::vector<VectorId>& ids, const std::vector<std::span<const float>>& vectors);

    Status Get(VectorId id, std::vector<float>* out, Metadata* meta);
    Status Exists(VectorId id, bool* exists);
    Status Delete(VectorId id);
    
    // IQueryEngine implementation
    Status Search(std::string_view membrane, std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out) override;
    Status Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult* out);
    Status SearchLexical(std::string_view membrane, const std::string& query, uint32_t topk, std::vector<core::LexicalHit>* out) override;
    Status GetNeighbors(std::string_view membrane, VertexId src, std::vector<pomai::Neighbor>* out) override;
    Status GetNeighbors(std::string_view membrane, VertexId src, EdgeType type, std::vector<pomai::Neighbor>* out) override;
    std::optional<core::LinkedObject> ResolveLinkedByVectorId(uint64_t vector_id) const override;

    Status PushSync(core::SyncReceiver* receiver);

    Status AddVertex(VertexId id, TagId tag, const Metadata& meta);
    Status AddEdge(VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta);
    Status DeleteVertex(VertexId id);
    Status DeleteEdge(VertexId src, VertexId dst, EdgeType type);

    // Non-interface variants (for internal use)
    Status GetNeighbors(VertexId src, std::vector<Neighbor>* out);
    Status GetNeighbors(VertexId src, EdgeType type, std::vector<Neighbor>* out);

    Status GetSnapshot(std::shared_ptr<Snapshot>* out);
    Status NewIterator(const std::shared_ptr<Snapshot>& snap, std::unique_ptr<SnapshotIterator>* out);

    std::size_t GetMemTableBytesUsed() const;
    void AddPostPutHook(std::shared_ptr<PostPutHook> hook);

private:
    core::MicroKernel kernel_;
    std::unique_ptr<core::QueryPlanner> planner_;
    std::vector<std::shared_ptr<PostPutHook>> hooks_;
};

} // namespace pomai
