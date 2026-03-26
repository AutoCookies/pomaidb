#pragma once
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/iterator.h"
#include "pomai/membrane_iterator.h"
#include "pomai/metadata.h"
#include "pomai/snapshot.h"
#include "pomai/rag.h"
#include "pomai/hooks.h"
#include "pomai/graph.h"
#include "core/query/query_planner.h"
#include "core/query/query_orchestrator.h"
#include "core/lifecycle/semantic_lifecycle.h"
#include "core/concurrency/scheduler.h"
#include "core/mesh/mesh_engine.h"
#include "core/linker/object_linker.h"
#include "core/audio/audio_engine.h"
#include "core/bloom/bloom_engine.h"
#include "core/document/document_engine.h"

namespace pomai {
    class GraphMembrane;
    struct TimeSeriesPoint;
    struct SpatialPoint;
    struct GeoPolygon;
    struct SparseEntry;
}
namespace pomai::core
{

    class VectorEngine;
    class RagEngine;
    class TextMembrane;
    class TimeSeriesEngine;
    class KeyValueEngine;
    class SketchEngine;
    class BlobEngine;
    class SpatialEngine;
    class MeshEngine;
    class SparseEngine;
    class BitsetEngine;
    class EdgeGateway;
    class SyncReceiver;

    class MembraneManager : public IQueryEngine
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
        Status UpdateMembraneRetention(std::string_view name, uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes);
        Status GetMembraneRetention(std::string_view name, uint32_t* ttl_sec, uint32_t* retention_max_count, uint64_t* retention_max_bytes) const;

        Status ListMembranes(std::vector<std::string> *out) const;

        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec);
        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta);
        Status PutVector(std::string_view membrane, VectorId id, std::span<const float> vec);
        Status PutVector(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta);
        Status PutChunk(std::string_view membrane, const pomai::RagChunk& chunk);
        Status PutBatch(std::string_view membrane,
                        const std::vector<VectorId>& ids,
                        const std::vector<std::span<const float>>& vectors);
        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out);
        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out, Metadata* out_meta);
        Status Exists(std::string_view membrane, VectorId id, bool *exists);
        Status Delete(std::string_view membrane, VectorId id);
        
        // IQueryEngine implementation
        Status Search(std::string_view membrane, std::span<const float> query, std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out) override;
        Status SearchLexical(std::string_view membrane, const std::string& query, uint32_t topk, std::vector<LexicalHit>* out) override;
        Status GetNeighbors(std::string_view membrane, VertexId src, std::vector<pomai::Neighbor>* out) override;
        Status GetNeighbors(std::string_view membrane, VertexId src, EdgeType type, std::vector<pomai::Neighbor>* out) override;

        // Overloads
        Status Search(std::string_view membrane, std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);
        Status SearchVector(std::string_view membrane, std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);
        Status SearchVector(std::string_view membrane, std::span<const float> query, std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out);
        /** @brief Zero-copy search vector directly into a sink. */
        Status SearchVector(std::string_view membrane, std::span<const float> query, std::uint32_t topk, const SearchOptions& opts, pomai::SearchHitSink& sink);
        Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries, std::uint32_t topk, std::vector<pomai::SearchResult>* out);
        Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries, std::uint32_t topk, const SearchOptions& opts, std::vector<pomai::SearchResult>* out);
        Status SearchRag(std::string_view membrane, const pomai::RagQuery& query, const pomai::RagSearchOptions& opts, pomai::RagSearchResult *out);
        Status SearchMultiModal(std::string_view membrane, const MultiModalQuery& query, SearchResult* out);

        Status TsPut(std::string_view membrane, uint64_t series_id, uint64_t timestamp, double value);
        Status TsRange(std::string_view membrane, uint64_t series_id, uint64_t start_ts, uint64_t end_ts, std::vector<pomai::TimeSeriesPoint>* out);
        Status KvPut(std::string_view membrane, std::string_view key, std::string_view value);
        Status KvGet(std::string_view membrane, std::string_view key, std::string* out);
        Status KvDelete(std::string_view membrane, std::string_view key);
        Status MetaPut(std::string_view membrane, std::string_view gid, std::string_view value);
        Status MetaGet(std::string_view membrane, std::string_view gid, std::string* out);
        Status MetaDelete(std::string_view membrane, std::string_view gid);
        Status LinkObjects(std::string_view gid, uint64_t vector_id, uint64_t graph_vertex_id, uint64_t mesh_id);
        Status UnlinkObjects(std::string_view gid);
        std::optional<LinkedObject> ResolveLinkedByVectorId(uint64_t vector_id) const override;
        Status StartEdgeGateway(uint16_t http_port, uint16_t ingest_port);
        Status StartEdgeGatewaySecure(uint16_t http_port, uint16_t ingest_port, std::string_view auth_token);
        Status StopEdgeGateway();
        Status SketchAdd(std::string_view membrane, std::string_view key, uint64_t increment);
        Status SketchEstimate(std::string_view membrane, std::string_view key, uint64_t* out);
        Status SketchSeen(std::string_view membrane, std::string_view key, bool* out);
        Status SketchUniqueEstimate(std::string_view membrane, uint64_t* out);
        Status BlobPut(std::string_view membrane, uint64_t blob_id, std::span<const uint8_t> data);
        Status BlobGet(std::string_view membrane, uint64_t blob_id, std::vector<uint8_t>* out);
        Status BlobDelete(std::string_view membrane, uint64_t blob_id);
        Status SpatialPut(std::string_view membrane, uint64_t entity_id, double latitude, double longitude);
        Status SpatialRadiusSearch(std::string_view membrane, double latitude, double longitude, double radius_meters, std::vector<pomai::SpatialPoint>* out);
        Status SpatialWithinPolygon(std::string_view membrane, const pomai::GeoPolygon& polygon, std::vector<pomai::SpatialPoint>* out);
        Status SpatialNearest(std::string_view membrane, double latitude, double longitude, uint32_t topk, std::vector<pomai::SpatialPoint>* out);
        Status MeshPut(std::string_view membrane, uint64_t mesh_id, std::span<const float> vertices_xyz);
        Status MeshRmsd(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, double* out);
        Status MeshIntersect(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, bool* out);
        Status MeshVolume(std::string_view membrane, uint64_t mesh_id, double* out);
        Status MeshRmsd(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, const MeshQueryOptions& opts, double* out);
        Status MeshIntersect(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, const MeshQueryOptions& opts, bool* out);
        Status MeshVolume(std::string_view membrane, uint64_t mesh_id, const MeshQueryOptions& opts, double* out);
        Status SparsePut(std::string_view membrane, uint64_t id, const pomai::SparseEntry& entry);
        Status SparseDot(std::string_view membrane, uint64_t a, uint64_t b, double* out);
        Status SparseIntersect(std::string_view membrane, uint64_t a, uint64_t b, uint32_t* out);
        Status SparseJaccard(std::string_view membrane, uint64_t a, uint64_t b, double* out);
        Status BitsetPut(std::string_view membrane, uint64_t id, std::span<const uint8_t> bits);
        Status BitsetAnd(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out);
        Status BitsetOr(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out);
        Status BitsetXor(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out);
        Status BitsetHamming(std::string_view membrane, uint64_t a, uint64_t b, double* out);
        Status BitsetJaccard(std::string_view membrane, uint64_t a, uint64_t b, double* out);

        // Audio membrane APIs
        Status AudioPut(std::string_view membrane, uint64_t clip_id, uint64_t timestamp_ms,
                        std::span<const float> embedding);
        Status AudioDelete(std::string_view membrane, uint64_t clip_id);
        Status AudioSearch(std::string_view membrane, std::span<const float> query,
                           uint64_t time_start_ms, uint64_t time_end_ms,
                           uint32_t topk, std::vector<AudioHit>* out);

        // Bloom filter membrane APIs
        Status BloomAdd(std::string_view membrane, uint64_t filter_id, std::string_view key);
        Status BloomMightContain(std::string_view membrane, uint64_t filter_id, std::string_view key, bool* out);
        Status BloomDrop(std::string_view membrane, uint64_t filter_id);
        Status BloomEstimateFPR(std::string_view membrane, uint64_t filter_id, double* out);

        // Document membrane APIs
        Status DocumentPut(std::string_view membrane, uint64_t doc_id, std::string_view json_content);
        Status DocumentGet(std::string_view membrane, uint64_t doc_id, std::string* out);
        Status DocumentDelete(std::string_view membrane, uint64_t doc_id);
        Status DocumentSearch(std::string_view membrane, const std::string& query,
                              uint32_t topk, std::vector<DocumentHit>* out);
        
        // Graph Operations
        Status AddVertex(std::string_view membrane, VertexId id, TagId tag, const Metadata& meta);
        Status AddEdge(std::string_view membrane, VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta);

        Status Freeze(std::string_view membrane);
        Status Compact(std::string_view membrane);
        Status NewIterator(std::string_view membrane, std::unique_ptr<pomai::SnapshotIterator> *out);
        /** Full-scan rows for any membrane kind (see pomai/membrane_iterator.h). */
        Status NewMembraneRecordIterator(std::string_view membrane, std::unique_ptr<pomai::MembraneRecordIterator> *out);
        Status NewMembraneRecordIterator(std::string_view membrane, const pomai::MembraneScanOptions& scan_opts,
                                         std::unique_ptr<pomai::MembraneRecordIterator>* out);
        Status GetSnapshot(std::string_view name, std::shared_ptr<pomai::Snapshot> *out);
        Status PushSync(std::string_view name, SyncReceiver* receiver);

        /// Re-applies one gateway sync record (same field layout as EdgeGateway upstream JSON: id, id2, u32_a, u32_b, aux_k, aux_v).
        Status ReplayGatewaySyncEvent(uint64_t seq, std::string_view type, std::string_view membrane, uint64_t id,
                                      uint64_t id2, uint32_t u32_a, uint32_t u32_b, std::string_view aux_k,
                                      std::string_view aux_v);
        void AddPostPutHook(std::string_view membrane, std::shared_ptr<PostPutHook> hook);
        Status NewIterator(std::string_view membrane, const std::shared_ptr<pomai::Snapshot>& snap, std::unique_ptr<pomai::SnapshotIterator> *out);

        const pomai::DBOptions& GetOptions() const { return base_; }
        void RunMeshLodSlice();

        // Default membrane convenience: use name "__default__"
        static constexpr std::string_view kDefaultMembrane = "__default__";

    private:
        struct MembraneState
        {
            pomai::MembraneSpec spec;
            std::unique_ptr<VectorEngine> vector_engine;
            std::unique_ptr<RagEngine> rag_engine;
            std::unique_ptr<pomai::GraphMembrane> graph_engine;
            std::unique_ptr<TextMembrane> text_engine;
            std::unique_ptr<TimeSeriesEngine> timeseries_engine;
            std::unique_ptr<KeyValueEngine> keyvalue_engine;
            std::unique_ptr<KeyValueEngine> meta_engine;
            std::unique_ptr<SketchEngine> sketch_engine;
            std::unique_ptr<BlobEngine> blob_engine;
            std::unique_ptr<SpatialEngine> spatial_engine;
            std::unique_ptr<MeshEngine> mesh_engine;
            std::unique_ptr<SparseEngine> sparse_engine;
            std::unique_ptr<BitsetEngine> bitset_engine;
            std::unique_ptr<AudioEngine> audio_engine;
            std::unique_ptr<BloomEngine> bloom_engine;
            std::unique_ptr<DocumentEngine> document_engine;
            SemanticLifecycle lifecycle;
            std::vector<std::shared_ptr<PostPutHook>> hooks;
        };

        MembraneState *GetMembraneOrNull(std::string_view name);
        const MembraneState *GetMembraneOrNull(std::string_view name) const;

        /** Backpressure helper: if enabled and over threshold, Freeze() before writes. */
        Status MaybeApplyBackpressure(MembraneState* state);
        void PollMaintenance();

        pomai::DBOptions base_;
        bool opened_ = false;

        // For now: keep engines in-memory; later you can add lazy-open by manifest.
        std::unordered_map<std::string, MembraneState> membranes_;
        ObjectLinker object_linker_;
        std::unique_ptr<EdgeGateway> edge_gateway_;
        std::unique_ptr<QueryOrchestrator> orchestrator_;
        TaskScheduler scheduler_;
    };

} // namespace pomai::core
