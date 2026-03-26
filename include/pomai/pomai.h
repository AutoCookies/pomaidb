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
#include "membrane_iterator.h"
#include "snapshot.h"

namespace pomai
{
    struct TimeSeriesPoint {
        uint64_t timestamp = 0;
        double value = 0.0;
    };
    struct SpatialPoint {
        uint64_t id = 0;
        double latitude = 0.0;
        double longitude = 0.0;
    };
    struct GeoPolygon {
        std::vector<SpatialPoint> vertices;
    };
    struct SparseEntry {
        std::vector<uint32_t> indices;
        std::vector<float> weights;
    };

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
        /** @brief Zero-copy search vector directly into a sink. skips SearchResult allocation/copy. */
        virtual Status SearchVector(std::span<const float> query, uint32_t topk,
                                    const SearchOptions& opts, SearchHitSink& sink) = 0;
        
        // Batch Search (runs concurrently across multiple queries)
        virtual Status SearchBatch(std::span<const float> queries, uint32_t num_queries, 
                                   uint32_t topk, std::vector<SearchResult>* out) = 0;
        virtual Status SearchBatch(std::span<const float> queries, uint32_t num_queries, 
                                   uint32_t topk, const SearchOptions& opts, std::vector<SearchResult>* out) = 0;
        virtual Status SearchRag(const RagQuery& query, const RagSearchOptions& opts, RagSearchResult *out) = 0;
        virtual Status SearchMultiModal(const MultiModalQuery& query, SearchResult* out) = 0; // Added

        // Membrane API
        virtual Status CreateMembrane(const MembraneSpec &spec) = 0;
        virtual Status DropMembrane(std::string_view name) = 0;
        virtual Status OpenMembrane(std::string_view name) = 0;
        virtual Status CloseMembrane(std::string_view name) = 0;
        virtual Status ListMembranes(std::vector<std::string> *out) const = 0;
        virtual Status UpdateMembraneRetention(std::string_view name, uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes) = 0;
        virtual Status GetMembraneRetention(std::string_view name, uint32_t* ttl_sec, uint32_t* retention_max_count, uint64_t* retention_max_bytes) const = 0;

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
        /** @brief Zero-copy membrane search vector directly into a sink. */
        virtual Status SearchVector(std::string_view membrane, std::span<const float> query,
                                    uint32_t topk, const SearchOptions& opts, SearchHitSink& sink) = 0;
        virtual Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries, 
                                   uint32_t topk, std::vector<SearchResult>* out) = 0;
        virtual Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries, 
                                   uint32_t topk, const SearchOptions& opts, std::vector<SearchResult>* out) = 0;
        virtual Status SearchRag(std::string_view membrane, const RagQuery& query,
                                 const RagSearchOptions& opts, RagSearchResult *out) = 0;
        virtual Status SearchMultiModal(std::string_view membrane, const MultiModalQuery& query, SearchResult* out) = 0; // Added

        // TimeSeries API
        virtual Status TsPut(std::string_view membrane, uint64_t series_id, uint64_t timestamp, double value) = 0;
        virtual Status TsRange(std::string_view membrane, uint64_t series_id, uint64_t start_ts, uint64_t end_ts, std::vector<TimeSeriesPoint>* out) = 0;

        // KeyValue API
        virtual Status KvPut(std::string_view membrane, std::string_view key, std::string_view value) = 0;
        virtual Status KvGet(std::string_view membrane, std::string_view key, std::string* out) = 0;
        virtual Status KvDelete(std::string_view membrane, std::string_view key) = 0;
        virtual Status MetaPut(std::string_view membrane, std::string_view gid, std::string_view value) = 0;
        virtual Status MetaGet(std::string_view membrane, std::string_view gid, std::string* out) = 0;
        virtual Status MetaDelete(std::string_view membrane, std::string_view gid) = 0;
        virtual Status LinkObjects(std::string_view gid, uint64_t vector_id, uint64_t graph_vertex_id, uint64_t mesh_id) = 0;
        virtual Status UnlinkObjects(std::string_view gid) = 0;
        virtual Status StartEdgeGateway(uint16_t http_port, uint16_t ingest_port) = 0;
        virtual Status StartEdgeGatewaySecure(uint16_t http_port, uint16_t ingest_port, std::string_view auth_token) = 0;
        virtual Status StopEdgeGateway() = 0;

        // Sketch API
        virtual Status SketchAdd(std::string_view membrane, std::string_view key, uint64_t increment) = 0;
        virtual Status SketchEstimate(std::string_view membrane, std::string_view key, uint64_t* out) = 0;
        virtual Status SketchSeen(std::string_view membrane, std::string_view key, bool* out) = 0;
        virtual Status SketchUniqueEstimate(std::string_view membrane, uint64_t* out) = 0;

        // Graph API
        virtual Status AddVertex(VertexId id, TagId tag, const Metadata& meta) = 0;
        virtual Status AddEdge(VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta) = 0;
        virtual Status GetNeighbors(VertexId src, std::vector<Neighbor>* out) = 0;
        virtual Status GetNeighbors(VertexId src, EdgeType type, std::vector<Neighbor>* out) = 0;

        // Blob API
        virtual Status BlobPut(std::string_view membrane, uint64_t blob_id, std::span<const uint8_t> data) = 0;
        virtual Status BlobGet(std::string_view membrane, uint64_t blob_id, std::vector<uint8_t>* out) = 0;
        virtual Status BlobDelete(std::string_view membrane, uint64_t blob_id) = 0;

        // Spatial API
        virtual Status SpatialPut(std::string_view membrane, uint64_t entity_id, double latitude, double longitude) = 0;
        virtual Status SpatialRadiusSearch(std::string_view membrane, double latitude, double longitude, double radius_meters, std::vector<SpatialPoint>* out) = 0;
        virtual Status SpatialWithinPolygon(std::string_view membrane, const GeoPolygon& polygon, std::vector<SpatialPoint>* out) = 0;
        virtual Status SpatialNearest(std::string_view membrane, double latitude, double longitude, uint32_t topk, std::vector<SpatialPoint>* out) = 0;

        // Mesh API
        virtual Status MeshPut(std::string_view membrane, uint64_t mesh_id, std::span<const float> vertices_xyz) = 0;
        virtual Status MeshRmsd(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, double* out) = 0;
        virtual Status MeshIntersect(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, bool* out) = 0;
        virtual Status MeshVolume(std::string_view membrane, uint64_t mesh_id, double* out) = 0;
        virtual Status MeshRmsd(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, const MeshQueryOptions& opts, double* out) = 0;
        virtual Status MeshIntersect(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, const MeshQueryOptions& opts, bool* out) = 0;
        virtual Status MeshVolume(std::string_view membrane, uint64_t mesh_id, const MeshQueryOptions& opts, double* out) = 0;

        // Sparse API
        virtual Status SparsePut(std::string_view membrane, uint64_t id, const SparseEntry& entry) = 0;
        virtual Status SparseDot(std::string_view membrane, uint64_t a, uint64_t b, double* out) = 0;
        virtual Status SparseIntersect(std::string_view membrane, uint64_t a, uint64_t b, uint32_t* out) = 0;
        virtual Status SparseJaccard(std::string_view membrane, uint64_t a, uint64_t b, double* out) = 0;

        // Bitset API
        virtual Status BitsetPut(std::string_view membrane, uint64_t id, std::span<const uint8_t> bits) = 0;
        virtual Status BitsetAnd(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) = 0;
        virtual Status BitsetOr(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) = 0;
        virtual Status BitsetXor(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) = 0;
        virtual Status BitsetHamming(std::string_view membrane, uint64_t a, uint64_t b, double* out) = 0;
        virtual Status BitsetJaccard(std::string_view membrane, uint64_t a, uint64_t b, double* out) = 0;

        virtual Status Freeze(std::string_view membrane) = 0;
        virtual Status Compact(std::string_view membrane) = 0;

        // Iterator API: Full-scan access to all live vectors
        virtual Status NewIterator(std::string_view membrane,
                                  std::unique_ptr<class SnapshotIterator> *out) = 0;

        /** Full-scan over a membrane for any MembraneKind (unified record shape; see membrane_iterator.h). */
        virtual Status NewMembraneRecordIterator(std::string_view membrane,
                                                  std::unique_ptr<MembraneRecordIterator>* out) = 0;
        virtual Status NewMembraneRecordIterator(std::string_view membrane, const MembraneScanOptions& scan_opts,
                                                  std::unique_ptr<MembraneRecordIterator>* out) = 0;

        // Snapshot API
        virtual Status GetSnapshot(std::string_view membrane, std::shared_ptr<Snapshot>* out) = 0;
        virtual Status NewIterator(std::string_view membrane, const std::shared_ptr<Snapshot>& snap,
                                   std::unique_ptr<class SnapshotIterator> *out) = 0;

        static Status Open(const DBOptions &options, std::unique_ptr<DB> *out);
    };

} // namespace pomai
