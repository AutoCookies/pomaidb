#include "pomai/pomai.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/membrane/manager.h"
#include "util/logging.h"

namespace pomai
{

    class DbImpl final : public DB
    {
    public:
        explicit DbImpl(DBOptions opt) : mgr_(std::move(opt)) {}

        Status Init() {
            POMAI_LOG_INFO("Opening PomaiDB at: {}", mgr_.GetOptions().path);
            auto st = mgr_.Open();
            if (!st.ok()) {
                POMAI_LOG_ERROR("Failed to open PomaiDB: {}", st.message());
                return st;
            }
            return Status::Ok();
        }

        // ---- DB lifetime ----
        Status Flush() override { return mgr_.FlushAll(); }
        Status Close() override { return mgr_.CloseAll(); }

        // ---- Default membrane convenience ----
        Status Put(VectorId id, std::span<const float> vec) override
        {
            return mgr_.Put(core::MembraneManager::kDefaultMembrane, id, vec);
        }

        Status Put(VectorId id, std::span<const float> vec, const Metadata& meta) override
        {
            return mgr_.Put(core::MembraneManager::kDefaultMembrane, id, vec, meta);
        }

        Status PutVector(VectorId id, std::span<const float> vec) override
        {
            return mgr_.PutVector(core::MembraneManager::kDefaultMembrane, id, vec);
        }

        Status PutVector(VectorId id, std::span<const float> vec, const Metadata& meta) override
        {
            return mgr_.PutVector(core::MembraneManager::kDefaultMembrane, id, vec, meta);
        }

        Status PutChunk(const RagChunk& chunk) override
        {
            return mgr_.PutChunk(core::MembraneManager::kDefaultMembrane, chunk);
        }



        Status PutBatch(const std::vector<VectorId>& ids,
                        const std::vector<std::span<const float>>& vectors) override
        {
            return mgr_.PutBatch(core::MembraneManager::kDefaultMembrane, ids, vectors);
        }

        Status Get(VectorId id, std::vector<float> *out) override
        {
            return mgr_.Get(core::MembraneManager::kDefaultMembrane, id, out);
        }

        Status Get(VectorId id, std::vector<float> *out, Metadata* out_meta) override
        {
            return mgr_.Get(core::MembraneManager::kDefaultMembrane, id, out, out_meta);
        }

        Status Exists(VectorId id, bool *exists) override
        {
            return mgr_.Exists(core::MembraneManager::kDefaultMembrane, id, exists);
        }

        Status Delete(VectorId id) override
        {
            return mgr_.Delete(core::MembraneManager::kDefaultMembrane, id);
        }



        // ---- Membrane API ----
        Status CreateMembrane(const MembraneSpec &spec) override
        {
            return mgr_.CreateMembrane(spec);
        }

        Status DropMembrane(std::string_view name) override
        {
            return mgr_.DropMembrane(name);
        }

        Status OpenMembrane(std::string_view name) override
        {
            return mgr_.OpenMembrane(name);
        }

        Status CloseMembrane(std::string_view name) override
        {
            return mgr_.CloseMembrane(name);
        }

        Status ListMembranes(std::vector<std::string> *out) const override
        {
            return mgr_.ListMembranes(out);
        }
        Status UpdateMembraneRetention(std::string_view name, uint32_t ttl_sec, uint32_t retention_max_count, uint64_t retention_max_bytes) override
        {
            return mgr_.UpdateMembraneRetention(name, ttl_sec, retention_max_count, retention_max_bytes);
        }
        Status GetMembraneRetention(std::string_view name, uint32_t* ttl_sec, uint32_t* retention_max_count, uint64_t* retention_max_bytes) const override
        {
            return mgr_.GetMembraneRetention(name, ttl_sec, retention_max_count, retention_max_bytes);
        }

        // ---- Membrane-scoped operations ----
        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec) override
        {
            return mgr_.Put(membrane, id, vec);
        }

        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta) override
        {
            return mgr_.Put(membrane, id, vec, meta);
        }

        Status PutVector(std::string_view membrane, VectorId id, std::span<const float> vec) override
        {
            return mgr_.PutVector(membrane, id, vec);
        }

        Status PutVector(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta) override
        {
            return mgr_.PutVector(membrane, id, vec, meta);
        }

        Status PutChunk(std::string_view membrane, const RagChunk& chunk) override
        {
            return mgr_.PutChunk(membrane, chunk);
        }

        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out) override
        {
            return mgr_.Get(membrane, id, out);
        }

        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out, Metadata* out_meta) override
        {
            return mgr_.Get(membrane, id, out, out_meta);
        }

        Status Exists(std::string_view membrane, VectorId id, bool *exists) override
        {
            return mgr_.Exists(membrane, id, exists);
        }

        Status Delete(std::string_view membrane, VectorId id) override
        {
            return mgr_.Delete(membrane, id);
        }

        Status Search(std::span<const float> query, uint32_t topk, SearchResult *out) override
        {
            return mgr_.Search(core::MembraneManager::kDefaultMembrane, query, topk, out);
        }

        Status Search(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult *out) override
        {
            return mgr_.Search(core::MembraneManager::kDefaultMembrane, query, topk, opts, out);
        }

        Status SearchVector(std::span<const float> query, uint32_t topk, SearchResult *out) override
        {
            return mgr_.SearchVector(core::MembraneManager::kDefaultMembrane, query, topk, out);
        }

        Status SearchVector(std::span<const float> query, uint32_t topk, const SearchOptions& opts, SearchResult *out) override
        {
            return mgr_.SearchVector(core::MembraneManager::kDefaultMembrane, query, topk, opts, out);
        }

        Status SearchRag(const RagQuery& query, const RagSearchOptions& opts, RagSearchResult *out) override
        {
            return mgr_.SearchRag(core::MembraneManager::kDefaultMembrane, query, opts, out);
        }

        Status SearchMultiModal(const MultiModalQuery& query, SearchResult* out) override {
            return mgr_.SearchMultiModal(core::MembraneManager::kDefaultMembrane, query, out);
        }

        Status SearchBatch(std::span<const float> queries, uint32_t num_queries,
                           uint32_t topk, std::vector<SearchResult>* out) override
        {
            return mgr_.SearchBatch(core::MembraneManager::kDefaultMembrane, queries, num_queries, topk, out);
        }

        Status SearchBatch(std::span<const float> queries, uint32_t num_queries,
                           uint32_t topk, const SearchOptions& opts, std::vector<SearchResult>* out) override
        {
            return mgr_.SearchBatch(core::MembraneManager::kDefaultMembrane, queries, num_queries, topk, opts, out);
        }

        // ...

        Status Search(std::string_view membrane, std::span<const float> query,
                      uint32_t topk, SearchResult *out) override
        {
            return mgr_.Search(membrane, query, topk, out);
        }

        Status Search(std::string_view membrane, std::span<const float> query,
                      uint32_t topk, const SearchOptions& opts, SearchResult *out) override
        {
            return mgr_.Search(membrane, query, topk, opts, out);
        }

        Status SearchVector(std::string_view membrane, std::span<const float> query,
                            uint32_t topk, SearchResult *out) override
        {
            return mgr_.SearchVector(membrane, query, topk, out);
        }

        Status SearchVector(std::string_view membrane, std::span<const float> query,
                            uint32_t topk, const SearchOptions& opts, SearchResult *out) override
        {
            return mgr_.SearchVector(membrane, query, topk, opts, out);
        }

        Status SearchRag(std::string_view membrane, const RagQuery& query,
                         const RagSearchOptions& opts, RagSearchResult *out) override
        {
            return mgr_.SearchRag(membrane, query, opts, out);
        }

        Status SearchMultiModal(std::string_view membrane, const MultiModalQuery& query, SearchResult* out) override {
            return mgr_.SearchMultiModal(membrane, query, out);
        }

        Status TsPut(std::string_view membrane, uint64_t series_id, uint64_t timestamp, double value) override {
            return mgr_.TsPut(membrane, series_id, timestamp, value);
        }
        Status TsRange(std::string_view membrane, uint64_t series_id, uint64_t start_ts, uint64_t end_ts, std::vector<TimeSeriesPoint>* out) override {
            return mgr_.TsRange(membrane, series_id, start_ts, end_ts, out);
        }
        Status KvPut(std::string_view membrane, std::string_view key, std::string_view value) override {
            return mgr_.KvPut(membrane, key, value);
        }
        Status KvGet(std::string_view membrane, std::string_view key, std::string* out) override {
            return mgr_.KvGet(membrane, key, out);
        }
        Status KvDelete(std::string_view membrane, std::string_view key) override {
            return mgr_.KvDelete(membrane, key);
        }
        Status MetaPut(std::string_view membrane, std::string_view gid, std::string_view value) override {
            return mgr_.MetaPut(membrane, gid, value);
        }
        Status MetaGet(std::string_view membrane, std::string_view gid, std::string* out) override {
            return mgr_.MetaGet(membrane, gid, out);
        }
        Status MetaDelete(std::string_view membrane, std::string_view gid) override {
            return mgr_.MetaDelete(membrane, gid);
        }
        Status LinkObjects(std::string_view gid, uint64_t vector_id, uint64_t graph_vertex_id, uint64_t mesh_id) override {
            return mgr_.LinkObjects(gid, vector_id, graph_vertex_id, mesh_id);
        }
        Status UnlinkObjects(std::string_view gid) override {
            return mgr_.UnlinkObjects(gid);
        }
        Status StartEdgeGateway(uint16_t http_port, uint16_t ingest_port) override {
            return mgr_.StartEdgeGateway(http_port, ingest_port);
        }
        Status StartEdgeGatewaySecure(uint16_t http_port, uint16_t ingest_port, std::string_view auth_token) override {
            return mgr_.StartEdgeGatewaySecure(http_port, ingest_port, auth_token);
        }
        Status StopEdgeGateway() override {
            return mgr_.StopEdgeGateway();
        }
        Status SketchAdd(std::string_view membrane, std::string_view key, uint64_t increment) override {
            return mgr_.SketchAdd(membrane, key, increment);
        }
        Status SketchEstimate(std::string_view membrane, std::string_view key, uint64_t* out) override {
            return mgr_.SketchEstimate(membrane, key, out);
        }
        Status SketchSeen(std::string_view membrane, std::string_view key, bool* out) override {
            return mgr_.SketchSeen(membrane, key, out);
        }
        Status SketchUniqueEstimate(std::string_view membrane, uint64_t* out) override {
            return mgr_.SketchUniqueEstimate(membrane, out);
        }
        Status BlobPut(std::string_view membrane, uint64_t blob_id, std::span<const uint8_t> data) override {
            return mgr_.BlobPut(membrane, blob_id, data);
        }
        Status BlobGet(std::string_view membrane, uint64_t blob_id, std::vector<uint8_t>* out) override {
            return mgr_.BlobGet(membrane, blob_id, out);
        }
        Status BlobDelete(std::string_view membrane, uint64_t blob_id) override {
            return mgr_.BlobDelete(membrane, blob_id);
        }
        Status SpatialPut(std::string_view membrane, uint64_t entity_id, double latitude, double longitude) override {
            return mgr_.SpatialPut(membrane, entity_id, latitude, longitude);
        }
        Status SpatialRadiusSearch(std::string_view membrane, double latitude, double longitude, double radius_meters, std::vector<SpatialPoint>* out) override {
            return mgr_.SpatialRadiusSearch(membrane, latitude, longitude, radius_meters, out);
        }
        Status SpatialWithinPolygon(std::string_view membrane, const GeoPolygon& polygon, std::vector<SpatialPoint>* out) override {
            return mgr_.SpatialWithinPolygon(membrane, polygon, out);
        }
        Status SpatialNearest(std::string_view membrane, double latitude, double longitude, uint32_t topk, std::vector<SpatialPoint>* out) override {
            return mgr_.SpatialNearest(membrane, latitude, longitude, topk, out);
        }
        Status MeshPut(std::string_view membrane, uint64_t mesh_id, std::span<const float> vertices_xyz) override {
            return mgr_.MeshPut(membrane, mesh_id, vertices_xyz);
        }
        Status MeshRmsd(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, double* out) override {
            return mgr_.MeshRmsd(membrane, mesh_a, mesh_b, out);
        }
        Status MeshIntersect(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, bool* out) override {
            return mgr_.MeshIntersect(membrane, mesh_a, mesh_b, out);
        }
        Status MeshVolume(std::string_view membrane, uint64_t mesh_id, double* out) override {
            return mgr_.MeshVolume(membrane, mesh_id, out);
        }
        Status MeshRmsd(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, const MeshQueryOptions& opts, double* out) override {
            return mgr_.MeshRmsd(membrane, mesh_a, mesh_b, opts, out);
        }
        Status MeshIntersect(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, const MeshQueryOptions& opts, bool* out) override {
            return mgr_.MeshIntersect(membrane, mesh_a, mesh_b, opts, out);
        }
        Status MeshVolume(std::string_view membrane, uint64_t mesh_id, const MeshQueryOptions& opts, double* out) override {
            return mgr_.MeshVolume(membrane, mesh_id, opts, out);
        }
        Status SparsePut(std::string_view membrane, uint64_t id, const SparseEntry& entry) override {
            return mgr_.SparsePut(membrane, id, entry);
        }
        Status SparseDot(std::string_view membrane, uint64_t a, uint64_t b, double* out) override {
            return mgr_.SparseDot(membrane, a, b, out);
        }
        Status SparseIntersect(std::string_view membrane, uint64_t a, uint64_t b, uint32_t* out) override {
            return mgr_.SparseIntersect(membrane, a, b, out);
        }
        Status SparseJaccard(std::string_view membrane, uint64_t a, uint64_t b, double* out) override {
            return mgr_.SparseJaccard(membrane, a, b, out);
        }
        Status BitsetPut(std::string_view membrane, uint64_t id, std::span<const uint8_t> bits) override {
            return mgr_.BitsetPut(membrane, id, bits);
        }
        Status BitsetAnd(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) override {
            return mgr_.BitsetAnd(membrane, a, b, out);
        }
        Status BitsetOr(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) override {
            return mgr_.BitsetOr(membrane, a, b, out);
        }
        Status BitsetXor(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) override {
            return mgr_.BitsetXor(membrane, a, b, out);
        }
        Status BitsetHamming(std::string_view membrane, uint64_t a, uint64_t b, double* out) override {
            return mgr_.BitsetHamming(membrane, a, b, out);
        }
        Status BitsetJaccard(std::string_view membrane, uint64_t a, uint64_t b, double* out) override {
            return mgr_.BitsetJaccard(membrane, a, b, out);
        }

        Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries,
                           uint32_t topk, std::vector<SearchResult>* out) override
        {
            return mgr_.SearchBatch(membrane, queries, num_queries, topk, out);
        }

        Status SearchBatch(std::string_view membrane, std::span<const float> queries, uint32_t num_queries,
                           uint32_t topk, const SearchOptions& opts, std::vector<SearchResult>* out) override
        {
            return mgr_.SearchBatch(membrane, queries, num_queries, topk, opts, out);
        }

        Status Freeze(std::string_view membrane) override
        {
            return mgr_.Freeze(membrane);
        }

        Status Compact(std::string_view membrane) override
        {
            return mgr_.Compact(membrane);
        }

        Status NewIterator(std::string_view membrane, std::unique_ptr<SnapshotIterator> *out) override
        {
            return mgr_.NewIterator(membrane, out);
        }

        Status NewMembraneRecordIterator(std::string_view membrane, std::unique_ptr<MembraneRecordIterator>* out) override {
            return mgr_.NewMembraneRecordIterator(membrane, out);
        }
        Status NewMembraneRecordIterator(std::string_view membrane, const MembraneScanOptions& scan_opts,
                                         std::unique_ptr<MembraneRecordIterator>* out) override {
            return mgr_.NewMembraneRecordIterator(membrane, scan_opts, out);
        }

        Status GetSnapshot(std::string_view membrane, std::shared_ptr<Snapshot>* out) override
        {
            return mgr_.GetSnapshot(membrane, out);
        }

        Status NewIterator(std::string_view membrane, const std::shared_ptr<Snapshot>& snap, std::unique_ptr<SnapshotIterator> *out) override
        {
            return mgr_.NewIterator(membrane, snap, out);
        }

    private:
        core::MembraneManager mgr_;
    };

    Status DB::Open(const DBOptions &options, std::unique_ptr<DB> *out)
    {
        if (!out)
            return Status::InvalidArgument("out=null");
        if (options.path.empty())
            return Status::InvalidArgument("path empty");
        if (options.dim == 0)
            return Status::InvalidArgument("dim must be > 0");
        if (options.shard_count == 0)
            return Status::InvalidArgument("shard_count must be > 0");
        
        DBOptions effective = options;
        effective.ApplyEdgeProfile();
        auto impl = std::make_unique<DbImpl>(std::move(effective));
        auto st = impl->Init();
        if (!st.ok()) {
             return st;
        }
        *out = std::move(impl);
        return Status::Ok();
    }

} // namespace pomai
