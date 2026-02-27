#include "pomai/pomai.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/membrane/manager.h"

namespace pomai
{

    class DbImpl final : public DB
    {
    public:
        explicit DbImpl(DBOptions opt) : mgr_(std::move(opt)) {}

        Status Init() {
            auto st = mgr_.Open();
            if (!st.ok()) return st;
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
        
        auto impl = std::make_unique<DbImpl>(options);
        auto st = impl->Init();
        if (!st.ok()) {
             return st;
        }
        *out = std::move(impl);
        return Status::Ok();
    }

} // namespace pomai
