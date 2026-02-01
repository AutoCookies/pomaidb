#pragma once
#include <atomic>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>

#include "pomai/core/blob_store.h"
#include "pomai/core/command.h"
#include "pomai/core/manifest.h"
#include "pomai/core/stats.h"
#include "pomai/core/wal.h"
#include "pomai/index/hnsw_wrapper.h"

namespace pomai::core
{
    struct ShardOptions
    {
        // Backward-compatible base path.
        // Step2 will create a directory "<wal_path>.d/" containing:
        //  - MANIFEST
        //  - wal/wal_<id>.log ...
        //  - checkpoints/...
        std::filesystem::path wal_path;

        FsyncPolicy fsync_policy{FsyncPolicy::Never};
        std::uint32_t vector_dim{0};

        // Legacy/compat: ops-triggered checkpointing.
        // Step3 production path: maintenance thread triggers checkpoint by WAL bytes/time,
        // so this SHOULD be 0 in production.
        std::uint64_t checkpoint_interval_ops{0};

        // Optional: persist HNSW (expensive). Default false for performance.
        bool persist_hnsw_on_checkpoint{false};
    };

    struct ReadSnapshot
    {
        std::uint64_t num_vectors{0};
    };

    class PagedVectorArena
    {
    public:
        static constexpr std::size_t kPageSizeBytes = 2 * 1024 * 1024;

        explicit PagedVectorArena(std::uint32_t dim) : dim_(dim)
        {
            std::size_t vec_size = static_cast<std::size_t>(dim_) * sizeof(float);
            vectors_per_page_ = kPageSizeBytes / vec_size;
            if (vectors_per_page_ < 1)
                vectors_per_page_ = 1;
            ids_.reserve(100000);
        }

        std::uint32_t Add(VectorId id, const std::vector<float> &vec)
        {
            std::uint32_t idx = static_cast<std::uint32_t>(ids_.size());
            std::uint32_t page_idx = idx / static_cast<std::uint32_t>(vectors_per_page_);
            std::uint32_t offset_in_page = idx % static_cast<std::uint32_t>(vectors_per_page_);

            if (page_idx >= pages_.size())
            {
                float *new_page = new float[vectors_per_page_ * dim_];
                pages_.push_back(std::unique_ptr<float[]>(new_page));
            }

            float *page_ptr = pages_[page_idx].get();
            float *dest = page_ptr + (static_cast<std::size_t>(offset_in_page) * dim_);
            std::memcpy(dest, vec.data(), static_cast<std::size_t>(dim_) * sizeof(float));

            ids_.push_back(id);
            return idx;
        }

        const float *GetVector(std::uint32_t idx) const
        {
            std::uint32_t page_idx = idx / static_cast<std::uint32_t>(vectors_per_page_);
            std::uint32_t offset_in_page = idx % static_cast<std::uint32_t>(vectors_per_page_);
            return pages_[page_idx].get() + (static_cast<std::size_t>(offset_in_page) * dim_);
        }

        const float *GetPage(std::uint32_t page_idx) const
        {
            if (page_idx >= pages_.size())
                return nullptr;
            return pages_[page_idx].get();
        }

        void SetIds(std::vector<VectorId> &&ids) { ids_ = std::move(ids); }

        void AllocatePages(std::size_t n_pages)
        {
            while (pages_.size() < n_pages)
            {
                float *new_page = new float[vectors_per_page_ * dim_];
                pages_.push_back(std::unique_ptr<float[]>(new_page));
            }
        }

        float *GetMutablePage(std::uint32_t page_idx)
        {
            if (page_idx >= pages_.size())
                return nullptr;
            return pages_[page_idx].get();
        }

        std::size_t Size() const { return ids_.size(); }
        std::uint32_t Dim() const { return dim_; }
        std::size_t NumPages() const { return pages_.size(); }
        std::size_t VectorsPerPage() const { return vectors_per_page_; }

        const std::vector<VectorId> &Ids() const { return ids_; }

    private:
        std::uint32_t dim_;
        std::size_t vectors_per_page_;
        std::vector<VectorId> ids_;
        std::vector<std::unique_ptr<float[]>> pages_;
    };

    class Shard final
    {
    public:
        explicit Shard(std::uint32_t shard_id, ShardOptions opt);

        Shard(const Shard &) = delete;
        Shard &operator=(const Shard &) = delete;

        pomai::Status Start();
        pomai::Status ApplyUpsert(std::vector<pomai::UpsertItem> &&items);
        SearchReply ExecuteSearch(const SearchRequest &req);
        PayloadBatchReply GetPayloadBatch(const std::vector<pomai::VectorId> &ids);

        pomai::Status Flush();
        void Stop();

        // Step2: manifest-based checkpoint.
        pomai::Status CreateCheckpoint();

        void MaybePublishSnapshot();
        std::shared_ptr<const ReadSnapshot> GetSnapshot() const;

        std::uint32_t ShardId() const { return shard_id_; }
        std::uint64_t UpsertCount() const { return upsert_count_.load(std::memory_order_relaxed); }
        std::uint64_t SearchCount() const { return search_count_.load(std::memory_order_relaxed); }

        const LatencyWindow &UpsertLatencyWindow() const { return upsert_lat_us_; }
        const LatencyWindow &SearchLatencyWindow() const { return search_lat_us_; }
        const WalWriter &Wal() const { return wal_; }

        // Step3 scheduling signals
        std::uint64_t WalBytes() const;
        std::uint64_t MsSinceLastCheckpoint() const;

    private:
        // Legacy single-file recovery (old layout).
        pomai::Status RecoverLegacySingleWal();

        // Step2 multi-file recovery using MANIFEST + wal rotation.
        pomai::Status RecoverFromManifest(const Manifest &m);

        pomai::Status ReplayPayload(const std::vector<std::byte> &payload);

        pomai::Status SaveSnapshot(const std::filesystem::path &path, std::uint64_t checkpoint_seq);
        pomai::Status LoadSnapshot(const std::filesystem::path &path, std::uint64_t *out_checkpoint_seq);

        // Layout helpers
        std::filesystem::path MetaDir() const;
        std::filesystem::path WalDir() const;
        std::filesystem::path CheckpointsDir() const;
        std::filesystem::path ManifestPath() const;

        pomai::Status OpenActiveWal(std::uint64_t wal_id);
        void PruneOldWalFiles(std::uint64_t keep_from_id);

        static std::uint64_t NowSteadyNs();

        std::unique_ptr<pomai::index::HnswIndex> index_;
        std::unique_ptr<BlobStore> blob_store_;

        const std::uint32_t shard_id_;
        const ShardOptions opt_;

        WalWriter wal_;
        std::uint64_t active_wal_id_{0};

        std::unique_ptr<PagedVectorArena> arena_;

        std::atomic<std::shared_ptr<const ReadSnapshot>> published_;
        std::uint64_t ops_since_publish_{0};

        // Kept for metrics; Step3 scheduler triggers checkpoint, not inline.
        std::uint64_t ops_since_checkpoint_{0};

        std::atomic<std::uint64_t> upsert_count_{0};
        std::atomic<std::uint64_t> search_count_{0};

        LatencyWindow upsert_lat_us_{4096};
        LatencyWindow search_lat_us_{4096};

        // Step3: scheduler uses this to enforce min interval
        std::atomic<std::uint64_t> last_checkpoint_time_ns_{0};
    };

} // namespace pomai::core
