#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pomai/database.h"
#include "pomai/options.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai
{

    // Lightweight, edge-focused options for AgentMemory.
    struct AgentMemoryOptions
    {
        // Filesystem path for the embedded Database backing this AgentMemory.
        std::string path;

        // Embedding dimension (must match all stored embeddings).
        std::uint32_t dim = 512;

        // Distance metric to use for similarity search.
        MetricType metric = MetricType::kL2;

        // Memtable backpressure configuration. 0 = derive from defaults/env.
        std::uint32_t max_memtable_mb = 0;
        std::uint32_t memtable_flush_threshold_mb = 64;
        bool auto_freeze_on_pressure = true;

        // Optional soft limits to keep memory bounded on edge devices.
        // 0 disables a particular limit.
        std::size_t max_messages_per_agent = 0;
        std::size_t max_device_bytes = 0;
    };

    enum class AgentMemoryKind : std::uint8_t
    {
        kMessage = 0,
        kSummary = 1,
        kKnowledge = 2,
    };

    struct AgentMemoryRecord
    {
        std::string agent_id;
        std::string session_id;
        AgentMemoryKind kind = AgentMemoryKind::kMessage;
        std::int64_t logical_ts = 0; // Monotonic per-agent timestamp supplied by caller.
        std::string text;
        std::vector<float> embedding;
    };

    struct AgentMemoryQuery
    {
        // Required: which agent to search within.
        std::string agent_id;

        // Optional additional filters.
        std::string session_id;  // empty = any session
        bool has_session_filter = false;

        AgentMemoryKind kind = AgentMemoryKind::kMessage;
        bool has_kind_filter = false;

        std::int64_t min_ts = std::numeric_limits<std::int64_t>::min();
        std::int64_t max_ts = std::numeric_limits<std::int64_t>::max();

        // Embedding to search with. Must have size == options.dim.
        std::vector<float> embedding;
        std::uint32_t topk = 10;
    };

    struct AgentMemoryHit
    {
        AgentMemoryRecord record;
        float score = 0.0f;
    };

    struct AgentMemorySearchResult
    {
        std::vector<AgentMemoryHit> hits;

        void Clear()
        {
            hits.clear();
        }
    };

    // AgentMemory: thin, thread-safe facade over a single embedded Database instance
    // providing higher-level APIs for agent context and semantic memory.
    //
    // Thread-safety:
    //  - All public methods are internally synchronized; the underlying Database
    //    remains single-threaded and is never accessed concurrently.
    class AgentMemory
    {
    public:
        AgentMemory(const AgentMemory&) = delete;
        AgentMemory& operator=(const AgentMemory&) = delete;

        ~AgentMemory();

        // Open (or create) an AgentMemory instance backed by an embedded Database.
        // On success, *out will own the new AgentMemory.
        static Status Open(const AgentMemoryOptions& options,
                           std::unique_ptr<AgentMemory>* out);

        // Append a single memory record. On success, out_id (if non-null) will
        // receive the assigned VectorId.
        Status AppendMessage(const AgentMemoryRecord& record, VectorId* out_id);

        // Append a batch of memory records with a single ingestion path.
        Status AppendBatch(const std::vector<AgentMemoryRecord>& records,
                           std::vector<VectorId>* out_ids);

        // Retrieve the most recent messages for a given agent/session.
        Status GetRecent(std::string_view agent_id,
                         std::string_view session_id,
                         std::size_t limit,
                         std::vector<AgentMemoryRecord>* out);

        // Semantic similarity search within an agent's memory.
        Status SemanticSearch(const AgentMemoryQuery& query,
                              AgentMemorySearchResult* out);

        // Prune old messages for a given agent, keeping at most keep_last_n and
        // never deleting entries with logical_ts >= min_ts_to_keep.
        Status PruneOld(std::string_view agent_id,
                        std::size_t keep_last_n,
                        std::int64_t min_ts_to_keep);

        // Best-effort device-wide prune to approximate the given byte budget.
        Status PruneDeviceWide(std::size_t target_total_bytes);

        // Delegate to Database::TryFreezeIfPressured to avoid unbounded growth
        // of the active memtable on edge devices.
        Status FreezeIfNeeded();

        std::uint32_t dim() const noexcept { return dim_; }

        // Internal constructor used by Open(); must be public for std::make_unique.
        AgentMemory(AgentMemoryOptions opts, std::unique_ptr<Database> db);

    private:

        Status EnsureOpen() const;

        // Encode/decode helper for mapping AgentMemoryRecord into Metadata.tenant.
        static std::string EncodeMetadata(const AgentMemoryRecord& record);
        static bool DecodeMetadata(const std::string& encoded, AgentMemoryRecord* out_record);

        // Scan full database to approximate total bytes and collect per-record
        // summaries for pruning. Implemented with snapshot + iterator and
        // point Gets to avoid additional in-RAM indexes.
        struct PruneEntry
        {
            VectorId id;
            std::string agent_id;
            std::string session_id;
            AgentMemoryKind kind;
            std::int64_t logical_ts;
            std::size_t approx_bytes;
        };

        Status CollectPruneEntries(std::vector<PruneEntry>* out_entries) const;

        // Core search helper used by SemanticSearch.
        Status SearchAndFilter(const AgentMemoryQuery& query,
                                     AgentMemorySearchResult* out);

        AgentMemoryOptions options_;
        std::unique_ptr<Database> db_;
        std::uint32_t dim_ = 0;
        bool opened_ = false;
        VectorId next_id_ = 1;
    };

} // namespace pomai

