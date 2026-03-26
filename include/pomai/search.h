#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "options.h"
#include "types.h"
#include "graph.h"

namespace pomai
{
    enum class AggregateOp : uint8_t {
        kNone = 0,
        kSum = 1,
        kAvg = 2,
        kMin = 3,
        kMax = 4,
        kCount = 5,
        kTopK = 6,
    };

    struct AggregateRequest {
        AggregateOp op = AggregateOp::kNone;
        std::string field = "score"; // score|timestamp|lsn
        uint32_t top_k = 0;
    };

    enum class QueryExecutionOrder : uint8_t {
        kAuto = 0,
        kVectorFirst = 1,
        kGraphFirst = 2,
    };

    struct MeshQueryOptions {
        MeshDetailPreference detail = MeshDetailPreference::kAutoLatencyFirst;
    };


    struct SearchHit
    {
        VectorId id = 0;
        float score = 0.0f; // higher is better
        std::vector<uint64_t> related_ids; // K-hop neighbors
    };

    /**
     * @brief Interface for collecting search results without intermediate allocations.
     * Concrete implementations can write to std::vector, a pre-allocated pool, or directly to a C-buffer.
     */
    class SearchHitSink {
    public:
        virtual ~SearchHitSink() = default;
        /** Collect one search hit. Implementation may decide to keep or discard based on top-K or filters. */
        virtual void Push(VectorId id, float score) = 0;
    };

    /** @brief Priority queue comparator for min-heap (lowest score at top). */
    struct WorseHit {
        bool operator()(const SearchHit& a, const SearchHit& b) const {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.id > b.id;
        }
    };

    /** @brief Check if 'a' is a better hit than 'b' (higher score, then lower ID). */
    inline bool IsBetterHit(const SearchHit& a, const SearchHit& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.id < b.id;
    }

    struct AggregateResult {
        AggregateOp op = AggregateOp::kNone;
        std::string field;
        double value = 0.0;
        std::vector<SearchHit> topk_hits;
    };

    struct ShardError
    {
        uint32_t shard_id;
        std::string message;
    };

    struct SemanticPointer {
        const void* raw_data_ptr = nullptr;
        uint32_t dim = 0;
        float quant_min = 0.0f;
        float quant_inv_scale = 0.0f;
        int quant_type = 0; // 0=None, 1=SQ8, 2=FP16
        uint64_t session_id = 0;
    };

    struct SearchResult
    {
        std::vector<SearchHit> hits;
        std::vector<AggregateResult> aggregates;
        std::vector<ShardError> errors; // Partial failures
        uint32_t routed_shards_count = 0;
        uint32_t total_shards_count = 0;
        uint32_t pruned_shards_count = 0;
        uint32_t routing_probe_centroids = 0;
        uint64_t routed_buckets_count = 0; // Candidate/bucket count when routing enabled.

        std::vector<SemanticPointer> zero_copy_pointers;
        uint64_t zero_copy_session_id = 0;
        
        std::vector<Neighbor> neighbors;

        void Clear() {
            hits.clear();
            aggregates.clear();
            errors.clear();
            neighbors.clear();
            routed_shards_count = 0;
            total_shards_count = 0;
            pruned_shards_count = 0;
            routing_probe_centroids = 0;
            routed_buckets_count = 0;
            zero_copy_pointers.clear();
            zero_copy_session_id = 0;
        }
    };

    /**
     * @brief Unified Multi-modal Query specification.
     */
    struct MultiModalQuery {
        std::vector<float> vector;      // Semantic query vector (Dense)
        std::string keywords;           // Keyword query string (Lexical/BM25)
        float alpha = 0.5f;             // Weight: alpha*Vector + (1-alpha)*Lexical
        
        uint32_t top_k = 10;            
        uint32_t graph_hops = 2;        
        EdgeType edge_type = 0;         

        // Temporal Filtering
        uint64_t start_ts = 0;
        uint64_t end_ts = 0;
        uint64_t as_of_ts = 0;
        uint64_t as_of_lsn = 0;

        // Cross-membrane routing (empty => use call-site membrane).
        std::string vector_membrane;
        std::string graph_membrane;
        std::string text_membrane;

        // Orchestrator execution policy.
        QueryExecutionOrder execution_order = QueryExecutionOrder::kAuto;
        
        // Potential future filters
        std::string filter_expression;
        std::string partition_device_id;
        std::string partition_location_id;
        std::vector<AggregateRequest> aggregates;
        // Optional spatial prefilter: center + radius meters.
        double prefilter_lat = 0.0;
        double prefilter_lon = 0.0;
        double prefilter_radius_m = 0.0;
        // Optional bitset/sparse prefilter hints.
        uint64_t prefilter_bitset_id = 0;
        uint64_t prefilter_sparse_id = 0;
        MeshDetailPreference mesh_detail_preference = MeshDetailPreference::kAutoLatencyFirst;
    };

} // namespace pomai
