#include "core/query/heuristic_engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include "pomai/metadata.h"

namespace pomai::core {

void HeuristicEngine::OptimizeSearchOptions(std::span<const float> query, pomai::SearchOptions* options) {
    if (!options || query.empty()) return;

    float variance = CalculateVariance(query);

    if (variance > 0.5f) {
        options->routing_probe_override = 16;
    } else if (variance < 0.1f) {
        options->routing_probe_override = 2;
    } else {
        options->routing_probe_override = 8;
    }

    if (query.size() > 1024) {
        options->zero_copy = true;
    }
}

bool HeuristicEngine::ShouldExpand(VertexId /*current*/, uint32_t current_hop, uint32_t max_hops, 
                                  float rank_avg, uint32_t degree) {
    // If unranked, allow basic expansion
    if (rank_avg == 0.0f && degree > 0) return true;

    // 1. Centrality Bias (Hub-ness)
    float centrality = CalculateCentralityBias(degree);
    
    // 2. Temporal/Rank Relevance
    float temporal = EstimateTemporalRelevance(rank_avg);
    
    // 3. Confidence Decay based on Hops
    float hop_confidence = (1.0f - (static_cast<float>(current_hop) / (static_cast<float>(max_hops) + 1.0f)));
    
    // Aggregate Score
    float total_score = hop_confidence * centrality * temporal;

    // Heuristic Threshold: If the aggregate signal is too weak, prune the branch.
    return total_score > 0.15f; 
}

float HeuristicEngine::CalculateCentralityBias(uint32_t degree) {
    if (degree == 0) return 0.5f;
    // Logarithmic scaling: high degree vertices are "hubs" that likely contain relevant cluster context.
    // Bias ranges from ~1.0 (degree 1) to ~2.0+ (high degree).
    return 1.0f + static_cast<float>(std::log2(1.0 + static_cast<double>(degree)) / 4.0);
}

float HeuristicEngine::EstimateTemporalRelevance(float rank_avg) {
    // Sigmoid-like relevance curve for ranks (0.0 to 1.0)
    // Stronger ranks (closer to 1.0) represent freshly linked or semantically tight context.
    return 1.0f / (1.0f + std::exp(-5.0f * (rank_avg - 0.4f)));
}

float HeuristicEngine::CalculateVariance(std::span<const float> vec) {
    if (vec.empty()) return 0.0f;
    float sum = 0.0f;
    for (float v : vec) sum += v;
    float mean = sum / vec.size();
    
    float sq_sum = 0.0f;
    for (float v : vec) {
        float diff = v - mean;
        sq_sum += diff * diff;
    }
    return sq_sum / vec.size();
}

} // namespace pomai::core
