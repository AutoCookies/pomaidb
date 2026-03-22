#pragma once
#include <vector>
#include <span>
#include <string_view>
#include "pomai/options.h"
#include "pomai/types.h"

namespace pomai {
    struct SearchOptions;
}

namespace pomai::core {

/**
 * @brief HeuristicEngine provides rule-based "Intelligence" for query optimization.
 * It uses pure calculation (no training) to enhance edge performance.
 */
class HeuristicEngine {
public:
    HeuristicEngine() = default;

    /**
     * @brief Suggests optimal search parameters (ef_search, nprobe) based on query vector stats.
     */
    void OptimizeSearchOptions(std::span<const float> query, pomai::SearchOptions* options);

    /**
     * @brief Evaluates the "Density" of a graph neighborhood to prune expansion.
     * Now includes degree and temporal signal.
     */
    bool ShouldExpand(VertexId current, uint32_t current_hop, uint32_t max_hops, 
                      float rank_avg, uint32_t degree);

    /**
     * @brief Calculates a bias score based on vertex degree (Hub-ness).
     */
    float CalculateCentralityBias(uint32_t degree);

    /**
     * @brief Estimates relevance based on edge timestamps or ranks.
     */
    float EstimateTemporalRelevance(float rank_avg);

private:
    float CalculateVariance(std::span<const float> vec);
};

} // namespace pomai::core
