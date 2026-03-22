#include "core/query/query_planner.h"
#include "core/graph/bitset_frontier.h" 
#include "pomai/search.h"
#include "pomai/graph.h"
#include "pomai/types.h"

namespace pomai::core {

Status QueryPlanner::Execute(std::string_view membrane, const MultiModalQuery& query, SearchResult* out) {
    if (!engine_ || !out) return Status::InvalidArgument("invalid args");

    SearchOptions opts;
    if (!query.vector.empty()) {
        heuristic_ai_.OptimizeSearchOptions(query.vector, &opts);
    }
    
    // Apply Temporal Constraints
    if (query.start_ts > 0 || query.end_ts > 0) {
        opts.filters.push_back(pomai::Filter::TimeRange(query.start_ts, query.end_ts));
    }

    bool has_vector = !query.vector.empty();
    bool has_lexical = !query.keywords.empty();

    if (has_vector && has_lexical) {
        // Hybrid Path (RRF)
        SearchResult v_res;
        std::vector<LexicalHit> l_hits;
        auto st_v = engine_->Search(membrane, query.vector, query.top_k * 4, opts, &v_res); 
        auto st_l = engine_->SearchLexical(membrane, query.keywords, query.top_k * 4, &l_hits);

        std::unordered_map<VectorId, float> rrf_scores;
        auto add_rrf = [&](VectorId id, uint32_t rank, float weight) {
            rrf_scores[id] += weight * (1.0f / (60.0f + static_cast<float>(rank)));
        };

        for (uint32_t i = 0; i < v_res.hits.size(); ++i) add_rrf(v_res.hits[i].id, i, query.alpha);
        for (uint32_t i = 0; i < l_hits.size(); ++i) add_rrf(l_hits[i].id, i, 1.0f - query.alpha);

        std::vector<std::pair<VectorId, float>> sorted(rrf_scores.begin(), rrf_scores.end());
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b){ return a.second > b.second; });

        for (uint32_t i = 0; i < std::min<uint32_t>(query.top_k, sorted.size()); ++i) {
            SearchHit hit;
            hit.id = sorted[i].first;
            hit.score = sorted[i].second;
            out->hits.push_back(std::move(hit));
        }
    } else if (has_lexical) {
        std::vector<LexicalHit> l_hits;
        auto st = engine_->SearchLexical(membrane, query.keywords, query.top_k, &l_hits);
        if (!st.ok()) return st;
        for (const auto& h : l_hits) {
            SearchHit hit;
            hit.id = h.id;
            hit.score = h.score;
            out->hits.push_back(std::move(hit));
        }
    } else {
        auto st = engine_->Search(membrane, query.vector, query.top_k, opts, out);
        if (!st.ok()) return st;
    }

    // 3. Graph Traversal (Context Amplification)
    for (auto& hit : out->hits) {
        BitsetFrontier frontier(1024); 
        BitsetFrontier seen(1024);
        frontier.Set(hit.id);
        seen.Set(hit.id);

        for (uint32_t h = 0; h < query.graph_hops; ++h) {
            BitsetFrontier next(1024);
            std::vector<VertexId> current_ids = frontier.ToIds();
            for (VertexId vid : current_ids) {
                std::vector<Neighbor> nb;
                Status gst;
                if (query.edge_type != 0) {
                    gst = engine_->GetNeighbors(membrane, vid, query.edge_type, &nb);
                } else {
                    gst = engine_->GetNeighbors(membrane, vid, &nb);
                }

                if (gst.ok()) {
                    float total_rank = 0;
                    for (const auto& n : nb) total_rank += n.rank;
                    float avg_rank = (nb.size() > 0) ? (total_rank / nb.size()) : 0.0f;

                    if (!heuristic_ai_.ShouldExpand(vid, h, query.graph_hops, avg_rank, static_cast<uint32_t>(nb.size()))) continue;

                    for (const auto& n : nb) {
                        if (!seen.IsSet(n.id)) {
                            next.Set(n.id);
                            seen.Set(n.id);
                            hit.related_ids.push_back(n.id);
                        }
                    }
                }
            }
            if (next.IsEmpty()) break;
            frontier = std::move(next);
        }
    }

    return Status::Ok();
}

} // namespace pomai::core
