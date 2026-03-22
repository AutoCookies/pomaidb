#include "core/query/query_planner.h"
#include "core/graph/bitset_frontier.h" 
#include "pomai/search.h"
#include "pomai/graph.h"
#include "pomai/types.h"

namespace pomai::core {

Status QueryPlanner::Execute(std::string_view membrane, const MultiModalQuery& query, SearchResult* out) {
    if (!engine_ || !out) return Status::InvalidArgument("invalid args");

    // 1. Vector Search
    SearchOptions opts;
    auto st = engine_->Search(membrane, query.vector, query.top_k, opts, out);
    if (!st.ok()) return st;

    // 2. Graph Traversal (Context Amplification)
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
