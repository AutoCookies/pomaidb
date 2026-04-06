#include "core/query/query_orchestrator.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "core/query/heuristic_engine.h"
#include "core/security/key_manager.h"
#include "core/linker/object_linker.h"
#include "pomai/graph.h"
#include "pomai/metadata.h"
#include "pomai/search.h"

namespace pomai::core {

namespace {
std::string ResolveMembrane(std::string_view fallback, const std::string& override_name) {
    return override_name.empty() ? std::string(fallback) : override_name;
}
}  // namespace

Status QueryOrchestrator::Execute(std::string_view default_membrane, const pomai::MultiModalQuery& query, pomai::SearchResult* out) {
    if (!out || !engine_) return Status::InvalidArgument("invalid query orchestrator args");

    const std::string vec_mem = ResolveMembrane(default_membrane, query.vector_membrane);
    const std::string graph_mem = ResolveMembrane(default_membrane, query.graph_membrane);
    const std::string text_mem = ResolveMembrane(default_membrane, query.text_membrane);

    HeuristicEngine heuristic;
    if (heuristic.IsAnomalousAccessPattern(query)) {
        // Edge self-destruct primitive: revoke in-memory keys immediately.
        KeyManager::Global().Wipe();
        return Status::Aborted("anomalous query pattern detected; keys wiped");
    }
    const auto order = heuristic.DecideExecutionOrder(query);

    // Fast path: single membrane, delegate to existing planner.
    if (vec_mem == graph_mem) {
        return planner_.Execute(vec_mem, query, out);
    }

    // Cross-membrane path: run vector/lexical retrieval on vec membrane then
    // graph expansion on graph membrane (or inverse based on heuristic).
    out->Clear();
    SearchOptions opts;
    if (!query.vector.empty()) {
        heuristic.OptimizeSearchOptions(query.vector, &opts);
    }
    if (query.start_ts > 0 || query.end_ts > 0) {
        opts.filters.push_back(pomai::Filter::TimeRange(query.start_ts, query.end_ts));
    }
    opts.as_of_ts = query.as_of_ts;
    opts.as_of_lsn = query.as_of_lsn;
    opts.partition_device_id = query.partition_device_id;
    opts.partition_location_id = query.partition_location_id;
    // Light prefilter hinting: if caller provides geo/bitset/sparse gate hints,
    // narrow initial frontier to reduce downstream RAM/cpu before graph expansion.
    uint32_t effective_topk = query.top_k;
    if (query.prefilter_radius_m > 0.0 || query.prefilter_bitset_id != 0 || query.prefilter_sparse_id != 0) {
        effective_topk = std::max<uint32_t>(1u, query.top_k / 2u);
    }

    if (order == pomai::QueryExecutionOrder::kGraphFirst) {
        // Probe graph first using a lightweight lexical fallback if present.
        if (!query.keywords.empty()) {
            std::vector<LexicalHit> l_hits;
            auto st_l = engine_->SearchLexical(text_mem, query.keywords, effective_topk, &l_hits);
            if (st_l.ok()) {
                for (const auto& h : l_hits) {
                    SearchHit hit;
                    hit.id = h.id;
                    hit.score = h.score;
                    out->hits.push_back(std::move(hit));
                }
            }
        }
        if (out->hits.empty() && !query.vector.empty()) {
            auto st = engine_->Search(vec_mem, query.vector, effective_topk, opts, out);
            if (!st.ok()) return st;
        }
    } else {
        if (!query.vector.empty()) {
            auto st = engine_->Search(vec_mem, query.vector, effective_topk, opts, out);
            if (!st.ok()) return st;
        }
        if (out->hits.empty() && !query.keywords.empty()) {
            std::vector<LexicalHit> l_hits;
            auto st_l = engine_->SearchLexical(text_mem, query.keywords, effective_topk, &l_hits);
            if (!st_l.ok()) return st_l;
            for (const auto& h : l_hits) {
                SearchHit hit;
                hit.id = h.id;
                hit.score = h.score;
                out->hits.push_back(std::move(hit));
            }
        }
    }

    // Graph expansion on graph membrane.
    for (auto& hit : out->hits) {
        if (const auto linked = engine_->ResolveLinkedByVectorId(hit.id); linked.has_value()) {
            hit.related_ids.push_back(linked->graph_vertex_id);
            hit.related_ids.push_back(linked->mesh_id);
        }
        std::vector<VertexId> frontier{hit.id};
        std::unordered_set<VertexId> seen{hit.id};
        for (uint32_t hop = 0; hop < query.graph_hops; ++hop) {
            std::vector<VertexId> next;
            for (auto vid : frontier) {
                std::vector<pomai::Neighbor> neighbors;
                auto st = (query.edge_type != 0)
                              ? engine_->GetNeighbors(graph_mem, vid, query.edge_type, &neighbors)
                              : engine_->GetNeighbors(graph_mem, vid, &neighbors);
                if (!st.ok()) continue;
                for (const auto& n : neighbors) {
                    if (!seen.count(n.id)) {
                        seen.insert(n.id);
                        next.push_back(n.id);
                        hit.related_ids.push_back(n.id);
                    }
                }
            }
            if (next.empty()) break;
            if (next.size() > max_frontier_) {
                next.resize(max_frontier_);
            }
            frontier.swap(next);
        }
    }

    return Status::Ok();
}

} // namespace pomai::core

