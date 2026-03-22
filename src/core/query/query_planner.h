#pragma once
#include <vector>
#include <string_view>
#include <span>
#include "pomai/status.h"
#include "pomai/types.h"
#include "core/query/heuristic_engine.h"

#include "core/query/lexical_index.h"

namespace pomai {
    class SearchResult;
    struct SearchOptions;
    struct MultiModalQuery;
    struct Neighbor;
}

namespace pomai::core {

class IQueryEngine {
public:
    virtual ~IQueryEngine() = default;
    virtual Status Search(std::string_view membrane, std::span<const float> query, uint32_t topk, const pomai::SearchOptions& opts, pomai::SearchResult* out) = 0;
    
    // V7: Lexical Search support
    virtual Status SearchLexical(std::string_view membrane, const std::string& query, uint32_t topk, std::vector<LexicalHit>* out) = 0;
    virtual Status GetNeighbors(std::string_view membrane, VertexId src, std::vector<pomai::Neighbor>* out) = 0;
    virtual Status GetNeighbors(std::string_view membrane, VertexId src, EdgeType type, std::vector<pomai::Neighbor>* out) = 0;
};

class QueryPlanner {
public:
    explicit QueryPlanner(IQueryEngine* engine) : engine_(engine) {}

    Status Execute(std::string_view membrane, const pomai::MultiModalQuery& query, pomai::SearchResult* out);

private:
    IQueryEngine* engine_;
    HeuristicEngine heuristic_ai_;
};

} // namespace pomai::core
