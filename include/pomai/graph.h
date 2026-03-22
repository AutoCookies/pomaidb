#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <span>

#include "pomai/status.h"
#include "pomai/types.h"
#include "pomai/metadata.h"

namespace pomai {

/**
 * @brief Represents a neighbor in the graph.
 */
struct Neighbor {
    VertexId id;
    EdgeType type;
    uint32_t rank;
    // Potentially add weight or metadata pointer here
};

/**
 * @brief Public interface for Graph operations.
 */
class GraphMembrane {
public:
    virtual ~GraphMembrane() = default;

    /** Ingestion */
    virtual Status AddVertex(VertexId id, TagId tag, const Metadata& meta) = 0;
    virtual Status AddEdge(VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta) = 0;

    /** Query */
    virtual Status GetNeighbors(VertexId src, std::vector<Neighbor>* out) = 0;
    virtual Status GetNeighbors(VertexId src, EdgeType type, std::vector<Neighbor>* out) = 0;
    
    /** Maintenance */
    virtual Status Flush() = 0;
};

} // namespace pomai
