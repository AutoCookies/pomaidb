#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "pomai/types.h"
#include "pomai/graph.h"
#include "pomai/slice.h"

#include <endian.h>
#define POMAI_BE64(x) htobe64(x)
#define POMAI_BE32(x) htobe32(x)

namespace pomai::core {

/**
 * @brief Utilities for Graph Key serialization.
 */
class GraphKey {
public:
    enum Type : uint8_t {
        kVertex = 'V',
        kEdge = 'E'
    };

    /**
     * @brief Serializes a VertexKey.
     * Format: V (1) | VertexID (8) | TagID (4)
     */
    static std::string EncodeVertex(VertexId vid, TagId tag) {
        std::string buf;
        buf.reserve(13);
        buf.push_back(kVertex);
        uint64_t v_be = POMAI_BE64(vid);
        buf.append(reinterpret_cast<const char*>(&v_be), 8);
        uint32_t t_be = POMAI_BE32(tag);
        buf.append(reinterpret_cast<const char*>(&t_be), 4);
        return buf;
    }

    /**
     * @brief Serializes an EdgeKey.
     * Format: E (1) | SrcID (8) | EdgeType (4) | Rank (4) | DstID (8)
     */
    static std::string EncodeEdge(VertexId src, EdgeType type, uint32_t rank, VertexId dst) {
        std::string buf;
        buf.reserve(25);
        buf.push_back(kEdge);
        uint64_t s_be = POMAI_BE64(src);
        buf.append(reinterpret_cast<const char*>(&s_be), 8);
        uint32_t t_be = POMAI_BE32(type);
        buf.append(reinterpret_cast<const char*>(&t_be), 4);
        uint32_t r_be = POMAI_BE32(rank);
        buf.append(reinterpret_cast<const char*>(&r_be), 4);
        uint64_t d_be = POMAI_BE64(dst);
        buf.append(reinterpret_cast<const char*>(&d_be), 8);
        return buf;
    }

    /**
     * @brief Prefix for all edges of a vertex.
     */
    static std::string EncodeEdgePrefix(VertexId src) {
        std::string buf;
        buf.reserve(9);
        buf.push_back(kEdge);
        uint64_t s_be = POMAI_BE64(src);
        buf.append(reinterpret_cast<const char*>(&s_be), 8);
        return buf;
    }
};

} // namespace pomai::core
