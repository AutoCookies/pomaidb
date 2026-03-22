#pragma once
#include <cstdint>
#include "pomai/types.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace pomai
{
    /**
     * @brief Metadata associated with a vector or vertex.
     */
    struct Metadata
    {
        std::string tenant;   // Primary use case: multi-tenancy filtering
        VertexId src_vid = 0; // Source vertex ID for automatic linkage (0 = none)
        
        Metadata() = default;
        explicit Metadata(std::string t, VertexId s = 0) 
            : tenant(std::move(t)), src_vid(s) {}
        
        bool operator==(const Metadata& other) const {
            return tenant == other.tenant && src_vid == other.src_vid;
        }
    };
    
    struct Filter
    {
        std::string field;
        std::string value;
        
        Filter() = default;
        Filter(std::string f, std::string v) 
            : field(std::move(f)), value(std::move(v)) {}
        
        bool Matches(const Metadata& meta) const {
            if (field == "tenant") {
                return meta.tenant == value;
            }
            return true;
        }
    };
    
    struct SearchOptions
    {
        std::vector<Filter> filters;
        bool force_fanout = false;
        uint32_t routing_probe_override = 0;
        bool zero_copy = false;
        
        SearchOptions() = default;
        
        bool Matches(const Metadata& meta) const {
            for (const auto& filter : filters) {
                if (!filter.Matches(meta)) {
                    return false;
                }
            }
            return true;
        }
    };

} // namespace pomai
