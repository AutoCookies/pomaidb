#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace pomai
{
    // Simplified metadata: single string field for MVP
    // Can be extended to multi-field later
    struct Metadata
    {
        std::string tenant; // Primary use case: multi-tenancy filtering
        
        Metadata() = default;
        explicit Metadata(std::string t) : tenant(std::move(t)) {}
        
        bool operator==(const Metadata& other) const {
            return tenant == other.tenant;
        }
    };
    
    // Filter predicate (simplified: EQ only for MVP)
    struct Filter
    {
        std::string field;  // For MVP, only "tenant" is supported
        std::string value;  // Expected value
        
        Filter() = default;
        Filter(std::string f, std::string v) 
            : field(std::move(f)), value(std::move(v)) {}
        
        // Evaluate filter against metadata
        bool Matches(const Metadata& meta) const {
            if (field == "tenant") {
                return meta.tenant == value;
            }
            return true; // Unknown fields don't filter
        }
    };
    
    // Search options with filters
    struct SearchOptions
    {
        std::vector<Filter> filters; // AND semantics
        
        SearchOptions() = default;
        
        // Check if metadata passes all filters
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
