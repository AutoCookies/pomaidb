#pragma once
#include <cstdint>
#include "pomai/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pomai
{
    /**
     * @brief Metadata associated with a vector or vertex.
     */
    struct Metadata
    {
        std::string tenant;   // Primary use case: multi-tenancy filtering
        VertexId src_vid = 0; // Source vertex ID for automatic linkage (0 = none)
        uint64_t timestamp = 0; // Timestamp for temporal queries (0 = none/epoch)
        std::string text; // For Lexical Search (Keyword Index)
        std::string payload; // Arbitrary Document/JSON data
        double lat = 0.0;
        double lon = 0.0;
        
        Metadata() = default;
        explicit Metadata(std::string t, VertexId s = 0, uint64_t ts = 0, std::string txt = "", std::string p = "", double lt = 0.0, double ln = 0.0) 
            : tenant(std::move(t)), src_vid(s), timestamp(ts), text(std::move(txt)), payload(std::move(p)), lat(lt), lon(ln) {}
        
        bool operator==(const Metadata& other) const {
            return tenant == other.tenant && src_vid == other.src_vid && 
                   timestamp == other.timestamp && text == other.text && 
                   payload == other.payload && lat == other.lat && lon == other.lon;
        }
    };
    
    struct Filter
    {
        std::string field;
        std::string value;

        // Temporal Range
        uint64_t min_ts = 0;
        uint64_t max_ts = 0;

        // Spatial Fields
        double spatial_lat = 0.0;
        double spatial_lon = 0.0;
        double spatial_radius = 0.0; // In kilometers
        bool is_radius_query = false;
        
        Filter() = default;
        Filter(std::string f, std::string v) 
            : field(std::move(f)), value(std::move(v)) {}
        
        static Filter TimeRange(uint64_t min, uint64_t max) {
            Filter f;
            f.field = "timestamp";
            f.min_ts = min;
            f.max_ts = max;
            return f;
        }

        static Filter Radius(double lat, double lon, double radius_km) {
            Filter f;
            f.field = "spatial";
            f.spatial_lat = lat;
            f.spatial_lon = lon;
            f.spatial_radius = radius_km;
            f.is_radius_query = true;
            return f;
        }

        bool Matches(const Metadata& meta) const {
            if (field == "tenant" && meta.tenant != value) return false;
            
            if (field == "timestamp") {
                if (min_ts > 0 && meta.timestamp < min_ts) return false;
                if (max_ts > 0 && meta.timestamp > max_ts) return false;
            }

            if (field == "spatial" && is_radius_query) {
                // Approximate distance calculation (Equirectangular) for edge performance
                // For high-precision: use Haversine.
                // Convert degrees to radians for cos function
                const double DEG_TO_RAD = M_PI / 180.0;
                double lat_rad_meta = meta.lat * DEG_TO_RAD;
                double lon_rad_meta = meta.lon * DEG_TO_RAD;
                double lat_rad_spatial = spatial_lat * DEG_TO_RAD;
                double lon_rad_spatial = spatial_lon * DEG_TO_RAD;

                double x = (lon_rad_meta - lon_rad_spatial) * std::cos((lat_rad_meta + lat_rad_spatial) / 2.0);
                double y = (lat_rad_meta - lat_rad_spatial);
                // Earth's radius in kilometers (mean radius)
                const double EARTH_RADIUS_KM = 6371.0; 
                double dist_km = std::sqrt(x*x + y*y) * EARTH_RADIUS_KM;
                
                if (dist_km > spatial_radius) return false;
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
