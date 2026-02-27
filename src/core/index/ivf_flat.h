#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <span>
#include <memory>
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::index {

// Persistent IVF Index (Centroids + Inverted Lists)
// Used as a sidecar (.idx) for Segments.
class IvfFlatIndex {
public:
    struct Options {
        uint32_t nlist = 64;
    };

    IvfFlatIndex(uint32_t dim, Options opt);
    ~IvfFlatIndex();

    // ---- Build Phase ----
    // Train centroids using sampled vectors
    pomai::Status Train(std::span<const float> data, size_t num_vectors);
    
    // Add vector to index (must be trained). Appends to inverted list.
    pomai::Status Add(uint32_t entry_index, std::span<const float> vec);

    // ---- Query Phase ----
    // Select candidates by probing nearest centroids.
    // Returns list of VectorIds.
    pomai::Status Search(std::span<const float> query, uint32_t nprobe, 
                         std::vector<uint32_t>* out_candidates) const;

    // ---- IO Phase ----
    // Save to file
    pomai::Status Save(const std::string& path) const;
    
    // Load from file
    static pomai::Status Load(const std::string& path, std::unique_ptr<IvfFlatIndex>* out);

    // Metadata
    uint32_t dim() const { return dim_; }
    uint32_t nlist() const { return opt_.nlist; }
    size_t total_vectors() const { return total_count_; }

private:
    uint32_t dim_;
    Options opt_;
    size_t total_count_ = 0;
    bool trained_ = false;

    // Centroids: nlist * dim (Row-major)
    std::vector<float> centroids_;
    
    // Inverted Lists: nlist vectors of local indices
    std::vector<std::vector<uint32_t>> lists_;

    // Helpers
    uint32_t FindNearestCentroid(std::span<const float> vec) const;
};

} // namespace pomai::index
