#include "core/index/ivf_flat.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <iostream>

#include "core/distance.h" // For Dot/L2Sq

namespace pomai::index {

namespace {
    // File Format Constants
    constexpr char kMagic[] = "POMAI_IVF_V1";
    constexpr size_t kMagicLen = 12; // including null or fixed size
    
    // Helper to write POD
    template<typename T>
    void WritePod(std::ofstream& out, const T& val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(T));
    }
    
    template<typename T>
    void ReadPod(std::ifstream& in, T& val) {
        in.read(reinterpret_cast<char*>(&val), sizeof(T));
    }
}

IvfFlatIndex::IvfFlatIndex(uint32_t dim, Options opt) 
    : dim_(dim), opt_(opt) {
    if (opt_.nlist == 0) opt_.nlist = 1;
    lists_.resize(opt_.nlist);
}

IvfFlatIndex::~IvfFlatIndex() = default;

uint32_t IvfFlatIndex::FindNearestCentroid(std::span<const float> vec) const {
    // Use DOT PRODUCT for assignment (max dot) as per requirement 
    // "Compute distances to centroids" in Coarse stage.
    // Assuming vectors are roughly normalized or standard angular distance.
    // If requirement implies L2 for assignment, we can swap.
    // "Phase 3: Coarse stage: Compute distances to centroids" -> ambiguous.
    // But since audit showed Dot product usage in rest of system, let's use Dot.
    
    float best_score = -std::numeric_limits<float>::infinity();
    uint32_t best_idx = 0;
    
    for (uint32_t i = 0; i < opt_.nlist; ++i) {
        std::span<const float> c(&centroids_[i * dim_], dim_);
        float score = pomai::core::Dot(vec, c);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }
    return best_idx;
}

pomai::Status IvfFlatIndex::Train(std::span<const float> data, size_t num_vectors) {
    if (num_vectors == 0) return pomai::Status::Ok();
    if (data.size() < num_vectors * dim_) 
        return pomai::Status::InvalidArgument("Data buffer too small");

    // Standard KMeans
    // Initialize centroids (Random sample)
    centroids_.resize(opt_.nlist * dim_);
    
    std::mt19937 rng(42); // Fixed seed for determinism
    std::uniform_int_distribution<size_t> dist(0, num_vectors - 1);
    
    for (uint32_t i = 0; i < opt_.nlist; ++i) {
        size_t idx = dist(rng);
        const float* src = &data[idx * dim_];
        float* dst = &centroids_[i * dim_];
        std::copy(src, src + dim_, dst);
    }

    // Iterations (Lloyd's)
    // Use L2 for clustering stability, even if assignment is Dot?
    // Let's use L2 for clustering to be safe standard KMeans.
    // Wait, if we use L2 for training but Dot for assignment, buckets might be suboptimal.
    // But Spherical KMeans is tricky without normalization.
    // Let's use L2.
    
    std::vector<uint32_t> assignments(num_vectors);
    std::vector<float> new_centroids(opt_.nlist * dim_);
    std::vector<uint32_t> counts(opt_.nlist);
    
    for (int iter = 0; iter < 10; ++iter) {
        // E-Step
        bool changed = false;
        for (size_t i = 0; i < num_vectors; ++i) {
            std::span<const float> vec(&data[i * dim_], dim_);
            
            float min_dist = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            
            for (uint32_t c = 0; c < opt_.nlist; ++c) {
                std::span<const float> cen(&centroids_[c * dim_], dim_);
                float d = pomai::core::L2Sq(vec, cen);
                if (d < min_dist) {
                    min_dist = d;
                    best_c = c;
                }
            }
            
            if (assignments[i] != best_c) changed = true;
            assignments[i] = best_c;
        }
        
        if (!changed && iter > 0) break;
        
        // M-Step
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        
        for (size_t i = 0; i < num_vectors; ++i) {
            uint32_t c = assignments[i];
            counts[c]++;
            const float* src = &data[i * dim_];
            float* dst = &new_centroids[c * dim_];
            for (uint32_t k = 0; k < dim_; ++k) {
                dst[k] += src[k];
            }
        }
        
        for (uint32_t c = 0; c < opt_.nlist; ++c) {
            if (counts[c] > 0) {
                float inv = 1.0f / counts[c];
                float* dst = &new_centroids[c * dim_];
                for (uint32_t k = 0; k < dim_; ++k) dst[k] *= inv;
            } else {
                // Re-init empty cluster? Keep old.
                std::copy(&centroids_[c * dim_], &centroids_[c * dim_] + dim_, &new_centroids[c * dim_]);
            }
        }
        centroids_ = new_centroids;
    }
    
    trained_ = true;
    return pomai::Status::Ok();
}

pomai::Status IvfFlatIndex::Add(pomai::VectorId id, std::span<const float> vec) {
    if (!trained_) return pomai::Status::Aborted("Index not trained");
    if (vec.size() != dim_) return pomai::Status::InvalidArgument("Dim mismatch");
    
    uint32_t c = FindNearestCentroid(vec);
    lists_[c].push_back(id);
    total_count_++;
    return pomai::Status::Ok();
}

pomai::Status IvfFlatIndex::Search(std::span<const float> query, uint32_t nprobe, 
                                   std::vector<pomai::VectorId>* out) const {
    if (!trained_) {
        // Fallback or empty? 
        // If not trained, we can't search via index.
        // Return Ok with empty lists -> caller must brute force (or handle fallback).
        return pomai::Status::Ok(); 
        // NOTE: In our design, "not trained" means index shouldn't exist.
    }
    
    // 1. Score Centroids (using Dot Product as per FindNearestCentroid)
    std::vector<std::pair<float, uint32_t>> scores;
    scores.reserve(opt_.nlist);
    
    for (uint32_t c = 0; c < opt_.nlist; ++c) {
        std::span<const float> cen(&centroids_[c * dim_], dim_);
        float s = pomai::core::Dot(query, cen);
        scores.push_back({s, c});
    }
    
    // 2. Select Top nprobe
    uint32_t K = std::min(nprobe, opt_.nlist);
    std::partial_sort(scores.begin(), scores.begin() + K, scores.end(), std::greater<>());
    
    // 3. Gather
    for (uint32_t k = 0; k < K; ++k) {
        uint32_t c = scores[k].second;
        const auto& lst = lists_[c];
        out->insert(out->end(), lst.begin(), lst.end());
    }
    
    return pomai::Status::Ok();
}

pomai::Status IvfFlatIndex::Save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return pomai::Status::Internal("Failed to open file for writing");
    
    // Header
    out.write(kMagic, kMagicLen);
    uint32_t version = 1;
    WritePod(out, version);
    WritePod(out, dim_);
    WritePod(out, opt_.nlist);
    
    size_t tc = total_count_;
    WritePod(out, tc);
    
    // Centroids
    out.write(reinterpret_cast<const char*>(centroids_.data()), centroids_.size() * sizeof(float));
    
    // Lists
    // Format: [ListSize(u32)] [IDs...] for each list
    for (const auto& lst : lists_) {
        uint32_t sz = static_cast<uint32_t>(lst.size());
        WritePod(out, sz);
        if (sz > 0) {
            out.write(reinterpret_cast<const char*>(lst.data()), lst.size() * sizeof(pomai::VectorId));
        }
    }
    
    if (!out) return pomai::Status::Internal("Write failed");
    return pomai::Status::Ok();
}

pomai::Status IvfFlatIndex::Load(const std::string& path, std::unique_ptr<IvfFlatIndex>* out) {
    if (!out) return pomai::Status::InvalidArgument("out is null");
    
    std::ifstream in(path, std::ios::binary);
    if (!in) return pomai::Status::NotFound("Index file not found");
    
    // Header
    char magic[kMagicLen];
    in.read(magic, kMagicLen);
    if (in.gcount() != kMagicLen || std::string(magic) != std::string(kMagic)) {
        return pomai::Status::Internal("Invalid index magic");
    }
    
    uint32_t version;
    ReadPod(in, version);
    if (version != 1) return pomai::Status::Internal("Unsupported version");
    
    uint32_t dim, nlist;
    ReadPod(in, dim);
    ReadPod(in, nlist);
    
    size_t total_count;
    ReadPod(in, total_count);
    
    Options opt;
    opt.nlist = nlist;
    auto idx = std::make_unique<IvfFlatIndex>(dim, opt);
    idx->total_count_ = total_count;
    idx->trained_ = true;
    
    // Centroids
    idx->centroids_.resize(nlist * dim);
    in.read(reinterpret_cast<char*>(idx->centroids_.data()), nlist * dim * sizeof(float));
    
    // Lists
    for (uint32_t i = 0; i < nlist; ++i) {
        uint32_t sz;
        ReadPod(in, sz);
        idx->lists_[i].resize(sz);
        if (sz > 0) {
            in.read(reinterpret_cast<char*>(idx->lists_[i].data()), sz * sizeof(pomai::VectorId));
        }
    }
    
    if (!in) return pomai::Status::Internal("Read failed/Truncated");
    
    *out = std::move(idx);
    return pomai::Status::Ok();
}

} // namespace pomai::index
