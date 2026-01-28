#pragma once
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <optional>
#include <memory>
#include <random>
#include <limits>
#include <stdexcept>

#include "types.h"       
#include "cpu_kernels.h" 

namespace pomai
{
    class SpatialRouter
    {
    public:
        struct HotspotInfo
        {
            std::size_t centroid_idx{0};
            double ratio{0.0};
            std::size_t total_hits{0};
            double average_hits{0.0};
        };

        SpatialRouter() = default;

        // Xóa copy constructor/assignment để đảm bảo an toàn bộ đếm atomic
        SpatialRouter(const SpatialRouter&) = delete;
        SpatialRouter& operator=(const SpatialRouter&) = delete;

        std::size_t PickShardForInsert(const Vector &vec) const;
        std::vector<std::size_t> CandidateShardsForQuery(const Vector &q, std::size_t P) const;
        void ReplaceCentroids(std::vector<Vector> new_centroids);
        std::vector<Vector> SnapshotCentroids() const;
        std::optional<HotspotInfo> DetectHotspot(double threshold_ratio = 2.0) const;

        static std::vector<Vector> BuildKMeans(const std::vector<Vector> &data, std::size_t k, int iterations = 10);

    private:
        mutable std::shared_mutex mu_;
        std::vector<Vector> centroids_;
        
        // Chuẩn Big Tech: Sử dụng mảng unique_ptr để quản lý atomic counters
        // Tránh lỗi biên dịch "use of deleted function" của std::vector
        mutable std::unique_ptr<std::atomic<std::uint64_t>[]> centroid_hits_;
        std::size_t num_centroids_{0};
    };
}