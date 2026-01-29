#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <new>
#include <algorithm>
#include "types.h"

namespace pomai
{
    // Aligned Allocator để đảm bảo tương thích AVX2/AVX-512
    template <typename T, std::size_t Alignment>
    struct AlignedAllocator
    {
        using value_type = T;

        template <typename U>
        struct rebind
        {
            using other = AlignedAllocator<U, Alignment>;
        };

        AlignedAllocator() noexcept = default;
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

        T *allocate(std::size_t n)
        {
            if (n == 0)
                return nullptr;
            void *ptr = nullptr;
            if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
                throw std::bad_alloc();
            return static_cast<T *>(ptr);
        }

        void deallocate(T *p, std::size_t) noexcept { std::free(p); }
    };

    template <typename T, std::size_t A, typename U, std::size_t B>
    bool operator==(const AlignedAllocator<T, A> &, const AlignedAllocator<U, B> &) noexcept { return A == B; }
    template <typename T, std::size_t A, typename U, std::size_t B>
    bool operator!=(const AlignedAllocator<T, A> &, const AlignedAllocator<U, B> &) noexcept { return A != B; }

    class Seed
    {
    public:
        // Dùng AlignedAllocator cho dữ liệu SQ8 để tránh Segfault khi Load SIMD
        using QData = std::vector<std::uint8_t, AlignedAllocator<std::uint8_t, 64>>;

        struct Store
        {
            std::size_t dim{0};
            std::vector<Id> ids;
            QData qdata;
            std::vector<float> qmins;
            std::vector<float> qscales;
            std::size_t accounted_bytes{0};
            std::atomic<bool> is_quantized{false};
        };

        using Snapshot = std::shared_ptr<const Store>;
        using MutableSnapshot = std::shared_ptr<Store>;

        explicit Seed(std::size_t dim);
        Seed(const Seed &other);
        Seed &operator=(const Seed &other);
        Seed(Seed &&other) noexcept;
        Seed &operator=(Seed &&other) noexcept;
        ~Seed();

        void ApplyUpserts(const std::vector<UpsertRequest> &batch);
        Snapshot MakeSnapshot() const;

        static void Quantize(MutableSnapshot snap);
        static void DequantizeRow(const Snapshot &snap, std::size_t row, float *out);
        static std::vector<float> DequantizeSnapshot(const Snapshot &snap);
        static std::vector<float> DequantizeSnapshotBounded(const Snapshot &snap, std::size_t max_bytes);

        static SearchResponse SearchSnapshot(const Snapshot &snap, const SearchRequest &req);

        // Global Calibration Methods
        void SetFixedBounds(const std::vector<float> &mins, const std::vector<float> &maxs);
        void InheritBounds(const Seed &other);
        void SetFixedBoundsAfterCount(std::size_t count);
        std::uint64_t ConsumeOutOfRangeCount();

        std::size_t Count() const { return ids_.size(); }
        std::size_t Dim() const { return dim_; }

    private:
        std::size_t dim_{0};
        std::vector<Id> ids_;
        QData qdata_;
        std::vector<float> qmins_;
        std::vector<float> qmaxs_;
        std::vector<float> qscales_;
        std::vector<float> qinv_scales_;
        std::unordered_map<Id, std::uint32_t> pos_;
        std::size_t accounted_bytes_{0};

        std::size_t qrows_{0};
        std::size_t qcap_{0};

        std::vector<float> sample_buf_;
        std::vector<Id> sample_ids_;
        std::size_t sample_rows_{0};
        std::size_t sample_threshold_{4096};
        bool calibrated_{false};
        bool is_fixed_{false};
        std::size_t total_ingested_{0};
        std::size_t fixed_bounds_after_{50000};
        std::atomic<std::uint64_t> out_of_range_rows_{0};

        void ReserveForAppend(std::size_t add_rows);
        void UpdateMemtableAccounting();
        void ReleaseMemtableAccounting();
        void EnsureCalibration();
        void FinalizeCalibrationAndQuantizeSamples();
        void RescaleAll(const std::vector<float> &new_mins, const std::vector<float> &new_scales);

        using QuantizeRowFn = void (*)(const float *src, const float *mins, const float *inv_scales, std::uint8_t *dst, std::size_t dim);
        static QuantizeRowFn quantize_row_impl_;
        static void InitQuantizeDispatch();

        static std::uint8_t *AlignedAllocBytes(std::size_t bytes);
        static void AlignedFreeBytes(std::uint8_t *p);
    };
}
