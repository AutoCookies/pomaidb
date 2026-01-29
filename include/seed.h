#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <cstdlib>
#include <new>
#include "types.h"

namespace pomai
{
    template <typename T, std::size_t Alignment>
    struct AlignedAllocator
    {
        using value_type = T;

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

        void deallocate(T *p, std::size_t) noexcept
        {
            std::free(p);
        }
    };

    template <typename T, std::size_t Alignment, typename U, std::size_t AlignmentU>
    bool operator==(const AlignedAllocator<T, Alignment> &, const AlignedAllocator<U, AlignmentU> &) noexcept
    {
        return Alignment == AlignmentU;
    }

    template <typename T, std::size_t Alignment, typename U, std::size_t AlignmentU>
    bool operator!=(const AlignedAllocator<T, Alignment> &, const AlignedAllocator<U, AlignmentU> &) noexcept
    {
        return Alignment != AlignmentU;
    }

    class Seed
    {
    public:
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

        using Snapshot = std::shared_ptr<Store>;

        explicit Seed(std::size_t dim);
        Seed(const Seed &other);
        Seed &operator=(const Seed &other);
        Seed(Seed &&other) noexcept;
        Seed &operator=(Seed &&other) noexcept;
        ~Seed();

        void ApplyUpserts(const std::vector<UpsertRequest> &batch);

        Snapshot MakeSnapshot() const;
        static void Quantize(Snapshot snap);
        static void DequantizeRow(const Snapshot &snap, std::size_t row, float *out);
        static std::vector<float> DequantizeSnapshot(const Snapshot &snap);

        static SearchResponse SearchSnapshot(const Snapshot &snap, const SearchRequest &req);

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

        void ReserveForAppend(std::size_t add_rows);
        void UpdateMemtableAccounting();
        void ReleaseMemtableAccounting();
    };
}
