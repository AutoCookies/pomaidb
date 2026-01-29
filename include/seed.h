#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>
#include "types.h"

namespace pomai
{
    class Seed
    {
    public:
        struct Store
        {
            std::size_t dim{0};
            std::vector<Id> ids;
            std::vector<float> data;
            std::vector<std::uint8_t> qdata;
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
        static bool TryDetachSnapshot(Snapshot &snap, std::vector<float> &data, std::vector<Id> &ids);

        static SearchResponse SearchSnapshot(const Snapshot &snap, const SearchRequest &req);

        std::size_t Count() const { return ids_.size(); }
        std::size_t Dim() const { return dim_; }

    private:
        std::size_t dim_{0};
        std::vector<Id> ids_;
        std::vector<float> data_;
        std::unordered_map<Id, std::uint32_t> pos_;
        std::size_t accounted_bytes_{0};

        void ReserveForAppend(std::size_t add_rows);
        void UpdateMemtableAccounting();
        void ReleaseMemtableAccounting();
    };
}