#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <pomai/core/types.h>

namespace pomai
{
    enum class ScanConsistency : std::uint8_t
    {
        ConsistentSnapshot = 0,
        BestEffort = 1
    };

    enum class ScanOrder : std::uint8_t
    {
        Natural = 0,
        IdAsc = 1
    };

    struct ScanRequest
    {
        std::size_t batch_size{1024};
        bool include_vectors{true};
        bool include_metadata{true};
        Filter filter{};
        std::string cursor{};
        ScanConsistency consistency{ScanConsistency::ConsistentSnapshot};
        ScanOrder order{ScanOrder::Natural};
    };

    struct ScanItem
    {
        Id id{};
        std::uint32_t namespace_id{0};
        std::size_t vector_offset{0};
        std::size_t tag_offset{0};
        std::size_t tag_count{0};
    };

    struct ScanStats
    {
        std::size_t scanned{0};
        std::size_t returned{0};
        bool partial{false};
    };

    enum class ScanStatus : std::uint8_t
    {
        Ok = 0,
        InvalidCursor = 1,
        UnsupportedOrder = 2,
        InvalidRequest = 3
    };

    struct ScanResponse
    {
        std::vector<ScanItem> items;
        std::vector<float> vectors;
        std::vector<TagId> tags;
        std::string next_cursor;
        ScanStats stats;
        ScanStatus status{ScanStatus::Ok};
        std::string error;
    };
} // namespace pomai
