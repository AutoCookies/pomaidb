#pragma once
#include <cstdint>
#include <vector>
#include <string> // <--- Bắt buộc phải có để dùng std::string

namespace pomai
{
    using VectorId = std::uint64_t;

    // Đổi tên từ 'Vector' thành 'VectorData' để khớp với UpsertItem bên dưới
    struct VectorData
    {
        std::vector<float> values;
    };

    struct UpsertItem
    {
        VectorId id;
        VectorData vec; // Bây giờ trình biên dịch đã hiểu VectorData là gì
        std::string payload;
    };

    struct SearchHit
    {
        VectorId id;
        float score;
        std::string payload;
    };

} // namespace pomai