/* src/core/synapse_codec.h */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace pomai::core
{
    class SynapseCodec
    {
    public:
        // Kích thước byte nén
        static inline size_t packed_byte_size(size_t dim) {
            return (dim + 1) / 2;
        }

        // Nén: Float -> 4-bit Packed
        static void pack_4bit_delta(
            size_t dim, 
            const float* vec, 
            const float* centroid, 
            float scale, 
            uint8_t* out_packed);

        // [MỚI] Tạo bảng LUT (Look-up Table) cho Query hiện tại
        // out_lut cần kích thước: dim * 16 (float)
        static void precompute_search_lut(
            size_t dim,
            const float* query,
            const float* centroid,
            float scale,
            float* out_lut
        );

        // [MỚI] Tính khoảng cách cực nhanh bằng cách tra bảng LUT
        // Không giải nén, không nhân float trong vòng lặp
        static float dist_4bit_lut(
            size_t dim,
            const float* lut,
            const uint8_t* packed_data
        );
        
        // (Giữ lại hàm cũ để fallback nếu cần, nhưng search chính sẽ dùng LUT)
        static float dist_4bit_delta(
            size_t dim, const float* query, const float* centroid, 
            float scale, const uint8_t* packed_data);
    };
}