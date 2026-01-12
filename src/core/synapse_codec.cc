/* src/core/synapse_codec.cc */
#include "src/core/synapse_codec.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace pomai::core
{
    void SynapseCodec::pack_4bit_delta(size_t dim, const float* vec, const float* centroid, float scale, uint8_t* out_packed)
    {
        float inv_scale = 1.0f / scale; // Dùng phép nhân thay vì chia
        // ... (Logic nén giữ nguyên như cũ vì đã tốt) ...
        for (size_t i = 0; i < dim; i += 2) {
            float d1 = (vec[i] - centroid[i]) * scale;
            int v1 = std::clamp(static_cast<int>(std::round(d1)), -8, 7);
            
            int v2 = 0;
            if (i + 1 < dim) {
                float d2 = (vec[i+1] - centroid[i+1]) * scale;
                v2 = std::clamp(static_cast<int>(std::round(d2)), -8, 7);
            }
            uint8_t n1 = static_cast<uint8_t>(v1 + 8); 
            uint8_t n2 = static_cast<uint8_t>(v2 + 8);
            out_packed[i / 2] = (n2 << 4) | (n1 & 0x0F);
        }
    }

    // [TỐI ƯU HÓA] Tạo bảng tra cứu
    void SynapseCodec::precompute_search_lut(size_t dim, const float* query, const float* centroid, float scale, float* out_lut)
    {
        // Với mỗi chiều (dim), ta có 16 giá trị 4-bit có thể xảy ra (-8 đến 7).
        // Ta tính trước khoảng cách L2Squared từ Query đến 16 giá trị đó.
        float inv_scale = 1.0f / scale;

        for (size_t i = 0; i < dim; ++i) {
            float q_minus_c = query[i] - centroid[i];
            float* dim_lut = out_lut + (i * 16); // Pointer đến bảng của chiều i

            // Duyệt 16 giá trị code (0..15) tương ứng delta (-8..7)
            for (int code = 0; code < 16; ++code) {
                int delta = code - 8;
                float reconstructed_val = delta * inv_scale; // Giá trị delta thực
                float diff = q_minus_c - reconstructed_val;  // (Q - C) - Delta
                dim_lut[code] = diff * diff;                 // Lưu bình phương khoảng cách
            }
        }
    }

    // [SIÊU TỐC] Tính khoảng cách từ bảng LUT
    float SynapseCodec::dist_4bit_lut(size_t dim, const float* lut, const uint8_t* packed_data)
    {
        float sum = 0.0f;
        size_t n_blocks = dim / 2;

        // Unroll loop nhẹ nhàng để CPU prefetch tốt hơn
        const uint8_t* ptr = packed_data;
        const float* lut_ptr = lut;

        for (size_t i = 0; i < n_blocks; ++i) {
            uint8_t byte = *ptr++;
            
            // Low nibble (chiều i*2)
            uint8_t code_low = byte & 0x0F;
            sum += lut_ptr[code_low]; 
            lut_ptr += 16; // Nhảy sang bảng của chiều tiếp theo

            // High nibble (chiều i*2 + 1)
            uint8_t code_high = byte >> 4;
            sum += lut_ptr[code_high];
            lut_ptr += 16;
        }

        // Xử lý chiều lẻ cuối cùng (nếu có)
        if (dim % 2 != 0) {
            uint8_t byte = *ptr;
            uint8_t code_low = byte & 0x0F;
            sum += lut_ptr[code_low];
        }

        return sum;
    }
}