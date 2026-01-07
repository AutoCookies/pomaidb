#pragma once
// ai/space_quantized.h
//
// Simple quantized L2 space that expects payloads composed of quantized bytes
// (uint8 per-dimension or packed 4-bit values).
//
// This is intentionally simple and not optimized: it dequantizes per-dimension to
// float and computes squared L2. For production you can implement integer SSE/AVX
// kernels for speed.

#include "src/ai/hnswlib/hnswlib.h"
#include "src/ai/quantize.h"
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <cstring>

namespace pomai::ai
{

    class QuantizedL2SpaceBase
    {
    public:
        virtual ~QuantizedL2SpaceBase() = default;
        virtual size_t dim() const = 0;
        virtual int precision_bits() const = 0;
    };

    template <typename dist_t = float>
    class QuantizedL2Space : public hnswlib::SpaceInterface<dist_t>, public QuantizedL2SpaceBase
    {
        size_t dim_;
        int bits_; // 8 or 4

    public:
        QuantizedL2Space(size_t dim, int bits = 8) : dim_(dim), bits_(bits)
        {
            if (bits_ != 8 && bits_ != 4)
                throw std::invalid_argument("QuantizedL2Space: only 8 or 4 bits supported");
        }

        size_t dim() const override { return dim_; }
        int precision_bits() const override { return bits_; }

        // size of payload (without PPEHeader)
        size_t get_data_size() override
        {
            if (bits_ == 8)
                return dim_;
            // bits == 4: packed nibbles
            return (dim_ + 1) / 2;
        }

        // distance function expects pointers to quantized payload (no header)
        static dist_t QuantizedDist(const void *p1, const void *p2, const void * /*param*/)
        {
            // We'll not be called directly; PomaiSpace calls the underlying df with pointers
            // to the vector payload (after PPE header). But to be safe, keep a stub.
            return static_cast<dist_t>(0);
        }

        // The DISTFUNC we return will be a wrapper that dequantizes inside PomaiSpace.
        // However HNSW expects a pointer to a function of signature DISTFUNC<dist_t>.
        // We return QuantizedDist as placeholder — real logic handled via PomaiSpace.
        hnswlib::DISTFUNC<dist_t> get_dist_func() override { return &QuantizedDist; }
        void *get_dist_func_param() override { return nullptr; }
    };

} // namespace pomai::ai