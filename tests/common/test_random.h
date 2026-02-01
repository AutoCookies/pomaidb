#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace pomai::test
{

    class Rng
    {
    public:
        explicit Rng(std::uint64_t seed) : gen_(seed) {}

        std::uint64_t U64() { return gen_(); }
        std::uint32_t U32() { return static_cast<std::uint32_t>(gen_()); }
        float F32()
        {
            // [0,1)
            return std::uniform_real_distribution<float>(0.0f, 1.0f)(gen_);
        }

        std::vector<float> Vec(std::uint32_t dim)
        {
            std::vector<float> v(dim);
            for (auto &x : v)
                x = F32();
            return v;
        }

    private:
        std::mt19937_64 gen_;
    };

} // namespace pomai::test
