#include "src/ai/fingerprint.h"
#include "src/ai/simhash.h"
#include "src/core/cpu_kernels.h"
#include "src/core/config.h"

#include <fstream>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cassert>
#include <memory>

namespace pomai::ai
{

    static inline pomai::config::FingerprintConfig make_fp_config(size_t bits)
    {
        pomai::config::FingerprintConfig c;
        c.fingerprint_bits = static_cast<uint32_t>(bits);
        return c;
    }

    static std::vector<float> load_rotation_matrix(const std::string &path, size_t expected_dim)
    {
        if (path.empty())
            return {};

        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            std::cerr << "[Fingerprint] Rotation file not found: " << path << ", using identity\n";
            return {};
        }

        uint64_t dim64 = 0;
        f.read(reinterpret_cast<char *>(&dim64), sizeof(dim64));
        if (!f || f.gcount() != sizeof(dim64))
        {
            std::cerr << "[Fingerprint] Failed to read rotation header: " << path << "\n";
            return {};
        }

        size_t dim = static_cast<size_t>(dim64);
        if (expected_dim != 0 && dim != expected_dim)
        {
            std::cerr << "[Fingerprint] Rotation dim mismatch: file=" << dim << " expected=" << expected_dim << "\n";
            return {};
        }

        if (dim > 16384)
        {
            std::cerr << "[Fingerprint] Rotation dimension too large: " << dim << "\n";
            return {};
        }

        size_t data_size = dim * dim * sizeof(float);
        std::vector<float> mat(dim * dim);
        f.read(reinterpret_cast<char *>(mat.data()), static_cast<std::streamsize>(data_size));

        if (!f || static_cast<size_t>(f.gcount()) != data_size)
        {
            std::cerr << "[Fingerprint] Rotation file corrupted or truncated: " << path << "\n";
            return {};
        }

        return mat;
    }

    class SimHashEncoder : public FingerprintEncoder
    {
    public:
        SimHashEncoder(size_t dim, size_t bits, uint64_t seed)
            : simhash_(dim, make_fp_config(bits), seed)
        {
        }

        size_t bytes() const noexcept override
        {
            return simhash_.bytes();
        }

        void compute(const float *vec, uint8_t *out_bytes) const override
        {
            simhash_.compute(vec, out_bytes);
        }

        void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const override
        {
            simhash_.compute_words(vec, out_words, word_count);
        }

    protected:
        SimHash simhash_;
    };

    class OPQSignEncoder : public FingerprintEncoder
    {
    public:
        OPQSignEncoder(size_t dim, size_t bits, uint64_t seed, std::vector<float> &&rotation)
            : dim_(dim), encoder_(dim, bits, seed), rotation_(std::move(rotation))
        {
            if (!rotation_.empty() && rotation_.size() != dim_ * dim_)
                throw std::invalid_argument("OPQSignEncoder: rotation size mismatch");
        }

        size_t bytes() const noexcept override
        {
            return encoder_.bytes();
        }

        void compute(const float *vec, uint8_t *out_bytes) const override
        {
            if (rotation_.empty())
            {
                encoder_.compute(vec, out_bytes);
                return;
            }

            const float *rotated_vec = apply_rotation(vec);
            encoder_.compute(rotated_vec, out_bytes);
        }

        void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const override
        {
            if (rotation_.empty())
            {
                encoder_.compute_words(vec, out_words, word_count);
                return;
            }

            const float *rotated_vec = apply_rotation(vec);
            encoder_.compute_words(rotated_vec, out_words, word_count);
        }

    private:
        const float *apply_rotation(const float *vec) const
        {
            static thread_local std::vector<float> scratch;
            constexpr size_t MAX_THREAD_SCRATCH = 262144;

            float *target = nullptr;

            if (dim_ <= MAX_THREAD_SCRATCH)
            {
                if (scratch.size() < dim_)
                    scratch.resize(dim_);
                target = scratch.data();
            }
            else
            {
                static thread_local std::vector<float> overflow_scratch;
                if (overflow_scratch.size() < dim_)
                    overflow_scratch.resize(dim_);
                target = overflow_scratch.data();
            }

            const float *R = rotation_.data();
            for (size_t r = 0; r < dim_; ++r)
            {
                const float *rrow = R + r * dim_;
                target[r] = ::pomai_dot(rrow, vec, dim_);
            }
            return target;
        }

        size_t dim_;
        SimHashEncoder encoder_;
        std::vector<float> rotation_;
    };

    std::unique_ptr<FingerprintEncoder> FingerprintEncoder::createSimHash(
        size_t dim,
        const pomai::config::FingerprintConfig &cfg,
        uint64_t seed)
    {
        size_t use_bits = cfg.fingerprint_bits;
        if (use_bits == 0)
            use_bits = 512;

        return std::make_unique<SimHashEncoder>(dim, use_bits, seed);
    }

    std::unique_ptr<FingerprintEncoder> FingerprintEncoder::createOPQSign(
        size_t dim,
        const pomai::config::FingerprintConfig &cfg,
        const std::string &rotation_path,
        uint64_t seed)
    {
        size_t use_bits = cfg.fingerprint_bits;
        if (use_bits == 0)
            use_bits = 512;

        std::vector<float> rotation = load_rotation_matrix(rotation_path, dim);
        return std::make_unique<OPQSignEncoder>(dim, use_bits, seed, std::move(rotation));
    }

} // namespace pomai::ai