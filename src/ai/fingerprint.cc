/*
 * src/ai/fingerprint.cc
 *
 * Implementation of FingerprintEncoder, SimHash wrapper and OPQ-sign wrapper.
 *
 * Security Fixes:
 * - Strict validation of rotation matrix dimensions to prevent buffer overflows.
 * - Safe fallback to Identity matrix on corruption/missing file.
 *
 * Performance:
 * - Thread-local scratch buffers for zero-alloc hot path.
 * - Direct usage of pomai_dot kernels.
 */

#include "src/ai/fingerprint.h"
#include "src/ai/simhash.h"
#include "src/core/cpu_kernels.h"
#include "src/core/config.h"

#include <fstream>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>

namespace pomai::ai
{
    // [ADDED] Helper to bridge legacy 'bits' param to Config struct
    static inline pomai::config::FingerprintConfig make_fp_config(size_t bits)
    {
        pomai::config::FingerprintConfig c;
        c.fingerprint_bits = static_cast<uint32_t>(bits);
        return c;
    }

    // -------------------- SimHashEncoder (thin wrapper) --------------------

    class SimHashEncoder : public FingerprintEncoder
    {
    public:
        SimHashEncoder(size_t dim, size_t bits, uint64_t seed)
            // [FIXED] Construct config on the fly
            : simhash_(dim, make_fp_config(bits), seed)
        {
        }

        size_t bytes() const noexcept override { return simhash_.bytes(); }

        void compute(const float *vec, uint8_t *out_bytes) const override
        {
            simhash_.compute(vec, out_bytes);
        }

        void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const override
        {
            simhash_.compute_words(vec, out_words, word_count);
        }

    private:
        SimHash simhash_;
    };

    // -------------------- OPQSignEncoder --------------------
    class OPQSignEncoder : public FingerprintEncoder
    {
    public:
        // rotation: empty => identity
        OPQSignEncoder(size_t dim, size_t bits, uint64_t seed, std::vector<float> &&rotation)
            // [FIXED] Construct config on the fly
            : dim_(dim), simhash_(dim, make_fp_config(bits), seed), rotation_(std::move(rotation))
        {
            // If rotation supplied, its size must be dim*dim. Otherwise rotation_ is empty.
            if (!rotation_.empty() && rotation_.size() != dim_ * dim_)
                throw std::invalid_argument("OPQSignEncoder: rotation size mismatch");
        }

        size_t bytes() const noexcept override { return simhash_.bytes(); }

        void compute(const float *vec, uint8_t *out_bytes) const override
        {
            // Fast path: identity rotation (no rotation matrix provided)
            if (rotation_.empty())
            {
                simhash_.compute(vec, out_bytes);
                return;
            }

            // Apply rotation: out = R * vec  (R is row-major)
            const float *R = rotation_.data();

            // Thread-local scratch buffer to avoid repeated allocations in hot path.
            // Bound the thread-local reservation to avoid unbounded memory per thread.
            static thread_local std::vector<float> scratch;
            constexpr size_t MAX_THREAD_SCRATCH = 262144; // max floats (~1 MiB) per thread buffer
            float *local_ptr = nullptr;
            std::vector<float> fallback; // used only if dim_ > MAX_THREAD_SCRATCH

            if (dim_ <= MAX_THREAD_SCRATCH)
            {
                if (scratch.size() < dim_)
                    scratch.resize(dim_);
                local_ptr = scratch.data();
            }
            else
            {
                // Rare path for extremely large dims: allocate once for this call to remain correct.
                fallback.resize(dim_);
                local_ptr = fallback.data();
            }

            // Use the central kernel directly (pomai_dot) to compute each row dot product.
            for (size_t r = 0; r < dim_; ++r)
            {
                const float *rrow = R + r * dim_;
                local_ptr[r] = ::pomai_dot(rrow, vec, dim_);
            }

            simhash_.compute(local_ptr, out_bytes);
        }

        void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const override
        {
            if (rotation_.empty())
            {
                simhash_.compute_words(vec, out_words, word_count);
                return;
            }

            const float *R = rotation_.data();

            // Thread-local scratch buffer same as above
            static thread_local std::vector<float> scratch;
            constexpr size_t MAX_THREAD_SCRATCH = 262144; // max floats (~1 MiB) per thread buffer
            float *local_ptr = nullptr;
            std::vector<float> fallback;

            if (dim_ <= MAX_THREAD_SCRATCH)
            {
                if (scratch.size() < dim_)
                    scratch.resize(dim_);
                local_ptr = scratch.data();
            }
            else
            {
                fallback.resize(dim_);
                local_ptr = fallback.data();
            }

            for (size_t r = 0; r < dim_; ++r)
            {
                const float *rrow = R + r * dim_;
                local_ptr[r] = ::pomai_dot(rrow, vec, dim_);
            }

            simhash_.compute_words(local_ptr, out_words, word_count);
        }

    private:
        size_t dim_;
        SimHash simhash_;
        std::vector<float> rotation_;
    };

    // -------------------- Rotation matrix helpers --------------------
    //
    // Simple binary format for rotation matrix:
    //   uint64_t dim
    //   float data[ dim * dim ]   // row-major
    //
    // If file cannot be read or format is invalid we return empty vector to indicate identity.
    static std::vector<float> load_rotation_matrix(const std::string &path, size_t expected_dim)
    {
        if (path.empty())
            return {}; // identity

        std::ifstream f(path, std::ios::binary);
        if (!f)
        {
            // Not fatal: caller can decide to proceed with identity; log for visibility.
            std::cerr << "[Fingerprint] rotation file not found: " << path << ", using identity\n";
            return {};
        }

        uint64_t dim64 = 0;
        f.read(reinterpret_cast<char *>(&dim64), sizeof(dim64));
        if (!f)
        {
            std::cerr << "[Fingerprint] rotation file read failed header: " << path << ", using identity\n";
            return {};
        }
        size_t dim = static_cast<size_t>(dim64);
        if (expected_dim != 0 && dim != expected_dim)
        {
            std::cerr << "[Fingerprint] rotation file dim mismatch: file=" << dim << " expected=" << expected_dim << ", using identity\n";
            return {};
        }

        // read floats
        std::vector<float> mat;
        // [FIX] Validate size before allocation to prevent OOM DOS
        if (dim > 16384)
        { // Arbitrary sanity limit (16384^2 floats = 1GB)
            std::cerr << "[Fingerprint] rotation dimension too large: " << dim << "\n";
            return {};
        }

        mat.resize(dim * dim);
        f.read(reinterpret_cast<char *>(mat.data()), static_cast<std::streamsize>(sizeof(float) * mat.size()));

        // [FIX] Critical: Check if we actually read enough bytes
        if (!f)
        {
            std::cerr << "[Fingerprint] rotation file corrupted (EOF/Short read): " << path << ", using identity\n";
            return {};
        }

        return mat;
    }

    // -------------------- Factory implementations --------------------

    std::unique_ptr<FingerprintEncoder> FingerprintEncoder::createSimHash(
        size_t dim,
        const pomai::config::FingerprintConfig &cfg,
        uint64_t seed)
    {
        size_t use_bits = cfg.fingerprint_bits;
        if (use_bits == 0)
            use_bits = 512;

        return std::unique_ptr<FingerprintEncoder>(new SimHashEncoder(dim, use_bits, seed));
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
        // If rotation empty -> identity (internal handling)
        return std::unique_ptr<FingerprintEncoder>(new OPQSignEncoder(dim, use_bits, seed, std::move(rotation)));
    }

} // namespace pomai::ai