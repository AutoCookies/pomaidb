/*
 * src/ai/fingerprint.cc
 *
 * Implementation of FingerprintEncoder, SimHash wrapper and OPQ-sign wrapper.
 *
 * The module purposely keeps compute() implementations simple and correct.
 * Optimizations (blocking, SIMD) can be added later in hot paths.
 *
 * File I/O for rotation matrix:
 *  - Binary layout: header = uint64_t dim, followed by dim*dim floats
 *    stored row-major (row0, row1, ...).
 *  - This is intentionally minimal and easy to read/write.
 *
 * Comments and explanations are written in clear English for maintainability.
 */

#include "src/ai/fingerprint.h"
#include "src/ai/simhash.h"

#include <fstream>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace pomai::ai
{

    // -------------------- SimHashEncoder (thin wrapper) --------------------

    class SimHashEncoder : public FingerprintEncoder
    {
    public:
        SimHashEncoder(size_t dim, size_t bits, uint64_t seed)
            : simhash_(dim, bits, seed)
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
    //
    // OPQSignEncoder holds a rotation matrix (dim x dim floats, row-major).
    // compute() first applies rotation: y = R * x, then forwards rotated vector to SimHash.
    // Rotation storage is contiguous vector<float> of size dim*dim.
    //
    // For correctness we perform straightforward dense matrix-vector multiply.
    // This is easy to understand and correct; if performance becomes critical we can
    // add a blocked / SIMD multiply later.
    class OPQSignEncoder : public FingerprintEncoder
    {
    public:
        // rotation: empty => identity
        OPQSignEncoder(size_t dim, size_t bits, uint64_t seed, std::vector<float> &&rotation)
            : dim_(dim), simhash_(dim, bits, seed), rotation_(std::move(rotation))
        {
            // If rotation supplied, its size must be dim*dim. Otherwise rotation_ is empty.
            if (!rotation_.empty() && rotation_.size() != dim_ * dim_)
                throw std::invalid_argument("OPQSignEncoder: rotation size mismatch");
            // Preallocate rotation scratch (per-call stack buffer alternative).
            scratch_.resize(dim_);
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
            // Use scratch_ as temporary buffer (per-instance scratch; reads-only after construction)
            // Note: scratch_ is mutable to allow thread-safety: each call uses local vector on stack instead.
            std::vector<float> local;
            local.resize(dim_);
            const float *R = rotation_.data();
            for (size_t r = 0; r < dim_; ++r)
            {
                const float *rrow = R + r * dim_;
                double acc = 0.0;
                // simple dot product
                for (size_t c = 0; c < dim_; ++c)
                    acc += static_cast<double>(rrow[c]) * static_cast<double>(vec[c]);
                local[r] = static_cast<float>(acc);
            }

            simhash_.compute(local.data(), out_bytes);
        }

        void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const override
        {
            if (rotation_.empty())
            {
                simhash_.compute_words(vec, out_words, word_count);
                return;
            }

            std::vector<float> local;
            local.resize(dim_);
            const float *R = rotation_.data();
            for (size_t r = 0; r < dim_; ++r)
            {
                const float *rrow = R + r * dim_;
                double acc = 0.0;
                for (size_t c = 0; c < dim_; ++c)
                    acc += static_cast<double>(rrow[c]) * static_cast<double>(vec[c]);
                local[r] = static_cast<float>(acc);
            }

            simhash_.compute_words(local.data(), out_words, word_count);
        }

    private:
        size_t dim_;
        SimHash simhash_;
        std::vector<float> rotation_;
        mutable std::vector<float> scratch_; // used only as temporary if needed
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
        mat.resize(dim * dim);
        f.read(reinterpret_cast<char *>(mat.data()), static_cast<std::streamsize>(sizeof(float) * mat.size()));
        if (!f)
        {
            std::cerr << "[Fingerprint] rotation file read failed data: " << path << ", using identity\n";
            return {};
        }

        return mat;
    }

    // -------------------- Factory implementations --------------------

    std::unique_ptr<FingerprintEncoder> FingerprintEncoder::createSimHash(size_t dim, size_t bits, uint64_t seed)
    {
        return std::unique_ptr<FingerprintEncoder>(new SimHashEncoder(dim, bits, seed));
    }

    std::unique_ptr<FingerprintEncoder> FingerprintEncoder::createOPQSign(size_t dim,
                                                                          size_t bits,
                                                                          const std::string &rotation_path,
                                                                          uint64_t seed)
    {
        std::vector<float> rotation = load_rotation_matrix(rotation_path, dim);
        // If rotation empty -> identity (internal handling)
        return std::unique_ptr<FingerprintEncoder>(new OPQSignEncoder(dim, bits, seed, std::move(rotation)));
    }

} // namespace pomai::ai