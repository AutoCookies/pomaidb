/*
 * src/ai/fingerprint.h
 *
 * High-level fingerprint encoder interface and concrete implementations:
 * - SimHashEncoder: uses dense random projections (SimHash) to produce
 * bitpacked fingerprints (reuses SimHash class).
 * - OPQSignEncoder: optional orthogonal rotation (OPQ-like) followed by
 * SimHash sign projection (helps improve PQ / binary-filter selectivity).
 *
 * Design goals (10/10):
 * - Small, clear public interface (FingerprintEncoder) suitable for runtime
 * selection and dependency injection.
 * - Thread-safe for concurrent reads after construction. No internal mutable
 * state is modified during compute().
 * - Clean, well-documented C++ with simple file-based persistence helpers
 * for rotation matrices (plain binary row-major floats).
 *
 * Usage:
 * // create SimHash encoder directly
 * auto enc = FingerprintEncoder::createSimHash(dim, cfg, seed);
 * std::vector<uint8_t> out(enc->bytes());
 * enc->compute(vec, out.data());
 *
 * // create OPQ+SimHash encoder (rotation loaded or identity)
 * auto enc2 = FingerprintEncoder::createOPQSign(dim, cfg, rotation_path, seed);
 *
 * Notes:
 * - OPQSignEncoder stores the rotation matrix as floats in row-major order.
 * If rotation file is not found/invalid, it falls back to identity rotation.
 * - The heavy lifting of projection/sign computation is delegated to SimHash.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "src/core/config.h" // [CHANGED] Include for config types

namespace pomai::ai
{

    // Forward-declare SimHash class (implemented in simhash.h/.cc)
    class SimHash;

    /*
     * Abstract fingerprint encoder interface.
     * Concrete implementations produce a bitpacked fingerprint for a float vector.
     *
     * Threading: compute() is thread-safe and may be called concurrently on the
     * same encoder instance after construction.
     */
    class FingerprintEncoder
    {
    public:
        virtual ~FingerprintEncoder() = default;

        // Number of output bytes required to hold the bitpacked fingerprint.
        virtual size_t bytes() const noexcept = 0;

        // Compute fingerprint for a single vector (length == dim).
        // - vec: pointer to float vector of length dim
        // - out_bytes: pointer to buffer of length bytes()
        virtual void compute(const float *vec, uint8_t *out_bytes) const = 0;

        // Convenience: compute into uint64_t word array (word_count must be >= (bits+63)/64).
        virtual void compute_words(const float *vec, uint64_t *out_words, size_t word_count) const = 0;

        // Factory helpers ----------------------------------------------------

        // Create a plain SimHash encoder.
        // - dim: dimensionality of input vectors
        // - cfg: Fingerprint configuration (contains bits)
        // - seed: RNG seed used by SimHash to initialize projection matrix
        // [CHANGED] Signature to take config object
        static std::unique_ptr<FingerprintEncoder> createSimHash(
            size_t dim,
            const pomai::config::FingerprintConfig &cfg,
            uint64_t seed = 123456789ULL);

        // Create an OPQ-sign encoder: applies rotation (dim x dim) then SimHash.
        // - rotation_path: optional path to a binary rotation matrix file (row-major floats).
        //   If empty or load fails, identity rotation is used.
        // - dim/cfg/seed as above.
        // [CHANGED] Signature to take config object
        static std::unique_ptr<FingerprintEncoder> createOPQSign(
            size_t dim,
            const pomai::config::FingerprintConfig &cfg,
            const std::string &rotation_path = std::string(),
            uint64_t seed = 123456789ULL);
    };

} // namespace pomai::ai