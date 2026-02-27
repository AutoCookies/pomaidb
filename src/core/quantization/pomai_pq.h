// pomai_pq.h — Product Quantizer (PQ8/PQ16) for PomaiDB.
//
// Phase 2: Self-contained PQ implementation, zero external dependencies.
// Design inspired by FAISS ProductQuantizer (MIT License, Meta Platforms).
//
// PQ splits each d-dim vector into M sub-vectors of dim d/M, and quantizes
// each using k=256 (PQ8) or k=65536 (PQ16) centroids trained by k-means.
// At query time, a M×k distance table is computed once, then all codes are
// scored in O(M) per code — much faster than exact distance.

#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

/// Product Quantizer — 8-bit codes (256 centroids per sub-space).
class ProductQuantizer {
public:
    /// @param dim    Input vector dimensionality. Must be divisible by M.
    /// @param M      Number of sub-quantizers (sub-spaces).
    /// @param nbits  Bits per code index. 8 → PQ8 (256 centroids). Only 8 supported.
    ProductQuantizer(uint32_t dim, uint32_t M, uint32_t nbits = 8);

    uint32_t dim()   const { return dim_; }
    uint32_t M()     const { return M_; }
    uint32_t nbits() const { return nbits_; }
    uint32_t ksub()  const { return ksub_; }
    uint32_t dsub()  const { return dsub_; }
    uint32_t code_size() const { return code_size_; }
    bool     trained()   const { return trained_; }

    // ── Training ──────────────────────────────────────────────────────────────
    /// Train on `n` vectors. Each row is `dim` contiguous floats.
    pomai::Status Train(const float* data, std::size_t n, int max_iter = 25);

    // ── Encoding ──────────────────────────────────────────────────────────────
    /// Encode one vector → code_size() bytes.
    void Encode(const float* x, uint8_t* code) const;

    /// Encode `n` vectors in bulk (row-major input).
    void EncodeBatch(const float* x, std::size_t n, uint8_t* codes) const;

    // ── Decoding ─────────────────────────────────────────────────────────────
    /// Decode one code → dim() floats.
    void Decode(const uint8_t* code, float* x) const;

    // ── Asymmetric Distance Computation (ADC) ─────────────────────────────────
    /// Precompute L2 distance table for query x: M × ksub floats.
    void ComputeL2Table(const float* x, float* table) const;

    /// Precompute inner-product table for query x: M × ksub floats.
    void ComputeIPTable(const float* x, float* table) const;

    /// Score one code using precomputed distance table (L2 or IP).
    float ScoreFromTable(const float* table, const uint8_t* code) const;

    /// Score n codes using precomputed table → out[i] = distance to code i.
    void ScoreAllFromTable(const float* table,
                           const uint8_t* codes, std::size_t n,
                           float* out) const;

    // ── Persistence ───────────────────────────────────────────────────────────
    pomai::Status Save(const std::string& path) const;
    static pomai::Status Load(const std::string& path,
                              std::unique_ptr<ProductQuantizer>* out);

    const float* centroids_data() const { return centroids_.data(); }

private:
    uint32_t dim_, M_, nbits_, ksub_, dsub_, code_size_;
    bool trained_ = false;
    std::vector<float> centroids_;  // M × ksub × dsub (row-major: [m][k][d])

    const float* GetCentroid(uint32_t m, uint32_t k) const {
        return centroids_.data() + (m * ksub_ + k) * dsub_;
    }
    float* GetCentroid(uint32_t m, uint32_t k) {
        return centroids_.data() + (m * ksub_ + k) * dsub_;
    }
};

} // namespace pomai::core
