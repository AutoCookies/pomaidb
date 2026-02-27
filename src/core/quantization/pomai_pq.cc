// pomai_pq.cc — Product Quantizer implementation.
//
// Phase 2: Self-contained PQ8 with Lloyd's k-means training.
// Copyright 2026 PomaiDB authors. MIT License.

#include "core/quantization/pomai_pq.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

namespace pomai::core {

// ── Constructor ────────────────────────────────────────────────────────────────
ProductQuantizer::ProductQuantizer(uint32_t dim, uint32_t M, uint32_t nbits)
    : dim_(dim), M_(M), nbits_(nbits),
      ksub_(1u << nbits),
      dsub_(dim / M),
      code_size_((M * nbits + 7) / 8)
{
    assert(dim_ % M_ == 0 && "dim must be divisible by M");
    assert(nbits_ == 8 && "only PQ8 (8-bit codes) supported");
    centroids_.resize(static_cast<std::size_t>(M_) * ksub_ * dsub_, 0.0f);
}

// ── Internal: k-means on one sub-space ────────────────────────────────────────
namespace {

void KMeans(const float* data, std::size_t n, uint32_t d,
            uint32_t k, float* centroids, int max_iter, std::mt19937& rng)
{
    // Forgy init: pick k random distinct samples
    std::vector<uint32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0u);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (uint32_t c = 0; c < k; ++c)
        std::memcpy(centroids + c * d, data + idx[c % n] * d, d * sizeof(float));

    std::vector<uint32_t> assign(n, 0);
    std::vector<uint32_t> counts(k, 0);
    std::vector<float>    new_cent(k * d, 0.0f);

    for (int iter = 0; iter < max_iter; ++iter) {
        bool changed = false;

        // Assignment step
        for (std::size_t i = 0; i < n; ++i) {
            const float* xi = data + i * d;
            float best_d = std::numeric_limits<float>::max();
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k; ++c) {
                float dist = 0.0f;
                const float* cc = centroids + c * d;
                for (uint32_t j = 0; j < d; ++j) {
                    float dj = xi[j] - cc[j]; dist += dj * dj;
                }
                if (dist < best_d) { best_d = dist; best_c = c; }
            }
            if (assign[i] != best_c) { assign[i] = best_c; changed = true; }
        }
        if (!changed && iter > 0) break;

        // Update step
        std::fill(counts.begin(), counts.end(), 0u);
        std::fill(new_cent.begin(), new_cent.end(), 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            uint32_t c = assign[i];
            counts[c]++;
            float* nc = new_cent.data() + c * d;
            const float* xi = data + i * d;
            for (uint32_t j = 0; j < d; ++j) nc[j] += xi[j];
        }
        for (uint32_t c = 0; c < k; ++c) {
            if (counts[c] == 0) continue; // keep old centroid
            float inv = 1.0f / static_cast<float>(counts[c]);
            float* cc = centroids + c * d;
            float* nc = new_cent.data() + c * d;
            for (uint32_t j = 0; j < d; ++j) cc[j] = nc[j] * inv;
        }
    }
}

} // namespace

// ── Training ──────────────────────────────────────────────────────────────────
pomai::Status ProductQuantizer::Train(const float* data, std::size_t n, int max_iter)
{
    if (n < ksub_)
        return pomai::Status::InvalidArgument(
            "PQ training requires at least ksub=" + std::to_string(ksub_) + " vectors");

    std::mt19937 rng(42);

    // Sub-vector slab for sub-space m
    std::vector<float> sub(n * dsub_);

    for (uint32_t m = 0; m < M_; ++m) {
        // Gather sub-vectors for sub-space m
        const uint32_t offset = m * dsub_;
        for (std::size_t i = 0; i < n; ++i)
            std::memcpy(sub.data() + i * dsub_,
                        data + i * dim_ + offset,
                        dsub_ * sizeof(float));

        KMeans(sub.data(), n, dsub_, ksub_,
               GetCentroid(m, 0), max_iter, rng);
    }

    trained_ = true;
    return pomai::Status::Ok();
}

// ── Encoding ──────────────────────────────────────────────────────────────────
void ProductQuantizer::Encode(const float* x, uint8_t* code) const
{
    for (uint32_t m = 0; m < M_; ++m) {
        const float* xm = x + m * dsub_;
        float best_d = std::numeric_limits<float>::max();
        uint8_t best_k = 0;
        for (uint32_t k = 0; k < ksub_; ++k) {
            const float* c = GetCentroid(m, k);
            float d = 0.0f;
            for (uint32_t j = 0; j < dsub_; ++j) {
                float dj = xm[j] - c[j]; d += dj * dj;
            }
            if (d < best_d) { best_d = d; best_k = static_cast<uint8_t>(k); }
        }
        code[m] = best_k;
    }
}

void ProductQuantizer::EncodeBatch(const float* x, std::size_t n,
                                   uint8_t* codes) const
{
    for (std::size_t i = 0; i < n; ++i)
        Encode(x + i * dim_, codes + i * code_size_);
}

// ── Decoding ──────────────────────────────────────────────────────────────────
void ProductQuantizer::Decode(const uint8_t* code, float* x) const
{
    for (uint32_t m = 0; m < M_; ++m)
        std::memcpy(x + m * dsub_, GetCentroid(m, code[m]), dsub_ * sizeof(float));
}

// ── ADC: precompute distance tables ──────────────────────────────────────────
void ProductQuantizer::ComputeL2Table(const float* x, float* table) const
{
    for (uint32_t m = 0; m < M_; ++m) {
        const float* xm = x + m * dsub_;
        float* tab_m = table + m * ksub_;
        for (uint32_t k = 0; k < ksub_; ++k) {
            const float* c = GetCentroid(m, k);
            float d = 0.0f;
            for (uint32_t j = 0; j < dsub_; ++j) {
                float dj = xm[j] - c[j]; d += dj * dj;
            }
            tab_m[k] = d;
        }
    }
}

void ProductQuantizer::ComputeIPTable(const float* x, float* table) const
{
    for (uint32_t m = 0; m < M_; ++m) {
        const float* xm = x + m * dsub_;
        float* tab_m = table + m * ksub_;
        for (uint32_t k = 0; k < ksub_; ++k) {
            const float* c = GetCentroid(m, k);
            float ip = 0.0f;
            for (uint32_t j = 0; j < dsub_; ++j) ip += xm[j] * c[j];
            tab_m[k] = ip;
        }
    }
}

float ProductQuantizer::ScoreFromTable(const float* table,
                                       const uint8_t* code) const
{
    float s = 0.0f;
    for (uint32_t m = 0; m < M_; ++m)
        s += table[m * ksub_ + code[m]];
    return s;
}

void ProductQuantizer::ScoreAllFromTable(const float* table,
                                         const uint8_t* codes, std::size_t n,
                                         float* out) const
{
    for (std::size_t i = 0; i < n; ++i)
        out[i] = ScoreFromTable(table, codes + i * code_size_);
}

// ── Persistence ───────────────────────────────────────────────────────────────
pomai::Status ProductQuantizer::Save(const std::string& path) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return pomai::Status::IOError("Cannot open PQ file for write: " + path);
    const uint32_t magic = 0x504D4151; // 'PMAQ'
    f.write(reinterpret_cast<const char*>(&magic),   sizeof(magic));
    f.write(reinterpret_cast<const char*>(&dim_),    sizeof(dim_));
    f.write(reinterpret_cast<const char*>(&M_),      sizeof(M_));
    f.write(reinterpret_cast<const char*>(&nbits_),  sizeof(nbits_));
    const std::size_t cent_sz = centroids_.size() * sizeof(float);
    f.write(reinterpret_cast<const char*>(centroids_.data()), cent_sz);
    if (!f) return pomai::Status::IOError("Write failed: " + path);
    return pomai::Status::Ok();
}

pomai::Status ProductQuantizer::Load(const std::string& path,
                                      std::unique_ptr<ProductQuantizer>* out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return pomai::Status::IOError("Cannot open PQ file: " + path);
    uint32_t magic, dim, M, nbits;
    f.read(reinterpret_cast<char*>(&magic),  sizeof(magic));
    f.read(reinterpret_cast<char*>(&dim),    sizeof(dim));
    f.read(reinterpret_cast<char*>(&M),      sizeof(M));
    f.read(reinterpret_cast<char*>(&nbits),  sizeof(nbits));
    if (magic != 0x504D4151u)
        return pomai::Status::Corruption("Bad PQ magic in " + path);
    auto pq = std::make_unique<ProductQuantizer>(dim, M, nbits);
    const std::size_t cent_sz = pq->centroids_.size() * sizeof(float);
    f.read(reinterpret_cast<char*>(pq->centroids_.data()), cent_sz);
    if (!f) return pomai::Status::IOError("Read failed: " + path);
    pq->trained_ = true;
    *out = std::move(pq);
    return pomai::Status::Ok();
}

} // namespace pomai::core
