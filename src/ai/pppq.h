#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <mutex>
#include <atomic>
#include "ppe_predictor.h"

namespace pomai::ai
{

    class PPPQ
    {
    public:
        // d: dimensionality, m: number of subquantizers, k: codebook size per sub (e.g., 256)
        PPPQ(size_t d, size_t m, size_t k, size_t max_elems, const std::string &mmap_file = "pppq_codes.mmap");
        ~PPPQ();

        // Train codebooks from samples (n_samples x dim)
        void train(const float *samples, size_t n_samples, size_t max_iters = 10);

        // Encode and add vector for id (0..max_elems-1). Predictor used to choose bits.
        void addVec(const float *vec, size_t id);

        // Approximate distance between two vectors by their PQ codes (L2). Will read demoted codes on demand.
        float approxDist(size_t id_a, size_t id_b);

        // Periodic purge: demote predicted-cold vectors to file-backed store
        void purgeCold(uint64_t cold_thresh_ms);

        // For integration / debugging
        size_t dim() const { return dim_; }
        size_t m() const { return m_; }
        size_t subdim() const { return subdim_; }
        size_t k() const { return k_; }

    private:
        size_t dim_, m_, k_, subdim_, max_elems_;
        std::vector<float> codebooks_; // size m * k * subdim
        // precomputed pairwise distances for each sub: m blocks of k * k floats
        std::vector<std::vector<float>> precomp_dists_;

        // In-RAM per-vector storage:
        // - codes8_: each element id stores m bytes (one 0..k-1 per sub)
        // - when demoted to 4-bit we pack nibbles to mmap file; code_nbits_ stores current bitness per id
        std::vector<uint8_t> codes8_;     // size max_elems_ * m_
        std::vector<uint8_t> code_nbits_; // 8 or 4, size max_elems_
        std::vector<uint8_t> in_mmap_;    // 0/1 flag size max_elems_

        // mmap/file-backed store file name and mutex
        std::string mmap_filename_;
        std::mutex mmap_file_mtx_;

        // PPE predictors per element
        std::vector<PPEPredictor> ppe_vecs_;

        // Helpers
        int findNearestCode(const float *subvec, float *codebook) const;
        void computePrecompDists();
        void writePacked4ToFile(size_t id, const uint8_t *nibble_bytes, size_t nibble_bytes_len);
        void readPacked4FromFile(size_t id, uint8_t *nibble_bytes, size_t nibble_bytes_len);
        void pack4From8(const uint8_t *src8, uint8_t *dst_nibbles);
        void unpack4To8(const uint8_t *src_nibbles, uint8_t *dst8);
        size_t packed4BytesPerVec() const { return (m_ + 1) / 2; }

        // disable copy
        PPPQ(const PPPQ &) = delete;
        PPPQ &operator=(const PPPQ &) = delete;
    };

} // namespace pomai::ai