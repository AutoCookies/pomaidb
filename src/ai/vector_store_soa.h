// src/ai/vector_store_soa.h
//
// Simple SoA-backed Vector Store helper (phase 1).
//
// Responsibilities (Phase 1):
//  - Create/open a single mmap'd SoA file with SoaMmapHeader layout.
//  - Provide append_vector(...) to write a fingerprint, packed PQ bytes and a 64-bit id/offset entry
//    into the preallocated slots.
//  - Provide read accessors: pq_packed_ptr(idx) and id_entry_at(idx) and basic metadata accessors.
//
// This is an intentionally small, robust implementation to satisfy tests and
// provide foundations for later features (atomic WAL, dynamic growth, codebooks etc).
//
// Threading:
//  - append_vector is serialized by an internal mutex (simple and safe for tests).
//

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>

#include "src/memory/mmap_file_manager.h"
#include "src/ai/soa_mmap_header.h"
#include "src/ai/atomic_utils.h"

namespace pomai::ai::soa
{

class VectorStoreSoA
{
public:
    // Default constructible (tests construct on stack and call open_or_create).
    VectorStoreSoA() noexcept = default;

    // Create a new SoA file (truncates/creates). Returns nullptr on failure.
    // - path: file path to create
    // - num_vectors: number of slots to reserve
    // - dim: vector dimensionality (floats) (kept for header; not used by append here)
    // - pq_m, pq_k: PQ layout (m,k) (used to compute packed4 bytes)
    // - fingerprint_bits: fingerprint bits per vector (0 -> absent)
    // - ppe_entry_bytes: bytes reserved per-vector for PPE (may be 0)
    // - user_meta: optional small metadata string to embed into user_meta block
    static std::unique_ptr<VectorStoreSoA> create_new(const std::string &path,
                                                      uint64_t num_vectors,
                                                      uint32_t dim,
                                                      uint16_t pq_m,
                                                      uint16_t pq_k,
                                                      uint16_t fingerprint_bits,
                                                      uint32_t ppe_entry_bytes = 0,
                                                      const std::string &user_meta = std::string());

    // Open an existing SoA file. Returns nullptr on failure or validation error.
    static std::unique_ptr<VectorStoreSoA> open_existing(const std::string &path);

    ~VectorStoreSoA();

    VectorStoreSoA(const VectorStoreSoA &) = delete;
    VectorStoreSoA &operator=(const VectorStoreSoA &) = delete;

    // Instance-style helper expected by tests:
    // Try to open existing mapping at 'path'. If missing, create/truncate and
    // initialize using the provided 'hdr_template' fields:
    //   hdr_template.num_vectors, dim, pq_m, pq_k, fingerprint_bits
    // Returns true on success.
    bool open_or_create(const std::string &path, const SoaMmapHeader &hdr_template);

    // Close mapping and release resources.
    void close();

    // Append a vector entry into the next free slot.
    // - fp: pointer to fingerprint bytes (may be nullptr if fingerprint_bits == 0)
    // - fp_len: length in bytes of fingerprint (must equal header fingerprint_bytes)
    // - pq_packed: pointer to packed-4 PQ bytes (if pq_m==0 then can be nullptr)
    // - pq_len: length of packed PQ bytes (must equal header packed bytes)
    // - id_entry: the uint64 value to store in ids array (label/offset)
    // Returns index (0..num_vectors-1) on success, SIZE_MAX on failure/full.
    size_t append_vector(const uint8_t *fp, uint32_t fp_len,
                         const uint8_t *pq_packed, uint32_t pq_len,
                         uint64_t id_entry);

    // Accessors used by tests ------------------------------------------------

    // pointer to packed PQ bytes for index (read-only). Returns nullptr if index out-of-range
    const uint8_t *pq_packed_ptr(size_t idx) const noexcept;

    // read id entry atomically
    uint64_t id_entry_at(size_t idx) const noexcept;

    // pointer to fingerprint bytes for index (read-only). nullptr if absent or not yet published.
    // This now checks a per-slot publish flag (atomic) to avoid torn reads.
    const uint8_t *fingerprint_ptr(size_t idx) const noexcept;

    // pointer to base ids array (const). Useful for passing to refine helpers.
    // Returns nullptr if ids block absent.
    const uint64_t *ids_ptr() const noexcept;

    // basic header info
    bool is_valid() const noexcept;
    uint64_t num_vectors() const noexcept;
    uint32_t dim() const noexcept;
    uint16_t pq_m() const noexcept;
    uint16_t pq_k() const noexcept;
    uint16_t fingerprint_bits() const noexcept;

    // flush a range of bytes (offset, len) from mapped file. sync=true -> msync(MS_SYNC)
    bool flush(size_t offset, size_t len, bool sync);

private:
    // compute helpers
    static size_t compute_codebooks_size_bytes(uint32_t dim, uint16_t pq_m, uint16_t pq_k) noexcept;
    static size_t compute_fingerprints_size_bytes(uint64_t num_vectors, uint16_t fingerprint_bits) noexcept;
    static size_t compute_fingerprint_flags_size_bytes(uint64_t num_vectors) noexcept;
    static size_t compute_pq_codes_size_bytes(uint64_t num_vectors, uint16_t pq_m) noexcept;
    static size_t compute_pq_packed4_size_bytes(uint64_t num_vectors, uint16_t pq_m) noexcept;
    static size_t compute_ids_size_bytes(uint64_t num_vectors) noexcept;
    static size_t compute_ppe_size_bytes(uint64_t num_vectors, uint32_t ppe_entry_bytes) noexcept;

    // init helpers
    bool init_from_mapping();
    bool build_and_write_header(uint64_t num_vectors,
                                uint32_t dim,
                                uint16_t pq_m,
                                uint16_t pq_k,
                                uint16_t fingerprint_bits,
                                uint32_t ppe_entry_bytes,
                                const std::string &user_meta);

private:
    // mapped file manager (owns mapping)
    pomai::memory::MmapFileManager mmap_;

    // raw pointer to base of mapping (valid while mmap_ open)
    const char *base_ptr_ = nullptr;

    // header overlay pointer in mapping
    const SoaMmapHeader *hdr_ = nullptr;

    // number of bytes for fingerprint per-vector and PQ-packed per-vector (cached)
    size_t fingerprint_bytes_{0};
    size_t pq_packed_bytes_{0};
    size_t ppe_entry_bytes_{0};
    // next free index (append). Protected by append_mu_
    std::atomic<uint64_t> next_index_{0};
    mutable std::mutex append_mu_;
};

} // namespace pomai::ai::soa