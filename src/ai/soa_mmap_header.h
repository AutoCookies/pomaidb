/*
 * src/ai/soa_mmap_header.h
 *
 * On-disk header for SoA (Structure-of-Arrays) mmap file used by the
 * Pomai "Thaut65"/PomaiLight vector store.
 *
 * Responsibilities:
 *  - declare a compact, versioned file header with fixed-size fields
 *    describing the layout offsets/sizes of the major blocks that follow:
 *      * codebooks (PQ centroids)
 *      * fingerprints (bitpacked SimHash/OPQ-sign)
 *      * pq_codes (8-bit per-subcode storage)
 *      * pq_packed4 (optional 4-bit packed on-disk codes)
 *      * ids (uint64 label/arena-offset per vector)
 *      * ppe (PPEPredictor/PPEHeader array)
 *      * user metadata (optional)
 *
 * Design goals:
 *  - Fixed-size 256-byte header (easy to mmap & version-check).
 *  - Clear, documented fields with explicit types.
 *  - Simple helpers to validate magic/version.
 *
 * Notes:
 *  - This header is stored in host (little-endian) byte-order for now.
 *    If cross-platform interchange is required, add explicit little/big
 *    endian conversion when reading/writing.
 *  - The header_size field allows future expansion (consumer should read
 *    'header_size' then mmap at least that many bytes).
 *
 * 10/10: clean, self-documented, compact.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <string>

namespace pomai::ai::soa
{

    // Fixed magic constant to identify SoA mmap files.
    // ASCII "SOAIDX1\0" encoded as little-endian 8-byte integer for readability.
    static constexpr uint64_t SOA_MMAP_MAGIC = 0x003149444958414FULL; // "SOAIDX1\0" (low-endian)

    // Current header version. Bump for incompatible header changes.
    static constexpr uint32_t SOA_MMAP_HEADER_VERSION = 1;

    // On-disk header for SoA mmap file.
    // Total size intentionally fixed to 256 bytes to simplify mapping & alignment.
    struct SoaMmapHeader
    {
        /* --- Identification / basic fields --- */

        // Magic signature, must equal SOA_MMAP_MAGIC.
        uint64_t magic;

        // Header format version (increment on incompatible changes).
        uint32_t version;

        // Actual size of this header in bytes (consumer may mmap at least this many bytes).
        // Typically equals sizeof(SoaMmapHeader) (256).
        uint32_t header_size;

        // Number of vectors stored in this SoA file (N).
        uint64_t num_vectors;

        // Vector dimensionality (D).
        uint32_t dim;

        // PQ layout: number of subquantizers (m) and codebook size per sub (k).
        uint16_t pq_m;
        uint16_t pq_k;

        // Fingerprint configuration: number of bits used for fingerprint per vector
        // (e.g. 256 or 512). If 0 the fingerprint block is absent.
        uint16_t fingerprint_bits;

        // Reserved for alignment / future small flags.
        uint16_t reserved16;

        /* --- Layout offsets and sizes (all bytes offsets from file start) --- */

        // Codebooks: contiguous block of floats representing PQ centroids.
        // Layout convention: codebooks block contains (pq_m * pq_k * subdim) floats.
        uint64_t codebooks_offset;
        uint64_t codebooks_size;

        // Fingerprints block: bitpacked fingerprints (num_vectors * fingerprint_bytes).
        uint64_t fingerprints_offset;
        uint64_t fingerprints_size;

        // Small per-vector publish flags for fingerprints: one uint32_t per vector.
        // Readers should check this flag (atomic load) before reading the fingerprint bytes
        // to avoid torn reads. If this offset is zero the flags block is absent.
        uint64_t fingerprint_flags_offset;
        uint64_t fingerprint_flags_size;

        // PQ codes (8-bit): one byte per (vector, subquantizer).
        uint64_t pq_codes_offset;
        uint64_t pq_codes_size;

        // Optional packed 4-bit codes on disk (for demoted PQ codes).
        // If absent, offset==0 and size==0.
        uint64_t pq_packed4_offset;
        uint64_t pq_packed4_size;

        // IDs / offsets block: uint64 per vector. Interpretation:
        // - If using arena-backed inline storage: this contains arena offset (local) or remote id.
        // - If using label-only approach: contains external label id.
        uint64_t ids_offset;
        uint64_t ids_size;

        // PPE block: auxiliary per-vector predictor/hints (opaque bytes of ppe_size).
        uint64_t ppe_offset;
        uint64_t ppe_size;

        // Optional user metadata block (JSON, protobuf, etc).
        uint64_t user_meta_offset;
        uint64_t user_meta_size;

        // Optional checksum over header + small region (0 == unused).
        uint64_t header_checksum;

        // Padding reserved for future fields and to make the struct 256 bytes.
        // Do not use this area for persistent metadata unless versioned.
        // NOTE: compute the size based on actual bytes before this pad (176),
        // so pad length = 256 - 176 = 80.
        std::array<char, 80> _pad;

        /* --- Helper methods --- */

        // Validate the magic and header_size minimally.
        inline bool is_magic_valid() const noexcept { return magic == SOA_MMAP_MAGIC; }

        // Validate version compatibility (here we accept equal versions only).
        // For forward/backward compatibility policies, adapt this check.
        inline bool is_version_compatible() const noexcept { return version == SOA_MMAP_HEADER_VERSION; }

        // Quick overall basic validity check.
        inline bool is_valid() const noexcept
        {
            if (!is_magic_valid())
                return false;
            if (!is_version_compatible())
                return false;
            if (header_size < sizeof(SoaMmapHeader))
                return false;
            if (dim == 0 || num_vectors == 0)
                return false;
            return true;
        }
    };

    // Compile-time check: header must be exactly 256 bytes.
    static_assert(sizeof(SoaMmapHeader) == 256, "SoaMmapHeader must be 256 bytes");

} // namespace pomai::ai::soa