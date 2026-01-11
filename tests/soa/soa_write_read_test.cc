// tests/soa/soa_write_read_test.cc
//
// Simple end-to-end test for SoA mmap helper (phase 1).
//
// This test exercises the expected minimal VectorStore SoA helper API:
//   - open_or_create(path, header)
//   - append_vector(fp_ptr, fp_len, pq_ptr, pq_len, id_entry) -> returns index
//   - fingerprint_ptr(idx) / pq_packed_ptr(idx) / id_entry_at(idx)
//   - close()
//
// The test writes N vectors with deterministic contents, closes the mapping,
// reopens the file and validates the stored bytes and id entries match.
//
// NOTE: This test assumes the existence of src/ai/vector_store_soa.h providing
// the minimal API described above. Implement that helper as part of Phase 1.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <unistd.h>

#include "src/ai/soa_mmap_header.h"
#include "src/ai/ids_block.h"

// VectorStore SoA helper header (to be implemented in Phase 1)
#include "src/ai/vector_store_soa.h"

using namespace pomai::ai::soa;

static std::string tmp_path(const std::string &base)
{
    // create path under /tmp
    return std::string("/tmp/") + base;
}

int main()
{
    const std::string base = "test_soa_store.bin";
    const std::string path = tmp_path(base);

    // Ensure clean slate
    unlink(path.c_str());

    const size_t N = 1024; // number of vectors to store
    const uint32_t dim = 16; // arbitrary dim for test
    const uint16_t pq_m = 8; // subquantizers
    const size_t fp_bits = 256;
    const size_t fp_bytes = (fp_bits + 7) / 8;
    const size_t pq_packed_bytes = (pq_m + 1) / 2; // packed-4 layout used in spec

    // Build header
    SoaMmapHeader hdr{};
    hdr.magic = SOA_MMAP_MAGIC;
    hdr.version = SOA_MMAP_HEADER_VERSION;
    hdr.header_size = static_cast<uint32_t>(sizeof(SoaMmapHeader));
    hdr.num_vectors = static_cast<uint64_t>(N);
    hdr.dim = dim;
    hdr.pq_m = pq_m;
    hdr.pq_k = 256;
    hdr.fingerprint_bits = static_cast<uint16_t>(fp_bits);

    // Create SoA store
    VectorStoreSoA store;
    if (!store.open_or_create(path, hdr))
    {
        std::cerr << "Failed to create SoA store at " << path << "\n";
        return 2;
    }

    // Prepare deterministic content & append
    for (size_t i = 0; i < N; ++i)
    {
        std::vector<uint8_t> fp(fp_bytes);
        std::vector<uint8_t> pq(pq_packed_bytes);

        // fill fingerprint with repeating pattern derived from index
        for (size_t b = 0; b < fp_bytes; ++b)
            fp[b] = static_cast<uint8_t>((i + 31 * b) & 0xFFu);

        // fill packed PQ bytes with a different pattern
        for (size_t b = 0; b < pq_packed_bytes; ++b)
            pq[b] = static_cast<uint8_t>((i * 7 + b * 13) & 0xFFu);

        // For this test store label id = i+1 (pack as LABEL)
        uint64_t label = static_cast<uint64_t>(i + 1);
        uint64_t id_entry = IdEntry::pack_label(label);

        size_t idx = store.append_vector(fp.data(), static_cast<uint32_t>(fp.size()),
                                         pq.data(), static_cast<uint32_t>(pq.size()),
                                         id_entry);
        if (idx != i)
        {
            std::cerr << "append_vector returned unexpected index: got=" << idx << " want=" << i << "\n";
            return 3;
        }
    }

    // Close to force unmap / flush
    store.close();

    // Re-open and validate contents
    VectorStoreSoA reopened;
    if (!reopened.open_or_create(path, hdr))
    {
        std::cerr << "Failed to reopen SoA store at " << path << "\n";
        return 4;
    }

    // Validate stored entries
    for (size_t i = 0; i < N; ++i)
    {
        const uint8_t *fp_ptr = reopened.fingerprint_ptr(i);
        const uint8_t *pq_ptr = reopened.pq_packed_ptr(i);
        uint64_t stored_entry = reopened.id_entry_at(i);

        if (fp_ptr == nullptr || pq_ptr == nullptr)
        {
            std::cerr << "Null pointer returned for index " << i << "\n";
            return 5;
        }

        // Recompute expected patterns and compare
        for (size_t b = 0; b < fp_bytes; ++b)
        {
            uint8_t want = static_cast<uint8_t>((i + 31 * b) & 0xFFu);
            if (fp_ptr[b] != want)
            {
                std::cerr << "Fingerprint mismatch at idx=" << i << " byte=" << b
                          << " got=" << static_cast<int>(fp_ptr[b]) << " want=" << static_cast<int>(want) << "\n";
                return 6;
            }
        }
        for (size_t b = 0; b < pq_packed_bytes; ++b)
        {
            uint8_t want = static_cast<uint8_t>((i * 7 + b * 13) & 0xFFu);
            if (pq_ptr[b] != want)
            {
                std::cerr << "PQ packed mismatch at idx=" << i << " byte=" << b
                          << " got=" << static_cast<int>(pq_ptr[b]) << " want=" << static_cast<int>(want) << "\n";
                return 7;
            }
        }

        // decode label
        if (!IdEntry::is_label(stored_entry))
        {
            std::cerr << "Stored id entry at idx=" << i << " is not a label (entry=" << stored_entry << ")\n";
            return 8;
        }
        uint64_t read_label = IdEntry::unpack_label(stored_entry);
        if (read_label != static_cast<uint64_t>(i + 1))
        {
            std::cerr << "Label mismatch at idx=" << i << " got=" << read_label << " want=" << (i + 1) << "\n";
            return 9;
        }
    }

    // cleanup
    reopened.close();

    std::cout << "SoA write/read test passed for " << N << " entries\n";
    return 0;
}