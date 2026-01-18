// src/facade/data_supplier.h
//
// Small utility extracted from server.h to provide data-fetch helpers used by ITERATE
// and related streaming paths. This file only *moves* logic out of server.h so the
// original behavior is preserved exactly.

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace pomai::core { struct Membrance; }

namespace pomai::server::data_supplier
{
    // Try to append raw vector bytes (float32 payload) for id_or_offset into `out`.
    // Returns true if a live vector was found (either from arena or orbit), false if zero-padding was appended.
    // NOTE: This now supports all storage DataType by decoding stored representation
    // into float32 before appending (preserving previous external behavior).
    bool fetch_vector_bytes_or_fallback(pomai::core::Membrance *m, uint64_t id_or_offset, std::string &out, size_t dim);

    // Robust resolver: fetch vector into out_vec.
    // Tries orbit.get(label) first, then arena blob pointer, then ordinal mapping via orbit->get_all_labels().
    // Always produces out_vec with length == dim (zero-filled on failure). Returns true if fetch found a real vector.
    bool fetch_vector(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<float> &out_vec);

    // Helper to construct simple binary header string used by ITERATE paths.
    // Example header: "OK BINARY 10 2048 16777216\n"
    std::string make_header(const char *tag, size_t cnt, size_t dim, size_t bytes);
}