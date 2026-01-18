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
namespace pomai::core { enum class DataType : uint8_t; }

namespace pomai::server::data_supplier
{
    // New: fetch raw stored bytes (original storage format) for id_or_offset into `out`.
    // Returns true if a live vector was found (either from arena or orbit), false if zero-padding was appended.
    // out: appended raw bytes in stored element format (elem_size * dim)
    // out_dtype: DataType of stored representation (set even on zero fallback)
    // out_elem_size: bytes per scalar element in stored representation
    bool fetch_vector_raw(pomai::core::Membrance *m, uint64_t id_or_offset, std::string &out, size_t dim,
                          pomai::core::DataType &out_dtype, uint32_t &out_elem_size);

    // Backwards-compatible helper kept (decodes into float32 bytes) — still available.
    bool fetch_vector_bytes_or_fallback(pomai::core::Membrance *m, uint64_t id_or_offset, std::string &out, size_t dim);

    // Robust resolver: fetch vector into out_vec (float conversion) - unchanged API.
    bool fetch_vector(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<float> &out_vec);

    // Helper to construct simple binary header string used by ITERATE paths.
    // NOTE: For compatibility with existing clients we emit the legacy header format:
    //   "OK BINARY <count> <dim> <bytes>\n"
    // The dtype parameter is kept in the signature for callers, but is NOT printed here.
    std::string make_header(const char *tag, const std::string &dtype, size_t cnt, size_t dim, size_t bytes);
}