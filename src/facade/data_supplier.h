#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace pomai::core
{
    struct Membrance;
}
namespace pomai::core
{
    enum class DataType : uint8_t;
}

namespace pomai::server::data_supplier
{
    bool fetch_vector_raw(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<char> &out, size_t dim,
                          pomai::core::DataType &out_dtype, uint32_t &out_elem_size);

    bool fetch_vector_bytes_or_fallback(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<char> &out, size_t dim);

    bool fetch_vector(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<float> &out_vec);

    std::string make_header(const char *tag, const std::string &dtype, size_t cnt, size_t dim, size_t bytes);

} // namespace pomai::server::data_supplier