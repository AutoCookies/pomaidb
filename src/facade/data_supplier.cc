// src/facade/data_supplier.cc
//
// Implementation for data_supplier.h
// Updated to support multiple storage data types: float32, float64, int32, int8, float16.
// The public API is unchanged: fetch_vector_bytes_or_fallback still appends float32 bytes
// (same external behavior) but now will decode stored representations as needed.

#include "src/facade/data_supplier.h"

#include <cstring>
#include <random>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "src/core/pomai_db.h"       // for Membrance
#include "src/memory/append_only_arena.h"
#include "src/memory/shard_arena.h"
#include "src/core/types.h"
#include "src/core/cpu_kernels.h" // fp16 helpers

namespace pomai::server::data_supplier
{

// Helper: decode raw storage slot (payload pointer) of given DataType into float buffer `out` (length dim).
static inline void decode_slot_to_float(const char *payload, size_t dim, pomai::core::DataType dt, float *out)
{
    using pomai::core::DataType;
    switch (dt)
    {
    case DataType::FLOAT32:
        std::memcpy(out, payload, dim * sizeof(float));
        break;

    case DataType::FLOAT64:
    {
        const double *dp = reinterpret_cast<const double *>(payload);
        for (size_t i = 0; i < dim; ++i) out[i] = static_cast<float>(dp[i]);
        break;
    }

    case DataType::INT32:
    {
        const int32_t *ip = reinterpret_cast<const int32_t *>(payload);
        for (size_t i = 0; i < dim; ++i) out[i] = static_cast<float>(ip[i]);
        break;
    }

    case DataType::INT8:
    {
        const int8_t *ip = reinterpret_cast<const int8_t *>(payload);
        for (size_t i = 0; i < dim; ++i) out[i] = static_cast<float>(ip[i]);
        break;
    }

    case DataType::FLOAT16:
    {
        const uint16_t *hp = reinterpret_cast<const uint16_t *>(payload);
        for (size_t i = 0; i < dim; ++i) out[i] = fp16_to_fp32(hp[i]);
        break;
    }

    default:
        // Fallback: try to memcpy as float32 up to available bytes
        std::memcpy(out, payload, dim * sizeof(float));
        break;
    }
}

bool fetch_vector_bytes_or_fallback(pomai::core::Membrance *m, uint64_t id_or_offset, std::string &out, size_t dim)
{
    // Default to float32 output for compatibility with callers.
    size_t out_bytes = dim * sizeof(float);

    if (!m)
    {
        out.append(out_bytes, 0);
        return false;
    }

    // 1) Try arena offset path (common for some internal indices)
    const char *ptr = nullptr;
    try
    {
        ptr = m->arena ? m->arena->blob_ptr_from_offset_for_map(id_or_offset) : nullptr;
    }
    catch (...)
    {
        ptr = nullptr;
    }

    if (ptr)
    {
        // blob header is uint32_t length at ptr
        uint32_t stored_len = *reinterpret_cast<const uint32_t *>(ptr);
        const char *payload = ptr + sizeof(uint32_t);

        // Determine element size from membrance config
        pomai::core::DataType dt = m->data_type;
        size_t elem_size = pomai::core::dtype_size(dt);
        size_t required = elem_size * dim;

        if (static_cast<size_t>(stored_len) >= required)
        {
            // decode to float32 and append
            std::vector<float> tmp(dim);
            decode_slot_to_float(payload, dim, dt, tmp.data());
            out.append(reinterpret_cast<const char *>(tmp.data()), out_bytes);
            return true;
        }
        // If stored_len too small, fallthrough to other paths (orbit/zero)
    }

    // 2) Fallback: treat id_or_offset as label -> retrieve via Orbit API
    try
    {
        std::vector<float> vec;
        if (m->orbit && m->orbit->get(id_or_offset, vec) && vec.size() == dim)
        {
            out.append(reinterpret_cast<const char *>(vec.data()), out_bytes);
            return true;
        }
    }
    catch (...)
    {
        // swallow and fall through to zero padding
    }

    // 3) Nothing found -> append zeros
    out.append(out_bytes, 0);
    return false;
}

bool fetch_vector(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<float> &out_vec)
{
    out_vec.clear();
    size_t dim = m ? m->dim : 0;
    if (dim == 0)
    {
        return false;
    }

    // 1) Orbit (id treated as label)
    try
    {
        if (m->orbit)
        {
            if (m->orbit->get(id_or_offset, out_vec) && out_vec.size() == dim)
                return true;
        }
    }
    catch (...)
    { /* ignore and continue */ }

    // 2) Arena offset / remote id
    try
    {
        if (m->arena)
        {
            const char *p = m->arena->blob_ptr_from_offset_for_map(id_or_offset);
            if (p)
            {
                uint32_t stored_len = *reinterpret_cast<const uint32_t *>(p);
                const char *payload = p + sizeof(uint32_t);

                pomai::core::DataType dt = m->data_type;
                size_t elem_size = pomai::core::dtype_size(dt);
                size_t required = elem_size * dim;

                if (static_cast<size_t>(stored_len) >= required)
                {
                    out_vec.assign(dim, 0.0f);
                    decode_slot_to_float(payload, dim, dt, out_vec.data());
                    return true;
                }
            }
        }
    }
    catch (...)
    { /* ignore */ }

    // 3) Ordinal mapping: if id_or_offset is small, treat as ordinal index into orbit's label list
    try
    {
        if (m->orbit)
        {
            auto all_labels = m->orbit->get_all_labels(); // may throw or be empty
            if (!all_labels.empty())
            {
                uint64_t ord = id_or_offset;
                if (ord < all_labels.size())
                {
                    uint64_t label = all_labels[ord];
                    if (m->orbit->get(label, out_vec) && out_vec.size() == dim)
                        return true;
                }
            }
        }
    }
    catch (...)
    { /* ignore */ }

    // Not found -> zero vector
    out_vec.assign(dim, 0.0f);
    return false;
}

std::string make_header(const char *tag, size_t cnt, size_t dim, size_t bytes)
{
    std::ostringstream h;
    h << tag << " " << cnt << " " << dim << " " << bytes << "\n";
    return h.str();
}

} // namespace pomai::server::data_supplier