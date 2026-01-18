// src/facade/data_supplier.cc
//
// Implementation for data_supplier.h
// Added fetch_vector_raw that returns original stored bytes + dtype info.
// Existing functions preserved for backward compatibility.

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

// New: fetch raw bytes + dtype info. Does NOT decode into float32.
// It will append exactly elem_size * dim bytes (if available) to `out`.
// If a blob exists in arena with sufficient bytes -> return raw payload bytes and dtype from membrance config.
// Else if Orbit can provide the vector only as floats (orbit.get returns floats) -> we will fall back to
// converting those floats into the membrance's configured storage dtype only if we have to. But preference is to return raw storage.
bool fetch_vector_raw(pomai::core::Membrance *m, uint64_t id_or_offset, std::string &out, size_t dim,
                      pomai::core::DataType &out_dtype, uint32_t &out_elem_size)
{
    // default fallback: return zeros as float32 (compatibility) and set dtype to float32
    out_dtype = pomai::core::DataType::FLOAT32;
    out_elem_size = static_cast<uint32_t>(pomai::core::dtype_size(out_dtype));
    size_t out_bytes = dim * out_elem_size;

    if (!m)
    {
        out.append(out_bytes, 0);
        return false;
    }

    // Prefer HotTier raw data if available (hot entries are in configured data_type and raw bytes)
    try
    {
        if (m->hot_tier && m->hot_tier->size() > 0)
        {
            // HotTier stores raw bytes in its buffer for recent pushes.
            // However HotTier is an in-memory aggregator; we can't directly map id -> offset here
            // so we fallthrough to arena/orbit paths (HotTier swap_and_flush is used in background worker).
        }
    }
    catch (...)
    { /* ignore */ }

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
        uint32_t stored_len = *reinterpret_cast<const uint32_t *>(ptr);
        const char *payload = ptr + sizeof(uint32_t);

        // Element size comes from membrance configuration
        out_dtype = m->data_type;
        out_elem_size = static_cast<uint32_t>(pomai::core::dtype_size(out_dtype));
        size_t required = static_cast<size_t>(out_elem_size) * dim;

        if (static_cast<size_t>(stored_len) >= required)
        {
            // append raw bytes exactly required bytes
            out.append(payload, required);
            return true;
        }
        // else fallthrough
    }

    // 2) Fallback: if Orbit can provide raw storage (most Orbit APIs return float vectors),
    // we will request orbit->get_raw_blob() if such API existed; since it doesn't, prefer orbit->get() -> floats
    // and then convert to target storage representation (this is the only path where conversion is unavoidable).
    try
    {
        std::vector<float> vec;
        if (m->orbit && m->orbit->get(id_or_offset, vec) && vec.size() == dim)
        {
            // convert floats to membrance configured dtype and append raw bytes
            out_dtype = m->data_type;
            out_elem_size = static_cast<uint32_t>(pomai::core::dtype_size(out_dtype));
            size_t required = static_cast<size_t>(out_elem_size) * dim;
            std::string buf;
            buf.resize(required);
            char *dst = &buf[0];

            switch (out_dtype)
            {
                case pomai::core::DataType::FLOAT32:
                    std::memcpy(dst, vec.data(), dim * sizeof(float));
                    break;
                case pomai::core::DataType::FLOAT64:
                {
                    double *dptr = reinterpret_cast<double *>(dst);
                    for (size_t i = 0; i < dim; ++i) dptr[i] = static_cast<double>(vec[i]);
                    break;
                }
                case pomai::core::DataType::INT32:
                {
                    int32_t *ip = reinterpret_cast<int32_t *>(dst);
                    for (size_t i = 0; i < dim; ++i) ip[i] = static_cast<int32_t>(std::lrintf(vec[i]));
                    break;
                }
                case pomai::core::DataType::INT8:
                {
                    int8_t *ip = reinterpret_cast<int8_t *>(dst);
                    for (size_t i = 0; i < dim; ++i)
                    {
                        int32_t v = static_cast<int32_t>(std::lrintf(vec[i]));
                        v = std::max<int32_t>(-128, std::min<int32_t>(127, v));
                        ip[i] = static_cast<int8_t>(v);
                    }
                    break;
                }
                case pomai::core::DataType::FLOAT16:
                {
                    uint16_t *hp = reinterpret_cast<uint16_t *>(dst);
                    for (size_t i = 0; i < dim; ++i) hp[i] = fp32_to_fp16(vec[i]);
                    break;
                }
                default:
                    std::memcpy(dst, vec.data(), std::min<size_t>(required, dim * sizeof(float)));
                    if (required > dim * sizeof(float))
                        std::memset(dst + dim * sizeof(float), 0, required - dim * sizeof(float));
                    break;
            }
            out.append(buf.data(), required);
            return true;
        }
    }
    catch (...)
    { /* ignore */ }

    // 3) Not found anywhere -> append zeros in membrance configured dtype (to keep header semantics)
    out_dtype = m->data_type;
    out_elem_size = static_cast<uint32_t>(pomai::core::dtype_size(out_dtype));
    out.append(dim * out_elem_size, 0);
    return false;
}

// Backwards-compatible: existing function that decodes to float32 bytes (keeps old behaviour if other code relies on it)
bool fetch_vector_bytes_or_fallback(pomai::core::Membrance *m, uint64_t id_or_offset, std::string &out, size_t dim)
{
    // This helper will try to fetch raw and then decode to float32.
    pomai::core::DataType dt;
    uint32_t elem_size;
    std::string raw;
    bool ok = fetch_vector_raw(m, id_or_offset, raw, dim, dt, elem_size);

    // If returned dtype is already float32 and elem_size == 4 -> append raw directly
    if (dt == pomai::core::DataType::FLOAT32 && elem_size == 4)
    {
        out.append(raw);
        return ok;
    }

    // Otherwise decode raw to float32
    std::vector<float> tmp(dim, 0.0f);
    if (!raw.empty())
    {
        // raw might be exactly elem_size * dim bytes
        decode_slot_to_float(raw.data(), dim, dt, tmp.data());
    }
    out.append(reinterpret_cast<const char *>(tmp.data()), dim * sizeof(float));
    return ok;
}

bool fetch_vector(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<float> &out_vec)
{
    out_vec.clear();
    size_t dim = m ? m->dim : 0;
    if (dim == 0)
    {
        return false;
    }

    // Try orbit first (returns floats)
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

    // Try arena raw and decode
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

    // Ordinal mapping fallback: try orbit labels as before
    try
    {
        if (m->orbit)
        {
            auto all_labels = m->orbit->get_all_labels();
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

    // Not found
    out_vec.assign(dim, 0.0f);
    return false;
}

std::string make_header(const char *tag, const std::string &dtype, size_t cnt, size_t dim, size_t bytes)
{
    std::ostringstream h;
    h << tag << " " << dtype << " " << cnt << " " << dim << " " << bytes << "\n";
    return h.str();
}

} // namespace pomai::server::data_supplier