// src/facade/data_supplier.cc
//
// Implementation for data_supplier.h
// Removed dependency on HotTier. fetch_vector_raw now prefers arena offsets,
// then Orbit raw retrieval, then falls back to decoding floats via Orbit::get.
// Backwards-compatible helpers preserved.

#include "src/facade/data_supplier.h"

#include <cstring>
#include <random>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "src/core/pomai_db.h" // for Membrance
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
            for (size_t i = 0; i < dim; ++i)
                out[i] = static_cast<float>(dp[i]);
            break;
        }

        case DataType::INT32:
        {
            const int32_t *ip = reinterpret_cast<const int32_t *>(payload);
            for (size_t i = 0; i < dim; ++i)
                out[i] = static_cast<float>(ip[i]);
            break;
        }

        case DataType::INT8:
        {
            const int8_t *ip = reinterpret_cast<const int8_t *>(payload);
            for (size_t i = 0; i < dim; ++i)
                out[i] = static_cast<float>(ip[i]);
            break;
        }

        case DataType::FLOAT16:
        {
            const uint16_t *hp = reinterpret_cast<const uint16_t *>(payload);
            for (size_t i = 0; i < dim; ++i)
                out[i] = fp16_to_fp32(hp[i]);
            break;
        }

        default:
            // Fallback: try to memcpy as float32 up to available bytes
            std::memcpy(out, payload, dim * sizeof(float));
            break;
        }
    }

    // Helper: safe read of 32-bit length prefix (stored as little-endian uint32)
    static inline uint32_t read_u32_le(const char *p)
    {
        uint32_t v = 0;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }

    // New: fetch raw bytes + dtype info. Does NOT decode into float32.
    // It will append exactly elem_size * dim bytes (if available) to `out`.
    // If a blob exists in arena with sufficient bytes -> return raw payload bytes and dtype from membrance config.
    // Else if Orbit can provide the vector only as floats (orbit.get returns floats) -> we will fetch floats and
    // return them as float32 raw bytes (set out_dtype accordingly). Returns true when a live vector was found;
    // false when we appended zero-padding fallback.
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
            // blob header stored as 4-byte length at head
            uint32_t blen = read_u32_le(ptr);
            const char *payload = ptr + sizeof(uint32_t);
            size_t payload_len = (blen > 0) ? static_cast<size_t>(blen) : 0;

            // Best-effort: if payload is at least expected size, return raw bytes (assume stored as membrance.data_type)
            uint32_t elem_size = static_cast<uint32_t>(pomai::core::dtype_size(m->data_type));
            size_t expected_bytes = dim * elem_size;
            if (payload_len >= expected_bytes)
            {
                out.append(payload, static_cast<size_t>(expected_bytes));
                out_dtype = m->data_type;
                out_elem_size = elem_size;
                return true;
            }
            // if payload shorter, we still return what we have padded to expected_bytes
            if (payload_len > 0)
            {
                out.append(payload, payload_len);
                if (expected_bytes > payload_len)
                    out.append(expected_bytes - payload_len, '\0');
                out_dtype = m->data_type;
                out_elem_size = elem_size;
                return true;
            }
            // otherwise fallthrough to orbit / fallback
        }

        // 2) Try orbit raw retrieval (Orbit may store vectors on disk/out-of-line). Use get_vectors_raw API for single id.
        try
        {
            if (m->orbit)
            {
                std::vector<std::string> outs;
                std::vector<uint64_t> ids{id_or_offset};
                if (m->orbit->get_vectors_raw(ids, outs))
                {
                    if (!outs.empty() && !outs[0].empty())
                    {
                        // If orbit returns raw with size matching expected storage format -> keep dtype as membrance config
                        uint32_t elem_size = static_cast<uint32_t>(pomai::core::dtype_size(m->data_type));
                        size_t expected_bytes = dim * elem_size;
                        if (outs[0].size() >= expected_bytes)
                        {
                            out.append(outs[0].data(), expected_bytes);
                            out_dtype = m->data_type;
                            out_elem_size = elem_size;
                            return true;
                        }
                        // If orbit returned float32 raw (common), normalize to float32 return
                        size_t float_bytes = dim * sizeof(float);
                        if (outs[0].size() >= float_bytes)
                        {
                            out.append(outs[0].data(), float_bytes);
                            out_dtype = pomai::core::DataType::FLOAT32;
                            out_elem_size = sizeof(float);
                            return true;
                        }
                        // else pad to float32 size
                        if (!outs[0].empty())
                        {
                            out.append(outs[0].data(), outs[0].size());
                            out.append((dim * sizeof(float)) - outs[0].size(), '\0');
                            out_dtype = pomai::core::DataType::FLOAT32;
                            out_elem_size = sizeof(float);
                            return true;
                        }
                    }
                }
            }
        }
        catch (...)
        {
            // ignore and continue to next fallback
        }

        // 3) Try direct float retrieval via orbit->get (returns decoded float vector). Convert to raw float32 bytes.
        try
        {
            if (m->orbit)
            {
                std::vector<float> tmp(dim, 0.0f);
                if (m->orbit->get(id_or_offset, tmp))
                {
                    out.append(reinterpret_cast<const char *>(tmp.data()), dim * sizeof(float));
                    out_dtype = pomai::core::DataType::FLOAT32;
                    out_elem_size = sizeof(float);
                    return true;
                }
            }
        }
        catch (...)
        {
            // ignore
        }

        // Final fallback: zeros padded as float32
        out.append(out_bytes, 0);
        out_dtype = pomai::core::DataType::FLOAT32;
        out_elem_size = static_cast<uint32_t>(pomai::core::dtype_size(out_dtype));
        return false;
    }

    // Backwards-compatible helper kept (decodes into float32 bytes) — still available.
    // This returns bytes containing dim * sizeof(float) float32 values (either directly from storage if already float32,
    // or decoded from stored representation). Returns true if live vector was found, false if zero-padded fallback.
    bool fetch_vector_bytes_or_fallback(pomai::core::Membrance *m, uint64_t id_or_offset, std::string &out, size_t dim)
    {
        pomai::core::DataType dtype;
        uint32_t elem_size = 0;
        std::string raw;
        bool ok = fetch_vector_raw(m, id_or_offset, raw, dim, dtype, elem_size);

        // If raw is already float32 sequence, and elem_size==4, just return normalized buffer
        if (!raw.empty() && dtype == pomai::core::DataType::FLOAT32 && elem_size == sizeof(float))
        {
            // ensure raw length is exactly dim * 4
            if (raw.size() >= dim * sizeof(float))
            {
                out.append(raw.data(), dim * sizeof(float));
                return ok;
            }
        }

        // Otherwise decode raw bytes into float32 vector
        std::vector<float> dec(dim, 0.0f);
        if (!raw.empty())
        {
            // If raw came from arena/orbit as storage dtype, decode accordingly
            if (elem_size > 0 && raw.size() >= dim * elem_size)
            {
                decode_slot_to_float(raw.data(), dim, dtype, dec.data());
                out.append(reinterpret_cast<const char *>(dec.data()), dim * sizeof(float));
                return ok;
            }
        }

        // Last chance: try fetch_vector (decoded path)
        try
        {
            if (fetch_vector(m, id_or_offset, dec))
            {
                out.append(reinterpret_cast<const char *>(dec.data()), dim * sizeof(float));
                return true;
            }
        }
        catch (...)
        {
        }

        // Zero fallback
        out.append(dim * sizeof(float), 0);
        return false;
    }

    // Robust resolver: fetch vector into out_vec (float conversion) - unchanged API.
    // Returns true if a live vector was found, false if zero-filled.
    bool fetch_vector(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<float> &out_vec)
    {
        if (!m)
        {
            std::fill(out_vec.begin(), out_vec.end(), 0.0f);
            return false;
        }

        size_t dim = out_vec.size();
        // 1) Try arena offset path
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
            uint32_t blen = read_u32_le(ptr);
            const char *payload = ptr + sizeof(uint32_t);
            size_t payload_len = (blen > 0) ? static_cast<size_t>(blen) : 0;

            // Decode based on membrance data_type if payload length plausible
            if (payload_len >= dim * pomai::core::dtype_size(m->data_type))
            {
                decode_slot_to_float(payload, dim, m->data_type, out_vec.data());
                return true;
            }

            // if payload exists but shorter, try best-effort decode then pad zeros
            if (payload_len > 0)
            {
                // Try treat as float32 if possible
                size_t copy_bytes = std::min(payload_len, dim * sizeof(float));
                if (copy_bytes > 0)
                    std::memcpy(out_vec.data(), payload, copy_bytes);
                for (size_t i = copy_bytes / sizeof(float); i < dim; ++i)
                    out_vec[i] = 0.0f;
                return true;
            }
        }

        // 2) Try orbit decoded get (preferred)
        try
        {
            if (m->orbit)
            {
                if (m->orbit->get(id_or_offset, out_vec))
                    return true;
            }
        }
        catch (...)
        {
        }

        // 3) Try orbit raw path + decode
        try
        {
            if (m->orbit)
            {
                std::vector<std::string> outs;
                std::vector<uint64_t> ids{id_or_offset};
                if (m->orbit->get_vectors_raw(ids, outs) && !outs.empty() && !outs[0].empty())
                {
                    // If raw seems float32 sized, memcpy
                    if (outs[0].size() >= dim * sizeof(float))
                    {
                        std::memcpy(out_vec.data(), outs[0].data(), dim * sizeof(float));
                        return true;
                    }
                    // else attempt to decode treating as membrance->data_type
                    decode_slot_to_float(outs[0].data(), dim, m->data_type, out_vec.data());
                    return true;
                }
            }
        }
        catch (...)
        {
        }

        // Final fallback: zero fill
        std::fill(out_vec.begin(), out_vec.end(), 0.0f);
        return false;
    }

    std::string make_header(const char *tag, const std::string &dtype, size_t cnt, size_t dim, size_t bytes)
    {
        // Legacy header format preserved. dtype parameter exists for future compatibility but is not printed.
        std::ostringstream ss;
        ss << (tag ? tag : "OK") << " BINARY " << cnt << " " << dim << " " << bytes << "\n";
        return ss.str();
    }

} // namespace pomai::server::data_supplier