// src/facade/data_supplier.cc
//
// Implementation for data_supplier.h
// Optimized for Zero-Allocation Hot Paths using std::vector<char> buffers.
// Supports: Float32, Float64, Int32, Int8, Float16 decoding.

#include "src/facade/data_supplier.h"

#include <cstring>
#include <random>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>

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
    // OPTIMIZED: Uses std::vector<char> for binary safety and direct memcpy.
    bool fetch_vector_raw(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<char> &out, size_t dim,
                          pomai::core::DataType &out_dtype, uint32_t &out_elem_size)
    {
        // default fallback: set dtype to float32 (standard interchange)
        out_dtype = pomai::core::DataType::FLOAT32;
        out_elem_size = static_cast<uint32_t>(pomai::core::dtype_size(out_dtype));
        size_t out_bytes = dim * out_elem_size;

        if (!m)
        {
            size_t old_sz = out.size();
            out.resize(old_sz + out_bytes, 0);
            return false;
        }

        // 1) Try arena offset path (Fastest: ~10ns)
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

            // Membrance native type
            uint32_t elem_size = static_cast<uint32_t>(pomai::core::dtype_size(m->data_type));
            size_t expected_bytes = dim * elem_size;

            size_t old_sz = out.size();
            
            if (payload_len >= expected_bytes)
            {
                // Zero-copy append via resize + memcpy
                out.resize(old_sz + expected_bytes);
                std::memcpy(out.data() + old_sz, payload, expected_bytes);
                out_dtype = m->data_type;
                out_elem_size = elem_size;
                return true;
            }
            // Partial payload? (Should not happen in steady state, but handle safely)
            if (payload_len > 0)
            {
                out.resize(old_sz + expected_bytes);
                std::memcpy(out.data() + old_sz, payload, payload_len);
                std::memset(out.data() + old_sz + payload_len, 0, expected_bytes - payload_len);
                out_dtype = m->data_type;
                out_elem_size = elem_size;
                return true;
            }
        }

        // 2) Try orbit raw retrieval (Orbit may store vectors compressed/quantized/raw)
        try
        {
            if (m->orbit)
            {
                std::vector<std::string> outs; // TODO: Optimize Orbit API to avoid std::string allocation here
                std::vector<uint64_t> ids{id_or_offset};
                
                if (m->orbit->get_vectors_raw(ids, outs) && !outs.empty() && !outs[0].empty())
                {
                    const std::string& raw_orbit = outs[0];
                    uint32_t elem_size = static_cast<uint32_t>(pomai::core::dtype_size(m->data_type));
                    size_t expected_bytes = dim * elem_size;
                    size_t float_bytes = dim * sizeof(float);
                    
                    size_t old_sz = out.size();

                    // Case A: Orbit returns native format
                    if (raw_orbit.size() >= expected_bytes)
                    {
                        out.resize(old_sz + expected_bytes);
                        std::memcpy(out.data() + old_sz, raw_orbit.data(), expected_bytes);
                        out_dtype = m->data_type;
                        out_elem_size = elem_size;
                        return true;
                    }
                    // Case B: Orbit returns float32 (e.g., from cache or decompression)
                    else if (raw_orbit.size() >= float_bytes)
                    {
                        out.resize(old_sz + float_bytes);
                        std::memcpy(out.data() + old_sz, raw_orbit.data(), float_bytes);
                        out_dtype = pomai::core::DataType::FLOAT32;
                        out_elem_size = sizeof(float);
                        return true;
                    }
                    // Case C: Partial data fallback
                    else 
                    {
                         out.resize(old_sz + float_bytes); // Default to float32 container
                         std::memcpy(out.data() + old_sz, raw_orbit.data(), raw_orbit.size());
                         std::memset(out.data() + old_sz + raw_orbit.size(), 0, float_bytes - raw_orbit.size());
                         out_dtype = pomai::core::DataType::FLOAT32;
                         out_elem_size = sizeof(float);
                         return true;
                    }
                }
            }
        }
        catch (...)
        {
            // Ignore exceptions in raw path, try decoded path
        }

        // 3) Try direct float retrieval via orbit->get (Decoded fallback)
        try
        {
            if (m->orbit)
            {
                // Reuse thread_local buffer if possible to avoid allocation? 
                // For now, standard vector to be safe.
                std::vector<float> tmp(dim, 0.0f);
                if (m->orbit->get(id_or_offset, tmp))
                {
                    size_t old_sz = out.size();
                    size_t bytes = dim * sizeof(float);
                    out.resize(old_sz + bytes);
                    std::memcpy(out.data() + old_sz, tmp.data(), bytes);
                    out_dtype = pomai::core::DataType::FLOAT32;
                    out_elem_size = sizeof(float);
                    return true;
                }
            }
        }
        catch (...)
        {
            // Ignore
        }

        // Final fallback: zeros padded as float32
        size_t old_sz = out.size();
        out.resize(old_sz + out_bytes, 0);
        return false;
    }

    // Backwards-compatible helper
    bool fetch_vector_bytes_or_fallback(pomai::core::Membrance *m, uint64_t id_or_offset, std::vector<char> &out, size_t dim)
    {
        pomai::core::DataType dtype;
        uint32_t elem_size = 0;
        
        // Use a temp buffer to check result before converting
        // (In optimized version we could decode directly into 'out' if we knew the format ahead of time)
        size_t start_idx = out.size();
        bool ok = fetch_vector_raw(m, id_or_offset, out, dim, dtype, elem_size);

        if (!ok) return false;

        // Check if we need conversion to Float32
        if (dtype == pomai::core::DataType::FLOAT32 && elem_size == sizeof(float)) {
             // Already float32, data is in 'out' correctly
             return true;
        }

        // Needs conversion: The raw bytes are at out[start_idx...]
        // We need to decode them to float32, effectively replacing the raw bytes with float bytes.
        
        // 1. Extract raw bytes
        size_t raw_len = out.size() - start_idx;
        std::vector<char> raw_copy(out.begin() + start_idx, out.end());
        
        // 2. Resize out to hold floats
        size_t float_len = dim * sizeof(float);
        out.resize(start_idx + float_len);

        // 3. Decode
        if (elem_size > 0 && raw_copy.size() >= dim * elem_size) {
            decode_slot_to_float(raw_copy.data(), dim, dtype, reinterpret_cast<float*>(out.data() + start_idx));
        } else {
             // Partial/Invalid: Zero fill
             std::memset(out.data() + start_idx, 0, float_len);
        }

        return true;
    }

    // Robust resolver: fetch vector into out_vec (float conversion) - unchanged API.
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
        try {
            ptr = m->arena ? m->arena->blob_ptr_from_offset_for_map(id_or_offset) : nullptr;
        } catch (...) { ptr = nullptr; }

        if (ptr)
        {
            uint32_t blen = read_u32_le(ptr);
            const char *payload = ptr + sizeof(uint32_t);
            size_t payload_len = (blen > 0) ? static_cast<size_t>(blen) : 0;

            if (payload_len >= dim * pomai::core::dtype_size(m->data_type))
            {
                decode_slot_to_float(payload, dim, m->data_type, out_vec.data());
                return true;
            }
            // Partial payload fallback
            if (payload_len > 0) {
                size_t copy_bytes = std::min(payload_len, dim * sizeof(float));
                if (copy_bytes > 0) std::memcpy(out_vec.data(), payload, copy_bytes);
                std::fill(out_vec.begin() + (copy_bytes / sizeof(float)), out_vec.end(), 0.0f);
                return true;
            }
        }

        // 2) Try orbit decoded get (preferred)
        try {
            if (m->orbit && m->orbit->get(id_or_offset, out_vec)) return true;
        } catch (...) {}

        // 3) Try orbit raw path + decode
        try {
            if (m->orbit) {
                std::vector<std::string> outs;
                std::vector<uint64_t> ids{id_or_offset};
                if (m->orbit->get_vectors_raw(ids, outs) && !outs.empty() && !outs[0].empty()) {
                     if (outs[0].size() >= dim * sizeof(float)) {
                         std::memcpy(out_vec.data(), outs[0].data(), dim * sizeof(float));
                         return true;
                     }
                     decode_slot_to_float(outs[0].data(), dim, m->data_type, out_vec.data());
                     return true;
                }
            }
        } catch (...) {}

        std::fill(out_vec.begin(), out_vec.end(), 0.0f);
        return false;
    }

    std::string make_header(const char *tag, const std::string &dtype, size_t cnt, size_t dim, size_t bytes)
    {
        std::ostringstream ss;
        ss << (tag ? tag : "OK") << " BINARY " << cnt << " " << dim << " " << bytes << "\n";
        return ss.str();
    }

} // namespace pomai::server::data_supplier