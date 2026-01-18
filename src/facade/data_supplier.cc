// src/facade/data_supplier.cc
//
// Implementation for data_supplier.h
// This file preserves the original logic extracted from server.h without modification.

#include "src/facade/data_supplier.h"

#include <cstring>
#include <random>
#include <sstream>
#include <iostream>

#include "src/core/pomai_db.h"       // for Membrance
#include "src/memory/append_only_arena.h"
#include "src/memory/shard_arena.h"

namespace pomai::server::data_supplier
{

bool fetch_vector_bytes_or_fallback(pomai::core::Membrance *m, uint64_t id_or_offset, std::string &out, size_t dim)
{
    size_t bytes_vec = dim * sizeof(float);

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
        // append payload (skip header)
        out.append(ptr + sizeof(uint32_t), bytes_vec);
        return true;
    }

    // 2) Fallback: treat id_or_offset as label -> retrieve via Orbit API
    try
    {
        std::vector<float> vec;
        if (m->orbit && m->orbit->get(id_or_offset, vec) && vec.size() == dim)
        {
            out.append(reinterpret_cast<const char *>(vec.data()), bytes_vec);
            return true;
        }
    }
    catch (...)
    {
        // swallow and fall through to zero padding
    }

    // 3) Nothing found -> append zeros
    out.append(bytes_vec, 0);
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

    try
    {
        // 1) Orbit (id treated as label)
        if (m->orbit)
        {
            if (m->orbit->get(id_or_offset, out_vec) && out_vec.size() == dim)
                return true;
        }
    }
    catch (...)
    { /* ignore and continue */ }

    try
    {
        // 2) Arena offset / remote id
        if (m->arena)
        {
            const char *p = m->arena->blob_ptr_from_offset_for_map(id_or_offset);
            if (p)
            {
                out_vec.assign(dim, 0.0f);
                std::memcpy(out_vec.data(), p + sizeof(uint32_t), dim * sizeof(float));
                return true;
            }
        }
    }
    catch (...)
    { /* ignore */ }

    try
    {
        // 3) Ordinal mapping: if id_or_offset is small, treat as ordinal index into orbit's label list
        if (m->orbit)
        {
            // optimistic: orbit may provide get_all_labels(); guard with try/catch
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