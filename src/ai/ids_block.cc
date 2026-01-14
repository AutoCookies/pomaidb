/*
 * src/ai/ids_block.cc
 *
 * Small TU implementing helpers declared in ids_block.h.
 */

#include "src/ai/ids_block.h"
#include "src/ai/atomic_utils.h" // atomic helpers for mmap'd memory

#include <atomic>

namespace pomai::ai::soa
{

    // try_pack implementations: avoid silent truncation when value does not fit in 62 bits.
    bool IdEntry::try_pack_local_offset(uint64_t offset, uint64_t &out) noexcept
    {
        if (!fits_payload(offset))
            return false;
        out = pack_local_offset(offset);
        return true;
    }

    bool IdEntry::try_pack_remote_id(uint64_t remote_id, uint64_t &out) noexcept
    {
        if (!fits_payload(remote_id))
            return false;
        out = pack_remote_id(remote_id);
        return true;
    }

    bool IdEntry::try_pack_label(uint64_t label, uint64_t &out) noexcept
    {
        if (!fits_payload(label))
            return false;
        out = pack_label(label);
        return true;
    }

    // Atomic helpers: thin wrappers that use atomic_utils to be robust on mmap'd memory.
    void atomic_store_entry(uint64_t *ptr, uint64_t v) noexcept
    {
        if (!ptr)
            return;
        pomai::ai::atomic_utils::atomic_store_u64(ptr, v);
    }

    uint64_t atomic_load_entry(const uint64_t *ptr) noexcept
    {
        if (!ptr)
            return 0;
        return pomai::ai::atomic_utils::atomic_load_u64(ptr);
    }

    bool atomic_compare_exchange_entry(uint64_t *ptr, uint64_t &expected, uint64_t desired) noexcept
    {
        if (!ptr)
            return false;
        // Use GCC/Clang builtin CAS which works with possibly unaligned mmap'd memory.
        // We choose __ATOMIC_SEQ_CST for conservative semantics (caller can build lighter if desired).
        return __atomic_compare_exchange_n(ptr, &expected, desired, /*weak=*/false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }

} // namespace pomai::ai::soa