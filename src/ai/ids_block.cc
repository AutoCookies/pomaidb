/*
 * src/ai/ids_block.cc
 */

#include "src/ai/ids_block.h"
#include "src/ai/atomic_utils.h"
#include <atomic>

namespace pomai::ai::soa
{
    // Implementation of safe packing helpers defined in struct IdEntry

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

    // Atomic helpers wrappers
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
        return pomai::ai::atomic_utils::atomic_compare_exchange_u64(ptr, expected, desired);
    }

} // namespace pomai::ai::soa