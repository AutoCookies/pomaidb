#include "src/ai/ids_block.h"
#include "src/ai/atomic_utils.h"
#include <atomic>
#include <cassert>

namespace pomai::ai::soa
{
    bool IdEntry::try_pack_local_offset(uint64_t offset, uint64_t &out) noexcept
    {
        if (!fits_payload(offset))
            return false;
        out = TAG_LOCAL | (offset & PAYLOAD_MASK);
        return true;
    }

    bool IdEntry::try_pack_remote_id(uint64_t remote_id, uint64_t &out) noexcept
    {
        if (!fits_payload(remote_id))
            return false;
        out = TAG_REMOTE | (remote_id & PAYLOAD_MASK);
        return true;
    }

    bool IdEntry::try_pack_label(uint64_t label, uint64_t &out) noexcept
    {
        if (!fits_payload(label))
            return false;
        out = TAG_LABEL | (label & PAYLOAD_MASK);
        return true;
    }

    void IdEntry::atomic_store(uint64_t *ptr, uint64_t val) noexcept
    {
        assert(ptr != nullptr && "Atomic store to nullptr");
        pomai::ai::atomic_utils::atomic_store_u64(ptr, val);
    }

    uint64_t IdEntry::atomic_load(const uint64_t *ptr) noexcept
    {
        assert(ptr != nullptr && "Atomic load from nullptr");
        return pomai::ai::atomic_utils::atomic_load_u64(ptr);
    }

    bool IdEntry::atomic_compare_exchange(uint64_t *ptr, uint64_t &expected, uint64_t desired) noexcept
    {
        assert(ptr != nullptr && "Atomic CAS on nullptr");
        return pomai::ai::atomic_utils::atomic_compare_exchange_u64(ptr, expected, desired);
    }

} // namespace pomai::ai::soa