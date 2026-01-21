#include "src/ai/ids_block.h"

#include <iostream>
#include <cstdint>
#include <string>

static const char *ANSI_GREEN = "\033[32m";
static const char *ANSI_RED = "\033[31m";
static const char *ANSI_RESET = "\033[0m";

struct Runner
{
    int passed = 0;
    int failed = 0;
    void expect(bool cond, const char *name)
    {
        if (cond)
        {
            std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << name << "\n";
            ++passed;
        }
        else
        {
            std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << name << "\n";
            ++failed;
        }
    }
    int summary()
    {
        std::cout << "\nResults: " << passed << " passed, " << failed << " failed.\n";
        return failed == 0 ? 0 : 1;
    }
};

int main()
{
    using namespace pomai::ai::soa;
    Runner r;

    // Create an oversized label that doesn't fit into PAYLOAD_BITS (use full 64-bit 1s)
    uint64_t huge = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t packed = IdEntry::pack_label(huge);

    // Should not abort; tag must be TAG_LABEL
    r.expect(IdEntry::tag_of(packed) == IdEntry::TAG_LABEL, "pack_label returns TAG_LABEL (no abort)");

    // Payload should equal masked value
    uint64_t expected_payload = huge & IdEntry::PAYLOAD_MASK;
    r.expect(IdEntry::payload_of(packed) == expected_payload, "pack_label masks payload to PAYLOAD_MASK");

    // try_pack_label should return false for overflow (semantic)
    uint64_t out = 0;
    bool try_ok = IdEntry::try_pack_label(huge, out);
    r.expect(!try_ok, "try_pack_label indicates overflow (returns false)");

    // But try_pack_label still writes nothing; pack_label remains usable and deterministic
    uint64_t small = IdEntry::PAYLOAD_MASK;
    uint64_t p2 = IdEntry::pack_label(small);
    r.expect(IdEntry::payload_of(p2) == small, "pack_label preserves payload when fits exactly");

    return r.summary();
}