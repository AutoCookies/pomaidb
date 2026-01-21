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
    void expect(bool cond, const std::string &msg)
    {
        if (cond)
        {
            std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << msg << "\n";
            ++passed;
        }
        else
        {
            std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << msg << "\n";
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

    // Basic invariants
    r.expect(IdEntry::PAYLOAD_MASK != 0, "PAYLOAD_MASK non-zero");
    r.expect(IdEntry::TAG_MASK != 0, "TAG_MASK non-zero");
    r.expect((IdEntry::TAG_MASK & IdEntry::PAYLOAD_MASK) == 0, "TAG and PAYLOAD masks are disjoint");

    // A safe label at the boundary should fit (largest allowed payload)
    uint64_t safe_label = IdEntry::PAYLOAD_MASK;
    uint64_t out_packed = 0;
    bool ok = IdEntry::try_pack_label(safe_label, out_packed);
    r.expect(ok, "try_pack_label succeeds for PAYLOAD_MASK boundary");
    if (ok)
    {
        r.expect(IdEntry::tag_of(out_packed) == IdEntry::TAG_LABEL, "packed tag == TAG_LABEL");
        r.expect(IdEntry::payload_of(out_packed) == safe_label, "packed payload equals original");
    }

    // An overflow label should be rejected by try_pack_label
    uint64_t overflow_label = IdEntry::PAYLOAD_MASK + 1ULL;
    uint64_t tmp = 0;
    bool ok2 = IdEntry::try_pack_label(overflow_label, tmp);
    r.expect(!ok2, "try_pack_label rejects overflow label");

    // Sanity: fits_payload should reflect the same
    r.expect(IdEntry::fits_payload(safe_label), "fits_payload true for safe_label");
    r.expect(!IdEntry::fits_payload(overflow_label), "fits_payload false for overflow_label");

    // IMPORTANT: callers must not call pack_label(...) with an unchecked label.
    // This test demonstrates the non-asserting path (try_pack_label) and ensures it behaves as expected.

    return r.summary();
}