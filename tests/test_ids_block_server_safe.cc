#include "src/ai/ids_block.h"
#include <iostream>
#include <cstdint>
#include <vector>
#include <cstring>

using namespace pomai::ai::soa;

static const char *GREEN = "\033[32m";
static const char *RED = "\033[31m";
static const char *RESET = "\033[0m";

struct Runner
{
    int pass = 0;
    int fail = 0;
    void expect(bool c, const char *name)
    {
        if (c)
        {
            std::cout << GREEN << "[PASS] " << RESET << name << "\n";
            ++pass;
        }
        else
        {
            std::cout << RED << "[FAIL] " << RESET << name << "\n";
            ++fail;
        }
    }
    int summary()
    {
        std::cout << "\nResults: " << pass << " passed, " << fail << " failed.\n";
        return fail == 0 ? 0 : 1;
    }
};

int main()
{
    Runner r;

    // Basic mask sanity
    r.expect(IdEntry::PAYLOAD_MASK != 0, "PAYLOAD_MASK non-zero");
    r.expect(IdEntry::TAG_MASK != 0, "TAG_MASK non-zero");
    r.expect((IdEntry::TAG_MASK & IdEntry::PAYLOAD_MASK) == 0, "TAG and PAYLOAD masks are disjoint");

    // Boundary label that exactly fills payload
    uint64_t safe_label = IdEntry::PAYLOAD_MASK;
    uint64_t packed = 0;
    bool ok = IdEntry::try_pack_label(safe_label, packed);
    r.expect(ok, "try_pack_label succeeds for PAYLOAD_MASK boundary");
    r.expect(IdEntry::tag_of(packed) == IdEntry::TAG_LABEL, "packed tag == TAG_LABEL");
    r.expect(IdEntry::payload_of(packed) == safe_label, "packed payload equals original");

    // Overflow label (one past payload mask)
    uint64_t overflow_label = IdEntry::PAYLOAD_MASK + 1;
    uint64_t out = 0;
    bool ok2 = IdEntry::try_pack_label(overflow_label, out);
    r.expect(!ok2, "try_pack_label rejects overflow label");

    // Confirm pack_label works for safe value (do not call pack_label with overflow - it asserts)
    bool pack_ok = false;
    try
    {
        uint64_t p = IdEntry::pack_label(safe_label); // safe
        pack_ok = (IdEntry::tag_of(p) == IdEntry::TAG_LABEL) && (IdEntry::payload_of(p) == safe_label);
    }
    catch (...)
    {
        pack_ok = false;
    }
    r.expect(pack_ok, "pack_label works for safe payload boundary");

    return r.summary();
}