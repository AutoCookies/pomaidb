// tests/ids/ids_block_test.cc
#include "src/ai/ids_block.h"
#include <cassert>
#include <iostream>

int main()
{
    using namespace pomai::ai::soa;

    uint64_t lbl = 0x12345678ULL;
    uint64_t packed = IdEntry::pack_label(lbl);
    assert(IdEntry::is_label(packed));
    uint64_t unpack = IdEntry::unpack_label(packed);
    assert(unpack == (lbl & IdEntry::PAYLOAD_MASK));

    uint64_t off = 0xABCDEFULL;
    uint64_t p2 = IdEntry::pack_local_offset(off);
    assert(IdEntry::is_local_offset(p2));
    uint64_t off2 = IdEntry::unpack_local_offset(p2);
    assert(off2 == (off & IdEntry::PAYLOAD_MASK));

    uint64_t rid = 0x5555ULL;
    uint64_t p3 = IdEntry::pack_remote_id(rid);
    assert(IdEntry::is_remote_id(p3));
    uint64_t rid2 = IdEntry::unpack_remote_id(p3);
    assert(rid2 == (rid & IdEntry::PAYLOAD_MASK));

    std::cout << "ids_block tests OK\n";
    return 0;
}