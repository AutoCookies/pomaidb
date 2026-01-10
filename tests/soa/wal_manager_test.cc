/*
 * tests/soa/wal_manager_test.cc
 *
 * Basic unit test for WalManager append + replay + truncate.
 *
 * This is a standalone test you can compile and run; it does not integrate
 * with SoaIdsManager directly but validates the WAL format and replay logic.
 */

#include "src/memory/wal_manager.h"

#include <iostream>
#include <vector>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>

using namespace pomai::memory;

struct IdsUpdate
{
    uint64_t idx;
    uint64_t value;
};

int main()
{
    std::string tmpdir = "/tmp";
    std::string walpath = tmpdir + "/test_pomai_wal.bin";

    unlink(walpath.c_str());

    WalManager::WalConfig cfg;
    cfg.sync_on_append = true;

    WalManager wal;
    if (!wal.open(walpath, true, cfg))
    {
        std::cerr << "wal.open failed\n";
        return 2;
    }

    std::vector<IdsUpdate> ins = {
        {10, 0xDEADBEEFDEADBEEFull},
        {400, 0x1234567890ABCDEFULL},
    };

    for (const auto &u : ins)
    {
        auto seq = wal.append_record(WAL_REC_IDS_UPDATE, &u, sizeof(u));
        if (!seq)
        {
            std::cerr << "append_record failed\n";
            return 3;
        }
        std::cout << "appended seq=" << *seq << " idx=" << u.idx << "\n";
    }

    // close and reopen to test replay path opening existing file
    wal.close();
    if (!wal.open(walpath, false, cfg))
    {
        std::cerr << "wal reopen failed\n";
        return 4;
    }

    std::vector<IdsUpdate> seen;
    bool ok = wal.replay([&](uint16_t type, const void *payload, uint32_t len, uint64_t seq) -> bool
                         {
        if (type == WAL_REC_IDS_UPDATE && payload && len == sizeof(IdsUpdate))
        {
            IdsUpdate u;
            std::memcpy(&u, payload, sizeof(u));
            seen.push_back(u);
            return true;
        }
        // ignore other types
        return true; });

    if (!ok)
    {
        std::cerr << "wal.replay failed\n";
        return 5;
    }

    if (seen.size() != ins.size())
    {
        std::cerr << "replay count mismatch (seen=" << seen.size() << " expected=" << ins.size() << ")\n";
        return 6;
    }

    for (size_t i = 0; i < ins.size(); ++i)
    {
        if (seen[i].idx != ins[i].idx || seen[i].value != ins[i].value)
        {
            std::cerr << "record mismatch at " << i << "\n";
            return 7;
        }
    }

    // truncate and check file size
    if (!wal.truncate_to_zero())
    {
        std::cerr << "truncate_to_zero failed\n";
        return 8;
    }

    struct stat st;
    if (stat(walpath.c_str(), &st) != 0)
    {
        std::cerr << "stat wal failed\n";
        return 9;
    }
    // file should contain only header (WAL_FILE_HEADER_SIZE)
    if (st.st_size < static_cast<off_t>(WalManager::WAL_FILE_HEADER_SIZE))
    {
        std::cerr << "wal size unexpected after truncate: " << st.st_size << "\n";
        return 10;
    }

    std::cout << "wal_manager_test PASSED\n";
    return 0;
}