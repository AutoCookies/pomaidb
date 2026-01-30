#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstring>
#include <fstream>

#include <pomai/storage/wal.h>
#include <pomai/core/seed.h>

using namespace pomai;
namespace fs = std::filesystem;

// Helper: create a temporary directory and return its path.
static std::string MakeTempDir()
{
    std::string tmpl = "/tmp/pomai_test.XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char *res = mkdtemp(buf.data());
    if (!res)
        throw std::runtime_error("mkdtemp failed");
    return std::string(res);
}

static void RemoveDir(const std::string &d)
{
    std::error_code ec;
    fs::remove_all(d, ec);
}

// Read full file into vector
static std::vector<uint8_t> ReadFile(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> v((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return v;
}

// Find offset of second record start by parsing first in-file. Returns -1 on error.
static off_t FindSecondRecordOffset(const std::string &wal_path)
{
    int fd = ::open(wal_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        return -1;
    // read header
    uint8_t head[8 + 4 + 2];
    ssize_t r = ::read(fd, head, sizeof(head));
    if (r != (ssize_t)sizeof(head))
    {
        ::close(fd);
        return -1;
    }
    const uint8_t *p = head;
    uint64_t lsn_le;
    uint32_t count_le;
    uint16_t dim_le;
    std::memcpy(&lsn_le, p, 8);
    p += 8;
    std::memcpy(&count_le, p, 4);
    p += 4;
    std::memcpy(&dim_le, p, 2);
#if __BYTE_ORDER == __BIG_ENDIAN
    count_le = __builtin_bswap32(count_le);
    dim_le = __builtin_bswap16(dim_le);
    lsn_le = __builtin_bswap64(lsn_le);
#endif
    uint32_t count = count_le;
    uint16_t dim = dim_le;
    // compute bytes for record payload: count * (id + dim*f32)
    off_t rest = (off_t)count * ((off_t)sizeof(uint64_t) + (off_t)dim * (off_t)sizeof(float));
    // footer size 24
    off_t second_offset = sizeof(head) + rest + 24;
    ::close(fd);
    return second_offset;
}

static void TruncateFile(const std::string &path, off_t newsize)
{
    int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0)
        throw std::runtime_error("open for truncate failed");
    if (ftruncate(fd, newsize) != 0)
    {
        ::close(fd);
        throw std::runtime_error("ftruncate failed");
    }
    // make trunc durable
    if (fdatasync(fd) != 0)
    {
        ::close(fd);
        throw std::runtime_error("fdatasync failed");
    }
    // fsync dir
    ::close(fd);
    // directory fsync not strictly required for these tests but production code does it
}

static void CorruptFooterCrc(const std::string &path)
{
    // Naive approach: flip a byte near EOF (in the crc field of last footer)
    std::vector<uint8_t> data = ReadFile(path);
    if (data.size() < 24)
        throw std::runtime_error("file too small to corrupt");
    // footer ends at EOF; crc is 8 bytes before last 8 bytes (magic)
    size_t footer_pos = data.size() - 24;
    size_t crc_pos = footer_pos + 8; // after payload_size+reserved (4+4)
    data[crc_pos] ^= 0x7F;           // flip a bit
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(data.data()), data.size());
    out.flush();
    out.close();
}

int main()
{
    std::cout << "WAL unit tests starting...\n";
    int failures = 0;

    // Test 1: Good records replay
    try
    {
        std::string dir = MakeTempDir();
        Wal w("shard-0", dir, 4);
        w.Start();

        // build a small batch of 10 vectors
        std::vector<UpsertRequest> batch;
        for (int i = 0; i < 10; ++i)
        {
            UpsertRequest r;
            r.id = i + 1;
            r.vec.data.resize(4);
            for (int j = 0; j < 4; ++j)
                r.vec.data[j] = float(i + j);
            batch.push_back(std::move(r));
        }

        Lsn l = w.AppendUpserts(batch, true); // sync durable
        (void)l;
        w.Stop();

        Seed seed(4);
        WalReplayStats s = w.ReplayToSeed(seed);
        if (s.records_applied != 1 || s.vectors_applied != 10)
        {
            std::cerr << "Test1 FAILED: expected 1 record,10 vectors; got " << s.records_applied << "," << s.vectors_applied << "\n";
            ++failures;
        }
        else
        {
            std::cout << "Test1 PASS\n";
        }
        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test1 exception: " << e.what() << "\n";
        ++failures;
    }

    // Test 2: Truncated footer at EOF - write two records then truncate second's footer
    try
    {
        std::string dir = MakeTempDir();
        Wal w("shard-0", dir, 4);
        w.Start();

        // two batches
        std::vector<UpsertRequest> b1, b2;
        for (int i = 0; i < 5; ++i)
        {
            UpsertRequest r;
            r.id = i;
            r.vec.data = {1, 2, 3, 4};
            b1.push_back(r);
        }
        for (int i = 0; i < 7; ++i)
        {
            UpsertRequest r;
            r.id = i + 100;
            r.vec.data = {5, 6, 7, 8};
            b2.push_back(r);
        }
        w.AppendUpserts(b1, true);
        w.AppendUpserts(b2, true);
        w.Stop();

        // Now truncate the last 10 bytes to simulate missing footer
        std::string wal_path = dir + "/shard-0.wal";
        off_t orig = (off_t)fs::file_size(wal_path);
        if (orig < 10)
            throw std::runtime_error("wal too small");
        TruncateFile(wal_path, orig - 10);

        Seed seed(4);
        WalReplayStats s = w.ReplayToSeed(seed);
        if (s.records_applied != 1 || s.vectors_applied != 5 || s.truncated_bytes == 0)
        {
            std::cerr << "Test2 FAILED: got records=" << s.records_applied << " vectors=" << s.vectors_applied << " truncated=" << s.truncated_bytes << "\n";
            ++failures;
        }
        else
            std::cout << "Test2 PASS\n";

        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test2 exception: " << e.what() << "\n";
        ++failures;
    }

    // Test 3: Corrupt CRC of second record -> first record applied, file truncated
    try
    {
        std::string dir = MakeTempDir();
        Wal w("shard-0", dir, 4);
        w.Start();

        std::vector<UpsertRequest> b1, b2;
        for (int i = 0; i < 3; ++i)
        {
            UpsertRequest r;
            r.id = i;
            r.vec.data = {1, 2, 3, 4};
            b1.push_back(r);
        }
        for (int i = 0; i < 4; ++i)
        {
            UpsertRequest r;
            r.id = i + 10;
            r.vec.data = {5, 6, 7, 8};
            b2.push_back(r);
        }
        w.AppendUpserts(b1, true);
        w.AppendUpserts(b2, true);
        w.Stop();

        std::string wal_path = dir + "/shard-0.wal";
        // Corrupt CRC of second record
        CorruptFooterCrc(wal_path);

        Seed seed(4);
        WalReplayStats s = w.ReplayToSeed(seed);
        if (s.records_applied != 1 || s.vectors_applied != 3 || s.truncated_bytes == 0)
        {
            std::cerr << "Test3 FAILED: got records=" << s.records_applied << " vectors=" << s.vectors_applied << " truncated=" << s.truncated_bytes << "\n";
            ++failures;
        }
        else
            std::cout << "Test3 PASS\n";

        RemoveDir(dir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Test3 exception: " << e.what() << "\n";
        ++failures;
    }

    if (failures == 0)
    {
        std::cout << "All unit tests PASSED.\n";
        return 0;
    }
    else
    {
        std::cerr << failures << " unit tests FAILED.\n";
        return 2;
    }
}