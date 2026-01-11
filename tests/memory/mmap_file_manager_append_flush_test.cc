/*
 tests/memory/mmap_file_manager_append_flush_test.cc

 End-to-end tests for MmapFileManager append() + flush(...) (msync) paths.

 The tests exercise:
  - append of an arbitrary byte buffer, flush(offset,len, sync=true) and verify
    the underlying file contains the appended bytes after closing the mapping.
  - append of an 8-byte value (the special-case path that uses atomic_store_u64)
    followed by flush(..., sync=true) and verification via direct file read.

 Files are created under /dev/shm when available (fast tmpfs) falling back to /tmp.
 These tests are standalone executables (no test framework) and return 0 on success,
 non-zero on failure.
*/

#include "src/memory/mmap_file_manager.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <vector>
#include <string>

static std::string choose_tmpdir()
{
    const char *d = std::getenv("TMPFS_DIR");
    if (d && d[0]) return std::string(d);
    // Use /dev/shm when available for faster tests; fall back to /tmp.
    struct stat st;
    if (stat("/dev/shm", &st) == 0 && S_ISDIR(st.st_mode))
        return "/dev/shm";
    return "/tmp";
}

static bool read_file_at(const std::string &path, size_t offset, void *dst, size_t len)
{
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
        std::cerr << "open for verify failed: " << path << " errno=" << strerror(errno) << "\n";
        return false;
    }
    ssize_t r = pread(fd, dst, static_cast<ssize_t>(len), static_cast<off_t>(offset));
    if (r != static_cast<ssize_t>(len))
    {
        std::cerr << "pread failed: wanted=" << len << " got=" << r << " errno=" << strerror(errno) << "\n";
        ::close(fd);
        return false;
    }
    ::close(fd);
    return true;
}

int main()
{
    using pomai::memory::MmapFileManager;

    std::string tmpdir = choose_tmpdir();
    std::string path1 = tmpdir + "/test_mmap_append.bin";
    std::string path2 = tmpdir + "/test_mmap_append_u64.bin";

    // Cleanup from previous runs
    unlink(path1.c_str());
    unlink(path2.c_str());

    const size_t map_size = 4096 * 4;

    // ----- Test 1: append arbitrary bytes and flush(sync=true) -----
    {
        MmapFileManager mgr;
        if (!mgr.open(path1, map_size, true))
        {
            std::cerr << "MmapFileManager::open failed for " << path1 << "\n";
            return 1;
        }

        const char sample[] = "hello_mmap_manager_append_flush";
        size_t sample_len = sizeof(sample) - 1;

        size_t off = mgr.append(sample, sample_len);
        if (off == SIZE_MAX)
        {
            std::cerr << "append returned SIZE_MAX\n";
            return 2;
        }

        // flush the appended range and require durability (msync)
        if (!mgr.flush(off, sample_len, /*sync=*/true))
        {
            std::cerr << "flush(sync=true) failed\n";
            return 3;
        }

        // close mapping so we can verify file contents via direct read
        mgr.close();

        std::vector<char> buf(sample_len);
        if (!read_file_at(path1, off, buf.data(), sample_len))
            return 4;

        if (std::memcmp(buf.data(), sample, sample_len) != 0)
        {
            std::cerr << "Data mismatch after append+flush for " << path1 << "\n";
            return 5;
        }

        std::cout << "Test1 append+flush bytes: OK (offset=" << off << " len=" << sample_len << ")\n";
    }

    // ----- Test 2: append 8-byte value (atomic path) and flush(sync=true) -----
    {
        MmapFileManager mgr;
        if (!mgr.open(path2, map_size, true))
        {
            std::cerr << "MmapFileManager::open failed for " << path2 << "\n";
            return 6;
        }

        uint64_t val = 0xDEADBEEFCAFEBABEULL;
        size_t off = mgr.append(&val, sizeof(val));
        if (off == SIZE_MAX)
        {
            std::cerr << "append(u64) returned SIZE_MAX\n";
            return 7;
        }

        // flush the appended 8-byte region to durability
        if (!mgr.flush(off, sizeof(val), /*sync=*/true))
        {
            std::cerr << "flush(sync=true) failed for u64 append\n";
            return 8;
        }

        mgr.close();

        uint64_t loaded = 0;
        if (!read_file_at(path2, off, &loaded, sizeof(loaded)))
            return 9;

        if (loaded != val)
        {
            std::cerr << "u64 mismatch after append+flush: expected=0x" << std::hex << val
                      << " got=0x" << loaded << std::dec << "\n";
            return 10;
        }

        std::cout << "Test2 append+flush u64: OK (offset=" << off << ")\n";
    }

    // cleanup
    unlink(path1.c_str());
    unlink(path2.c_str());

    std::cout << "mmap_file_manager_append_flush_test PASSED\n";
    return 0;
}