#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"
#include "util/posix_file.h"
#include <filesystem>
#include <vector>
#include <fstream>

namespace pomai
{
    namespace fs = std::filesystem;
    using storage::Wal;

    // Helper to extract data from MemTable
    static std::vector<float> GetVec(const table::MemTable& mem, VectorId id, uint32_t dim) {
        const float* ptr = nullptr;
        if (!mem.Get(id, &ptr).ok() || !ptr) return {};
        return std::vector<float>(ptr, ptr + dim);
    }

    POMAI_TEST(Wal_RoundTrip)
    {
        std::string dir = pomai::test::TempDir("wal_roundtrip");
        {
            auto wal = std::make_unique<Wal>(dir, 0, 1024*1024, FsyncPolicy::kNever);
            POMAI_EXPECT_OK(wal->Open());
            
            std::vector<float> v1 = {1.0f, 2.0f};
            POMAI_EXPECT_OK(wal->AppendPut(1, v1));
            
            std::vector<float> v2 = {3.0f, 4.0f};
            POMAI_EXPECT_OK(wal->AppendPut(2, v2));
            
            POMAI_EXPECT_OK(wal->AppendDelete(1));
        }

        {
            // Replay
            table::MemTable mem(2, 4096);
            auto wal = std::make_unique<Wal>(dir, 0, 1024*1024, FsyncPolicy::kNever);

            POMAI_EXPECT_OK(wal->ReplayInto(mem));
            
            // Verify
            auto out2 = GetVec(mem, 2, 2);
            POMAI_EXPECT_EQ(out2.size(), static_cast<size_t>(2));
            POMAI_EXPECT_EQ(out2[0], 3.0f);

            auto out1 = GetVec(mem, 1, 2);
            POMAI_EXPECT_TRUE(out1.empty()); // deleted (GetVec returns empty for null/missing)
        }
        fs::remove_all(dir);
    }

    POMAI_TEST(Wal_TruncatedRecord_Ignored)
    {
        std::string dir = pomai::test::TempDir("wal_trunc");
        
        // 1. Write valid
        {
            auto wal = std::make_unique<Wal>(dir, 0, 1024*1024, FsyncPolicy::kNever);
            POMAI_EXPECT_OK(wal->Open());
            std::vector<float> v1 = {1.0f, 1.0f};
            POMAI_EXPECT_OK(wal->AppendPut(10, v1));
        }

        // 2. Append Partial Garbage to end of file manually
        std::string log_path = dir + "/wal_0_0.log"; 
        POMAI_EXPECT_TRUE(fs::exists(log_path));
        
        {
            std::ofstream f(log_path, std::ios::app | std::ios::binary);
            uint32_t len = 100;
            f.write(reinterpret_cast<char*>(&len), sizeof(len));
            f.write("1234567890", 10);
        }

        // 3. Replay
        {
            table::MemTable mem(2, 4096);
            auto wal = std::make_unique<Wal>(dir, 0, 1024*1024, FsyncPolicy::kNever);
            
            POMAI_EXPECT_OK(wal->ReplayInto(mem));

            // Should have 10
            auto out = GetVec(mem, 10, 2);
            POMAI_EXPECT_EQ(out.size(), static_cast<size_t>(2));
        }
        fs::remove_all(dir);
    }

     POMAI_TEST(Wal_CorruptedBody_Fails)
    {
        std::string dir = pomai::test::TempDir("wal_corrupt");
        
        // 1. Write valid
        {
            auto wal = std::make_unique<Wal>(dir, 0, 1024*1024, FsyncPolicy::kNever);
            POMAI_EXPECT_OK(wal->Open());
            std::vector<float> v1 = {1.0f, 1.0f};
            POMAI_EXPECT_OK(wal->AppendPut(10, v1));
        }

        // 2. Corrupt the file in the middle (byte 20 or so)
        std::string log_path = dir + "/wal_0_0.log"; 
        {
            std::fstream f(log_path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(20); 
            f.write("\xFF", 1);
        }

        // 3. Replay
        {
            table::MemTable mem(2, 4096);
            auto wal = std::make_unique<Wal>(dir, 0, 1024*1024, FsyncPolicy::kNever);
            
            Status st = wal->ReplayInto(mem);
            POMAI_EXPECT_TRUE(!st.ok());
        }
        fs::remove_all(dir);
    }
}
