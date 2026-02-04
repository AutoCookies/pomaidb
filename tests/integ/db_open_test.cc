#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include <filesystem>
#include <fstream>

namespace pomai
{
    namespace fs = std::filesystem;

    POMAI_TEST(OpenFailsInternalErrorIfPathInvalid)
    {
        std::string test_dir = pomai::test::TempDir("pomai_test_open_fail");
        DBOptions opt;
        // Linux won't allow null char in path obviously, but empty path or weird paths?
        // Let's try opening a file as directory.
        std::string file_path = test_dir + "/conflict";
        {
            std::ofstream f(file_path);
            f << "garbage";
        }

        opt.path = file_path; // Points to a file, but DB expects directory
        // This should fail, maybe IOError or Internal.
        // Previously would succeed silently.

        std::unique_ptr<DB> db;
        Status st = DB::Open(opt, &db);
        POMAI_EXPECT_TRUE(!st.ok());
        // fs::remove_all(test_dir); // TempDir cleans up? No, need manual or RAII. 
        // test_tmpdir.h usually produces unique paths. Cleanup is optional or handled by OS / test runner if extended.
        // For now let's just leave it or manual cleanup.
        fs::remove_all(test_dir);
    }

    POMAI_TEST(OpenMissingPathCreatesIt)
    {
        std::string test_dir = pomai::test::TempDir("pomai_test_open_ok");
        DBOptions opt;
        opt.path = test_dir + "/new_db";

        std::unique_ptr<DB> db;
        Status st = DB::Open(opt, &db);
        POMAI_EXPECT_TRUE(st.ok());
        
        POMAI_EXPECT_TRUE(fs::exists(opt.path));
        POMAI_EXPECT_TRUE(fs::is_directory(opt.path));
        fs::remove_all(test_dir);
    }

    POMAI_TEST(OpenFailsInvalidConfig)
    {
        std::string test_dir = pomai::test::TempDir("pomai_test_config");
        
        // Empty path
        {
            DBOptions opt;
            opt.path = "";
            opt.dim = 128;
            opt.shard_count = 1;
            std::unique_ptr<DB> db;
            Status st = DB::Open(opt, &db);
            POMAI_EXPECT_TRUE(!st.ok());
             // Not testing exact string message to avoid fragility, but we expect error.
        }

        // Zero Dim
        {
            DBOptions opt;
            opt.path = test_dir;
            opt.dim = 0;
            opt.shard_count = 1;
            std::unique_ptr<DB> db;
            Status st = DB::Open(opt, &db);
            POMAI_EXPECT_TRUE(!st.ok());
        }

        // Zero Shards
        {
            DBOptions opt;
            opt.path = test_dir;
            opt.dim = 128;
            opt.shard_count = 0;
            std::unique_ptr<DB> db;
            Status st = DB::Open(opt, &db);
            POMAI_EXPECT_TRUE(!st.ok());
        }
        
        fs::remove_all(test_dir);
    }
}
