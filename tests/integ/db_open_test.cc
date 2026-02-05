#include "tests/common/test_main.h"
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <ctime>

// Include necessary headers for DB
#include "pomai/pomai.h"

namespace pomai::tests {

namespace fs = std::filesystem;

class TestDir {
public:
    TestDir() {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        path_ = "test_db_open_" + std::to_string(std::rand());
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TestDir() {
        fs::remove_all(path_);
    }
    std::string path() const { return path_; }
private:
    std::string path_;
};

POMAI_TEST(Database_OpenAndClose) {
    TestDir td;
    DBOptions opt;
    opt.path = td.path();
    opt.dim = 4;
    opt.shard_count = 1;

    std::unique_ptr<DB> db;
    Status st = DB::Open(opt, &db);
    POMAI_EXPECT_TRUE(st.ok());
    POMAI_EXPECT_TRUE(db != nullptr);
    POMAI_EXPECT_TRUE(fs::exists(td.path()));

    st = db->Close();
    POMAI_EXPECT_TRUE(st.ok());
}

POMAI_TEST(Database_FailIfPathInvalid) {
    DBOptions opt;
    opt.path = "";
    opt.dim = 4;
    std::unique_ptr<DB> db;
    Status st = DB::Open(opt, &db);
    POMAI_EXPECT_TRUE(!st.ok());
}

POMAI_TEST(Database_CleanupOnFirstOpenFailure) {
    TestDir td;
    // Create a file conflict to force directory creation failure?
    // Engine::Open calls create_directories(opt.path).
    // If opt.path exists as a FILE, create_directories should fail.
    
    std::string bad_path = td.path() + "/collision";
    {
        std::ofstream f(bad_path);
        f << "I am a file";
    }

    DBOptions opt;
    opt.path = bad_path; // Use the file path as the dir path
    opt.dim = 4;
    opt.shard_count = 1;

    std::unique_ptr<DB> db;
    Status st = DB::Open(opt, &db);
    POMAI_EXPECT_TRUE(!st.ok());
}

POMAI_TEST(Database_CorruptWalReplayFailsOpen) {
    TestDir td;
    std::string db_path = td.path() + "/db";

    // 1. Create valid DB
    {
        DBOptions opt;
        opt.path = db_path;
        opt.dim = 4;
        opt.shard_count = 1;
        std::unique_ptr<DB> db;
        auto st = DB::Open(opt, &db);
        POMAI_EXPECT_TRUE(st.ok());
        // Write something so WAL exists
        std::vector<float> vec = {1, 2, 3, 4};
        db->Put(1, vec);
        db->Close();
    }

    // 2. Corrupt the WAL file
    bool found_wal = false;
    if (fs::exists(db_path)) {
        for (const auto& entry : fs::directory_iterator(db_path)) {
            if (entry.path().string().find("wal_") != std::string::npos) {
                // Truncate/Corrupt it
                std::ofstream f(entry.path(), std::ios::binary | std::ios::trunc);
                f << "GARBAGE";
                f.close();
                found_wal = true;
                break;
            }
        }
    }
    POMAI_EXPECT_TRUE(found_wal);

    // 3. Try to Open again
    {
        DBOptions opt;
        opt.path = db_path;
        opt.dim = 4;
        opt.shard_count = 1;
        std::unique_ptr<DB> db;
        Status st = DB::Open(opt, &db);
        // SOT says: Fail closed by default on corruption
        POMAI_EXPECT_TRUE(!st.ok());
    }
}

} // namespace pomai::tests
