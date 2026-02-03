#include "tests/common/test_main.h"
#include "core/shard/manifest.h"
#include "tests/common/test_tmpdir.h"
#include <filesystem>
#include <vector>
#include <string>

namespace {
    namespace fs = std::filesystem;

    POMAI_TEST(ShardManifest_CommitLoad) {
        const std::string root = pomai::test::TempDir("shard-manifest-test");
        
        // Initial load -> empty
        std::vector<std::string> segs;
        POMAI_EXPECT_OK(pomai::core::ShardManifest::Load(root, &segs));
        POMAI_EXPECT_TRUE(segs.empty());
        
        // Commit
        std::vector<std::string> expected = {"seg_1.dat", "seg_0.dat"};
        POMAI_EXPECT_OK(pomai::core::ShardManifest::Commit(root, expected));
        
        // Check file exists
        POMAI_EXPECT_TRUE(fs::exists(fs::path(root) / "manifest.current"));
        POMAI_EXPECT_TRUE(!fs::exists(fs::path(root) / "manifest.new")); // Should be renamed
        
        // Load back
        std::vector<std::string> loaded;
        POMAI_EXPECT_OK(pomai::core::ShardManifest::Load(root, &loaded));
        POMAI_EXPECT_EQ(loaded.size(), 2);
        POMAI_EXPECT_EQ(loaded[0], "seg_1.dat");
        POMAI_EXPECT_EQ(loaded[1], "seg_0.dat");
        
        // Update again
        expected.push_back("seg_2.dat");
        POMAI_EXPECT_OK(pomai::core::ShardManifest::Commit(root, expected));
        
        POMAI_EXPECT_OK(pomai::core::ShardManifest::Load(root, &loaded));
        POMAI_EXPECT_EQ(loaded.size(), 3);
        POMAI_EXPECT_EQ(loaded[2], "seg_2.dat");
    }

} // namespace
