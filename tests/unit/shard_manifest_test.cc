#include "tests/common/test_main.h"
#include "core/shard/manifest.h"
#include "tests/common/test_tmpdir.h"
#include <filesystem>
#include <vector>
#include <string>
#include <fstream>

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

        // backup file retained for fallback
        POMAI_EXPECT_TRUE(fs::exists(fs::path(root) / "manifest.prev"));
    }

    POMAI_TEST(ShardManifest_LoadFallbackToPrevOnCorruption) {
        const std::string root = pomai::test::TempDir("shard-manifest-fallback-test");
        std::vector<std::string> first = {"seg_a.dat"};
        std::vector<std::string> second = {"seg_b.dat", "seg_a.dat"};

        POMAI_EXPECT_OK(pomai::core::ShardManifest::Commit(root, first));
        POMAI_EXPECT_OK(pomai::core::ShardManifest::Commit(root, second));

        // Corrupt current manifest payload while keeping header to trigger CRC failure.
        const fs::path curr = fs::path(root) / "manifest.current";
        {
            std::ofstream out(curr, std::ios::trunc);
            out << "pomai.shard_manifest.v2\n";
            out << "crc32c 1\n";
            out << "seg_corrupt.dat\n";
        }

        std::vector<std::string> loaded;
        POMAI_EXPECT_OK(pomai::core::ShardManifest::Load(root, &loaded));
        POMAI_EXPECT_EQ(loaded.size(), 1);
        POMAI_EXPECT_EQ(loaded[0], "seg_a.dat"); // from manifest.prev
    }

    POMAI_TEST(ShardManifest_LoadLegacyFormatStillSupported) {
        const std::string root = pomai::test::TempDir("shard-manifest-legacy-test");
        const fs::path curr = fs::path(root) / "manifest.current";

        {
            std::ofstream out(curr, std::ios::trunc);
            out << "seg_legacy_1.dat\n";
            out << "seg_legacy_0.dat\n";
        }

        std::vector<std::string> loaded;
        POMAI_EXPECT_OK(pomai::core::ShardManifest::Load(root, &loaded));
        POMAI_EXPECT_EQ(loaded.size(), 2);
        POMAI_EXPECT_EQ(loaded[0], "seg_legacy_1.dat");
        POMAI_EXPECT_EQ(loaded[1], "seg_legacy_0.dat");
    }

} // namespace
