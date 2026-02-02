#include "tests/common/test_main.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "pomai/status.h"
#include "storage/manifest/manifest.h"
#include "tests/common/test_tmpdir.h"

namespace
{

    static std::string ReadAllOrDie(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        POMAI_EXPECT_TRUE(in.is_open());

        in.seekg(0, std::ios::end);
        const std::streamoff n = in.tellg();
        in.seekg(0, std::ios::beg);

        std::string buf;
        if (n > 0)
        {
            buf.resize(static_cast<std::size_t>(n));
            in.read(buf.data(), n);
        }
        return buf;
    }

    POMAI_TEST(Manifest_EnsureInitialized_CreatesLayout)
    {
        const std::string root = pomai::test::TempDir("pomai-manifest-init");

        POMAI_EXPECT_OK(pomai::storage::Manifest::EnsureInitialized(root));

        namespace fs = std::filesystem;
        POMAI_EXPECT_TRUE(fs::exists(fs::path(root) / "membranes"));

        const auto root_manifest = (fs::path(root) / "MANIFEST").string();
        POMAI_EXPECT_TRUE(fs::exists(root_manifest));

        const std::string content = ReadAllOrDie(root_manifest);
        POMAI_EXPECT_TRUE(content.rfind("pomai.manifest.v2\n", 0) == 0);

        // Idempotent.
        POMAI_EXPECT_OK(pomai::storage::Manifest::EnsureInitialized(root));
    }

    POMAI_TEST(Manifest_CreateListGetDrop_HappyPath)
    {
        const std::string root = pomai::test::TempDir("pomai-manifest-happy");

        pomai::MembraneSpec a;
        a.name = "alpha";
        a.shard_count = 3;
        a.dim = 8;

        pomai::MembraneSpec b;
        b.name = "beta";
        b.shard_count = 4;
        b.dim = 16;

        POMAI_EXPECT_OK(pomai::storage::Manifest::CreateMembrane(root, a));
        POMAI_EXPECT_OK(pomai::storage::Manifest::CreateMembrane(root, b));

        std::vector<std::string> names;
        POMAI_EXPECT_OK(pomai::storage::Manifest::ListMembranes(root, &names));
        POMAI_EXPECT_EQ(names.size(), static_cast<std::size_t>(2));
        POMAI_EXPECT_EQ(names[0], std::string("alpha"));
        POMAI_EXPECT_EQ(names[1], std::string("beta"));

        pomai::MembraneSpec got;
        POMAI_EXPECT_OK(pomai::storage::Manifest::GetMembrane(root, "beta", &got));
        POMAI_EXPECT_EQ(got.name, std::string("beta"));
        POMAI_EXPECT_EQ(got.shard_count, static_cast<std::uint32_t>(4));
        POMAI_EXPECT_EQ(got.dim, static_cast<std::uint32_t>(16));

        // Create again => AlreadyExists.
        auto st = pomai::storage::Manifest::CreateMembrane(root, a);
        POMAI_EXPECT_TRUE(!st.ok());
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kAlreadyExists);

        namespace fs = std::filesystem;
        const auto b_manifest = (fs::path(root) / "membranes" / "beta" / "MANIFEST").string();
        POMAI_EXPECT_TRUE(fs::exists(b_manifest));
        const std::string mcontent = ReadAllOrDie(b_manifest);
        POMAI_EXPECT_TRUE(mcontent.rfind("pomai.membrane.v1\n", 0) == 0);

        POMAI_EXPECT_OK(pomai::storage::Manifest::DropMembrane(root, "alpha"));

        names.clear();
        POMAI_EXPECT_OK(pomai::storage::Manifest::ListMembranes(root, &names));
        POMAI_EXPECT_EQ(names.size(), static_cast<std::size_t>(1));
        POMAI_EXPECT_EQ(names[0], std::string("beta"));

        POMAI_EXPECT_TRUE(!fs::exists(fs::path(root) / "membranes" / "alpha"));
    }

    POMAI_TEST(Manifest_Validation)
    {
        const std::string root = pomai::test::TempDir("pomai-manifest-validate");

        pomai::MembraneSpec bad;
        bad.name = "../oops";
        bad.dim = 8;
        bad.shard_count = 1;
        auto st = pomai::storage::Manifest::CreateMembrane(root, bad);
        POMAI_EXPECT_TRUE(!st.ok());
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kInvalidArgument);

        pomai::MembraneSpec z;
        z.name = "ok";
        z.dim = 0;
        z.shard_count = 1;
        st = pomai::storage::Manifest::CreateMembrane(root, z);
        POMAI_EXPECT_TRUE(!st.ok());
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kInvalidArgument);

        z.dim = 8;
        z.shard_count = 0;
        st = pomai::storage::Manifest::CreateMembrane(root, z);
        POMAI_EXPECT_TRUE(!st.ok());
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kInvalidArgument);

        pomai::MembraneSpec out;
        st = pomai::storage::Manifest::GetMembrane(root, "missing", &out);
        POMAI_EXPECT_TRUE(!st.ok());
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kNotFound);
    }

    POMAI_TEST(Manifest_Corruption_BadHeader)
    {
        namespace fs = std::filesystem;
        const std::string root = pomai::test::TempDir("pomai-manifest-corrupt");

        fs::create_directories(root);

        {
            std::ofstream out(fs::path(root) / "MANIFEST", std::ios::binary | std::ios::trunc);
            out << "not-a-real-header\n";
        }

        std::vector<std::string> names;
        auto st = pomai::storage::Manifest::ListMembranes(root, &names);
        POMAI_EXPECT_TRUE(!st.ok());
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kAborted);
    }

} // namespace
