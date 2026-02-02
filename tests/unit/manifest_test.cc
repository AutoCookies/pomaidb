#include "tests/common/test_main.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "pomai/status.h"
#include "storage/manifest/manifest.h"
#include "tests/common/test_tmpdir.h"
#include "util/crc32c.h"

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
        // Expect v3 header
        POMAI_EXPECT_TRUE(content.rfind("pomai.manifest.v3\n", 0) == 0);

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
        a.metric = pomai::MetricType::kInnerProduct;

        pomai::MembraneSpec b;
        b.name = "beta";
        b.shard_count = 4;
        b.dim = 16;
        b.metric = pomai::MetricType::kCosine;
        b.index_params.num_lists = 99;

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
        POMAI_EXPECT_TRUE(got.metric == pomai::MetricType::kCosine);
        POMAI_EXPECT_EQ(got.index_params.num_lists, static_cast<std::uint32_t>(99));

        // Create again => AlreadyExists.
        auto st = pomai::storage::Manifest::CreateMembrane(root, a);
        POMAI_EXPECT_TRUE(!st.ok());
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kAlreadyExists);

        namespace fs = std::filesystem;
        const auto b_manifest = (fs::path(root) / "membranes" / "beta" / "MANIFEST").string();
        POMAI_EXPECT_TRUE(fs::exists(b_manifest));
        const std::string mcontent = ReadAllOrDie(b_manifest);
        // Expect v2 header
        POMAI_EXPECT_TRUE(mcontent.rfind("pomai.membrane.v2\n", 0) == 0);

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
        POMAI_EXPECT_OK(pomai::storage::Manifest::EnsureInitialized(root));
        
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

        // Manually write bad header without CRC logic (or even with)
        {
            std::ofstream out(fs::path(root) / "MANIFEST", std::ios::binary | std::ios::trunc);
            std::string content = "not-a-real-header\n";
            // If we just write content, it will fail CRC check if > 4 bytes, or be too short.
            // Let's write dummy CRC to pass size check if needed, but here we want header check failure.
            // Actually, if we use AtomicWriteFile it appends CRC.
            // But we are simulating existing corruption.
            
            // Write content + valid CRC but bad header
            out.write(content.data(), content.size());
            uint32_t crc = pomai::util::Crc32c(content.data(), content.size());
            out.write(reinterpret_cast<const char*>(&crc), 4);
        }

        std::vector<std::string> names;
        auto st = pomai::storage::Manifest::ListMembranes(root, &names);
        POMAI_EXPECT_TRUE(!st.ok());
        // Could be Corruption due to header
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kAborted);
    }
    
    POMAI_TEST(Manifest_Corruption_BadCRC)
    {
        namespace fs = std::filesystem;
        const std::string root = pomai::test::TempDir("pomai-manifest-badcrc");

        // Initialize correctly first
        POMAI_EXPECT_OK(pomai::storage::Manifest::EnsureInitialized(root));
        
        // Corrupt the file content
        {
             std::string path = (fs::path(root) / "MANIFEST").string();
             std::fstream out(path, std::ios::in | std::ios::out | std::ios::binary);
             out.seekp(0);
             out.put('X'); // Corrupt first byte
        }
        
        std::vector<std::string> names;
        auto st = pomai::storage::Manifest::ListMembranes(root, &names);
        POMAI_EXPECT_TRUE(!st.ok());
        // Should be CRC mismatch
        POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kAborted);
    }

} // namespace
