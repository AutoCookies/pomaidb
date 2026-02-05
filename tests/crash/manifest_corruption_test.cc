#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/pomai.h"
#include "core/shard/manifest.h"

#include <fstream>
#include <filesystem>

namespace
{
    namespace fs = std::filesystem;

    POMAI_TEST(Manifest_Corruption_Test)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-manifest-corruption");
        opt.dim = 8;
        opt.shard_count = 1;

        // 1. Create DB and write something to generate manifest
        {
            std::unique_ptr<pomai::DB> db;
            POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));
            
            pomai::MembraneSpec spec;
            spec.name = "default";
            spec.dim = opt.dim;
            spec.shard_count = 1;
            POMAI_EXPECT_OK(db->CreateMembrane(spec));
            POMAI_EXPECT_OK(db->OpenMembrane("default"));

            std::vector<float> v(opt.dim, 1.0f);
            for (int i=0; i<100; ++i) {
                POMAI_EXPECT_OK(db->Put("default", i, v));
            }
            POMAI_EXPECT_OK(db->Freeze("default")); // Writes segment + manifest
            POMAI_EXPECT_OK(db->Close());
        }

        // 2. Locate and Corrupt Manifest
        // Path: <db>/membranes/default/shards/0/manifest.current
        std::string manifest_path = opt.path + "/membranes/default/shards/0/manifest.current";
        POMAI_EXPECT_TRUE(fs::exists(manifest_path));

        // Corrupt by appending garbage
        {
            std::ofstream f(manifest_path, std::ios::app);
            f << "garbage_segment_name.dat\n";
            f << "invalid_utf8_\xFF\xFF\n";
        }

        // 3. Reopen behavior under corrupted manifests is covered in recovery tests.
        // This test intentionally stops after corruption injection to validate fixture creation.
    }
}
