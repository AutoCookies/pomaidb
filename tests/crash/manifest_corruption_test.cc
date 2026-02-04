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

        // 3. Try Reopen
        {
            std::unique_ptr<pomai::DB> db;
            auto st = pomai::DB::Open(opt, &db);
            // Current expected behavior: Open succeeds but maybe logs error, 
            // OR checks existence of segments and fails?
            // ShardManifest::Load just reads lines.
            // SegmentReader::Open will be called on "garbage_segment_name.dat".
            // It will fail.
            // ShardRuntime::LoadSegments propagates error.
            // So DB::Open -> Manager::Open -> Shard::Start -> LoadSegments -> Error.
            
            // We EXPECT failure (or at least partial failure).
            // Currently, ShardRuntime::Start returns Status.
            // But DB::Open (manager) spawns background shards... wait.
            // DbImpl ctor calls mgr_.Open().
            // MembraneManager::Open calls Shard::Start?
            // Let's check.
            
            // If DB::Open returns OK but shard is broken, that's bad.
            // If DB::Open returns Error, that's good (fail fast).
            
            // We expect Open to FAIL if shard 0 fails.
            // Or Open Membrane to fail.
            
            // Wait, DB::Open just creates DbImpl. DbImpl ctor calls mgr_.Open().
            // mgr_.Open() opens default membrane if it exists.
            // If implicit open fails...
        }
    }
}
