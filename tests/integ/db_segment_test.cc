#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <filesystem>
#include <vector>
#include <fstream>

#include "pomai/pomai.h"
#include "pomai/options.h"
#include "pomai/search.h"
#include "table/segment.h"
#include "storage/manifest/manifest.h"

namespace {

namespace fs = std::filesystem;

POMAI_TEST(DB_SegmentLoading_ReadTest) {
    const std::string root = pomai::test::TempDir("pomai-db-segment-test");
    const std::string membrane = "default";
    const uint32_t dim = 4;
    
    // 1. Initialize DB layout manually to inject segment
    pomai::MembraneSpec spec;
    spec.name = membrane;
    spec.dim = dim;
    spec.shard_count = 1;
    spec.metric = pomai::MetricType::kL2;
    
    POMAI_EXPECT_OK(pomai::storage::Manifest::CreateMembrane(root, spec));
    
    // 2. Create Shard Directory
    fs::path shard_dir = fs::path(root) / "membranes" / membrane / "shards" / "0";
    fs::create_directories(shard_dir);
    
    // 3. Create a segment file
    std::string seg_path = (shard_dir / "seg_00001.dat").string();
    pomai::table::SegmentBuilder builder(seg_path, dim);
    
    std::vector<float> vec1 = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec2 = {0.0f, 1.0f, 0.0f, 0.0f};
    
    POMAI_EXPECT_OK(builder.Add(10, vec1));
    POMAI_EXPECT_OK(builder.Add(20, vec2));
    POMAI_EXPECT_OK(builder.Finish());
    
    // 4. Open DB
    pomai::DBOptions opt;
    opt.path = root;
    
    std::unique_ptr<pomai::DB> db;
    POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));
    
    // Register "default" membrane in Manager (since Manager doesn't scan Manifest yet)
    POMAI_EXPECT_OK(db->CreateMembrane(spec));
    
    // Must open membrane to load shards
    POMAI_EXPECT_OK(db->OpenMembrane(membrane));
    
    // 5. Verify Get
    std::vector<float> out;
    POMAI_EXPECT_OK(db->Get(membrane, 10, &out));
    POMAI_EXPECT_EQ(out.size(), (size_t)dim);
    POMAI_EXPECT_EQ(out[0], 1.0f);
    
    out.clear();
    POMAI_EXPECT_OK(db->Get(membrane, 20, &out));
    POMAI_EXPECT_EQ(out[0], 0.0f);
    POMAI_EXPECT_EQ(out[1], 1.0f);
    
    // Non-existent
    pomai::Status st = db->Get(membrane, 99, &out);
    POMAI_EXPECT_EQ(st.code(), pomai::ErrorCode::kNotFound);
    
    // 6. Verify Search (Brute force should pick up segments)
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f}; // Exact match for 10
    
    pomai::SearchResult res;
    POMAI_EXPECT_OK(db->Search(membrane, query, 5, &res));
    POMAI_EXPECT_TRUE(res.hits.size() >= 1);
    POMAI_EXPECT_EQ(res.hits[0].id, (pomai::VectorId)10);
    // Score should be 1.0 (Dot Product 1*1).
    POMAI_EXPECT_TRUE(std::abs(res.hits[0].score - 1.0f) < 0.001f);
    
    // 7. Verify Exists
    bool exists = false;
    POMAI_EXPECT_OK(db->Exists(membrane, 20, &exists));
    POMAI_EXPECT_TRUE(exists);
    POMAI_EXPECT_OK(db->Exists(membrane, 99, &exists));
    POMAI_EXPECT_TRUE(!exists);
}

} // namespace
