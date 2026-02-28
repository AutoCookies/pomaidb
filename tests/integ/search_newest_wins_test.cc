#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <filesystem>
#include <vector>
#include <span>

#include "core/shard/manifest.h"
#include "pomai/options.h"
#include "pomai/pomai.h"
#include "pomai/search.h"
#include "pomai/types.h"
#include "storage/manifest/manifest.h"
#include "table/segment.h"

namespace {

namespace fs = std::filesystem;

POMAI_TEST(SearchNewestWins_DeterministicAndTombstone) {
    const std::string root = pomai::test::TempDir("pomai-search-newest-wins");
    const std::string membrane = "default";
    const uint32_t dim = 4;

    pomai::MembraneSpec spec;
    spec.name = membrane;
    spec.dim = dim;
    spec.shard_count = 1;
    spec.metric = pomai::MetricType::kInnerProduct;  // exact match gives score 1.0; L2 would give 0

    POMAI_EXPECT_OK(pomai::storage::Manifest::CreateMembrane(root, spec));

    fs::path shard_dir = fs::path(root) / "membranes" / membrane / "shards" / "0";
    fs::create_directories(shard_dir);

    const pomai::VectorId target_id = 50000;
    const pomai::VectorId tomb_id = 60000;

    std::vector<float> vec_old = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> vec_new = {0.0f, 1.0f, 0.0f, 0.0f};
    std::vector<float> vec_tomb = {0.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> vec_filler = {0.0f, 0.0f, 0.0f, 0.0f};

    std::string old_path = (shard_dir / "seg_old.dat").string();
    pomai::table::SegmentBuilder old_builder(old_path, dim);
    POMAI_EXPECT_OK(old_builder.Add(target_id, pomai::VectorView(std::span<const float>(vec_old)), false));
    POMAI_EXPECT_OK(old_builder.Add(tomb_id, pomai::VectorView(std::span<const float>(vec_tomb)), false));
    POMAI_EXPECT_OK(old_builder.Finish());

    std::string new_path = (shard_dir / "seg_new.dat").string();
    pomai::table::SegmentBuilder new_builder(new_path, dim);
    for (pomai::VectorId id = 0; id < target_id; ++id) {
        if (id == tomb_id) {
            continue;
        }
        POMAI_EXPECT_OK(new_builder.Add(id, pomai::VectorView(std::span<const float>(vec_filler)), false));
    }
    POMAI_EXPECT_OK(new_builder.Add(target_id, pomai::VectorView(std::span<const float>(vec_new)), false));
    POMAI_EXPECT_OK(new_builder.Add(tomb_id, pomai::VectorView(std::span<const float>(vec_tomb)), true));
    POMAI_EXPECT_OK(new_builder.Finish());

    std::vector<std::string> segs = {"seg_new.dat", "seg_old.dat"};
    POMAI_EXPECT_OK(pomai::core::ShardManifest::Commit(shard_dir.string(), segs));

    pomai::DBOptions opt;
    opt.path = root;
    opt.dim = dim;
    opt.shard_count = 1;

    std::unique_ptr<pomai::DB> db;
    POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

    auto create_st = db->CreateMembrane(spec);
    POMAI_EXPECT_TRUE(create_st.ok() || create_st.code() == pomai::ErrorCode::kAlreadyExists);
    POMAI_EXPECT_OK(db->OpenMembrane(membrane));

    for (int i = 0; i < 50; ++i) {
        pomai::SearchResult res;
        POMAI_EXPECT_OK(db->Search(membrane, vec_new, 5, &res));
        POMAI_EXPECT_TRUE(!res.hits.empty());
        POMAI_EXPECT_EQ(res.hits[0].id, target_id);
        POMAI_EXPECT_TRUE(res.hits[0].score > 0.9f);  // IP: vec_new·vec_new = 1.0
    }

    {
        pomai::SearchResult res;
        POMAI_EXPECT_OK(db->Search(membrane, vec_tomb, 10, &res));
        bool found = false;
        for (const auto& hit : res.hits) {
            if (hit.id == tomb_id) {
                found = true;
                break;
            }
        }
        POMAI_EXPECT_TRUE(!found);
    }
}

} // namespace
