#include "tests/common/test_main.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "pomai/options.h"
#include "pomai/pomai.h"
#include "pomai/search.h"
#include "tests/common/test_tmpdir.h"

namespace
{

    static std::vector<float> MakeVec(std::uint32_t dim, float base)
    {
        std::vector<float> v(dim);
        for (std::uint32_t i = 0; i < dim; ++i)
            v[i] = base + static_cast<float>(i) * 0.001f;
        return v;
    }

    POMAI_TEST(DB_Basic_OpenPutSearchDelete)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-db-basic");
        opt.dim = 8;
        opt.shard_count = 4;
        opt.fsync = pomai::FsyncPolicy::kNever;

        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

        pomai::MembraneSpec spec;
        spec.name = "default";
        spec.dim = opt.dim;
        spec.shard_count = opt.shard_count;
        POMAI_EXPECT_OK(db->CreateMembrane(spec));
        POMAI_EXPECT_OK(db->OpenMembrane("default"));

        // IMPORTANT: search score is DOT(query, vec) (higher is better).
        // So we choose v2 to be "anti-correlated" with v1 so v1 is ranked above v2.
        const auto v1 = MakeVec(opt.dim, 1.0f);
        const auto v2 = MakeVec(opt.dim, -1.0f);

        POMAI_EXPECT_OK(db->Put("default", 100, v1));
        POMAI_EXPECT_OK(db->Put("default", 200, v2));
        // Must Freeze to make writes visible to Search
        POMAI_EXPECT_OK(db->Freeze("default"));

        pomai::SearchResult r;
        POMAI_EXPECT_OK(db->Search("default", v1, /*topk*/ 2, &r));
        POMAI_EXPECT_TRUE(!r.hits.empty());
        POMAI_EXPECT_EQ(r.hits[0].id, static_cast<pomai::VectorId>(100));

        // Delete then search again: should not return 100 in small topk.
        POMAI_EXPECT_OK(db->Delete("default", 100));
        // Must Freeze to make delete visible
        POMAI_EXPECT_OK(db->Freeze("default"));

        r.Clear();
        POMAI_EXPECT_OK(db->Search("default", v1, /*topk*/ 2, &r));
        for (const auto &h : r.hits)
        {
            POMAI_EXPECT_TRUE(h.id != static_cast<pomai::VectorId>(100));
        }
        {
        // Test Get/Exists
        // Note: New consistency model requires Freeze (Snapshot update) for visibility.
        // Active MemTable is NOT visible to Get/Exists.
        POMAI_EXPECT_OK(db->Freeze("default"));

        pomai::Status st;
        std::vector<float> vec_out;
        st = db->Get("default", 200, &vec_out); 
        POMAI_EXPECT_OK(st);
        POMAI_EXPECT_EQ(vec_out.size(), opt.dim);
        if (!vec_out.empty()) {
            POMAI_EXPECT_EQ(vec_out[0], v2[0]);
        }

        bool exists = false;
        st = db->Exists("default", 200, &exists); 
        POMAI_EXPECT_OK(st);
        POMAI_EXPECT_TRUE(exists);

        st = db->Exists("default", 100, &exists); 
        POMAI_EXPECT_OK(st);
        POMAI_EXPECT_TRUE(!exists);

        st = db->Exists("default", 999, &exists); 
        POMAI_EXPECT_OK(st);
        POMAI_EXPECT_TRUE(!exists);

        POMAI_EXPECT_OK(db->CloseMembrane("default"));
        POMAI_EXPECT_OK(db->Close());
    }
}
} // namespace
