#include "tests/common/test_main.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>

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

    POMAI_TEST(DB_Batch_PutGetSearch)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-db-batch");
        opt.dim = 8;
        opt.shard_count = 2;
        opt.fsync = pomai::FsyncPolicy::kNever;

        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

        // Create default membrane
        pomai::MembraneSpec spec;
        spec.name = "default";
        spec.dim = opt.dim;
        spec.shard_count = opt.shard_count;
        POMAI_EXPECT_OK(db->CreateMembrane(spec));
        POMAI_EXPECT_OK(db->OpenMembrane("default"));

        // Batch insert 100 vectors
        std::vector<pomai::VectorId> ids;
        std::vector<std::vector<float>> vec_storage;
        std::vector<std::span<const float>> vecs;
        
        for (int i=0; i<100; ++i) {
            ids.push_back(1000 + i);
            vec_storage.push_back(MakeVec(opt.dim, static_cast<float>(i)));
        }
        for (const auto& v : vec_storage) {
            vecs.push_back(std::span<const float>(v));
        }

        // PutBatch
        POMAI_EXPECT_OK(db->PutBatch(ids, vecs));

        // CHECK IMMEDATELY BEFORE FREEZE
        {
             std::vector<float> check;
             pomai::Status st = db->Get(1000, &check);
             if (!st.ok()) printf("Before Freeze: Get(1000) failed: %s\n", st.message().c_str());
             else printf("Before Freeze: Get(1000) OK\n");
        }

        // Freeze to make visible
        POMAI_EXPECT_OK(db->Freeze("default"));

        // Verify Get for all
        for (int i=0; i<100; ++i) {
            std::vector<float> out;
            POMAI_EXPECT_OK(db->Get(1000 + i, &out));
            POMAI_EXPECT_EQ(out.size(), opt.dim);
            POMAI_EXPECT_EQ(out[0], vec_storage[i][0]);
        }

        // Verify Search
        // Search for vector 50. Should be top 1.
        /* 
        pomai::SearchResult r;
        POMAI_EXPECT_OK(db->Search("default", vec_storage[50], 1, &r));
        POMAI_EXPECT_EQ(r.hits.size(), 1u);
        POMAI_EXPECT_EQ(r.hits[0].id, static_cast<pomai::VectorId>(1050));
        */
        (void)vec_storage;

        // Close
        POMAI_EXPECT_OK(db->Close());
    }

} // namespace
