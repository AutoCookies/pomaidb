#include "tests/common/test_main.h"

#include <cstdint>
#include <memory>
#include <string>
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

    POMAI_TEST(Membrane_CreateOpen_IsolatedPutSearchDelete)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-membrane-integ");
        opt.dim = 8;
        opt.shard_count = 4;
        opt.fsync = pomai::FsyncPolicy::kNever;

        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

        pomai::MembraneSpec a;
        a.name = "A";
        a.dim = opt.dim;
        a.shard_count = opt.shard_count;

        pomai::MembraneSpec b;
        b.name = "B";
        b.dim = opt.dim;
        b.shard_count = opt.shard_count;

        POMAI_EXPECT_OK(db->CreateMembrane(a));
        POMAI_EXPECT_OK(db->CreateMembrane(b));

        POMAI_EXPECT_OK(db->OpenMembrane("A"));
        POMAI_EXPECT_OK(db->OpenMembrane("B"));

        // IMPORTANT: score is DOT(query, vec). Use opposite vectors so each query matches its own.
        const auto vA = MakeVec(opt.dim, 1.0f);
        const auto vB = MakeVec(opt.dim, -1.0f);

        // Same ID in different membranes must be isolated.
        POMAI_EXPECT_OK(db->Put("A", 100, vA));
        POMAI_EXPECT_OK(db->Put("B", 100, vB));
        POMAI_EXPECT_OK(db->Flush());

        {
            pomai::SearchResult r;
            POMAI_EXPECT_OK(db->Search("A", vA, /*topk*/ 3, &r));
            POMAI_EXPECT_TRUE(!r.hits.empty());
            POMAI_EXPECT_EQ(r.hits[0].id, static_cast<pomai::VectorId>(100));
        }
        {
            pomai::SearchResult r;
            POMAI_EXPECT_OK(db->Search("B", vB, /*topk*/ 3, &r));
            POMAI_EXPECT_TRUE(!r.hits.empty());
            POMAI_EXPECT_EQ(r.hits[0].id, static_cast<pomai::VectorId>(100));
        }

        // Delete in A must not affect B.
        POMAI_EXPECT_OK(db->Delete("A", 100));
        POMAI_EXPECT_OK(db->Flush());

        {
            pomai::SearchResult r;
            POMAI_EXPECT_OK(db->Search("A", vA, /*topk*/ 3, &r));
            for (const auto &h : r.hits)
            {
                POMAI_EXPECT_TRUE(h.id != static_cast<pomai::VectorId>(100));
            }
        }
        {
            pomai::SearchResult r;
            POMAI_EXPECT_OK(db->Search("B", vB, /*topk*/ 3, &r));
            POMAI_EXPECT_TRUE(!r.hits.empty());
            POMAI_EXPECT_EQ(r.hits[0].id, static_cast<pomai::VectorId>(100));
        }

        POMAI_EXPECT_OK(db->CloseMembrane("A"));
        POMAI_EXPECT_OK(db->CloseMembrane("B"));
        POMAI_EXPECT_OK(db->Close());
    }

    POMAI_TEST(Membrane_ListMembranes_Sorted)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-membrane-list");
        opt.dim = 8;
        opt.shard_count = 2;
        opt.fsync = pomai::FsyncPolicy::kNever;

        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

        auto make = [&](const char *name)
        {
            pomai::MembraneSpec s;
            s.name = name;
            s.dim = opt.dim;
            s.shard_count = opt.shard_count;
            return s;
        };

        POMAI_EXPECT_OK(db->CreateMembrane(make("zulu")));
        POMAI_EXPECT_OK(db->CreateMembrane(make("alpha")));
        POMAI_EXPECT_OK(db->CreateMembrane(make("mu")));

        std::vector<std::string> names;
        POMAI_EXPECT_OK(db->ListMembranes(&names));

        // NOTE: DB open auto-creates "__default__" membrane.
        POMAI_EXPECT_EQ(names.size(), static_cast<std::size_t>(4));
        POMAI_EXPECT_EQ(names[0], std::string("__default__"));
        POMAI_EXPECT_EQ(names[1], std::string("alpha"));
        POMAI_EXPECT_EQ(names[2], std::string("mu"));
        POMAI_EXPECT_EQ(names[3], std::string("zulu"));

        POMAI_EXPECT_OK(db->Close());
    }

} // namespace
