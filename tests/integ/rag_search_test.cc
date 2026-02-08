#include "tests/common/test_main.h"

#include <memory>
#include <vector>

#include "pomai/pomai.h"
#include "pomai/rag.h"
#include "tests/common/test_tmpdir.h"

namespace
{
    POMAI_TEST(RagSearch_LexicalAndVector)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-rag-search");
        opt.dim = 3;
        opt.shard_count = 2;

        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

        pomai::MembraneSpec rag;
        rag.name = "rag";
        rag.dim = opt.dim;
        rag.shard_count = opt.shard_count;
        rag.kind = pomai::MembraneKind::kRag;
        POMAI_EXPECT_OK(db->CreateMembrane(rag));
        POMAI_EXPECT_OK(db->OpenMembrane("rag"));

        std::vector<float> vec_a{1.0f, 0.0f, 0.0f};
        std::vector<float> vec_b{0.0f, 1.0f, 0.0f};

        pomai::RagChunk a;
        a.chunk_id = 10;
        a.doc_id = 1;
        a.tokens = {100, 200};
        a.vec = pomai::VectorView(vec_a);

        pomai::RagChunk b;
        b.chunk_id = 20;
        b.doc_id = 2;
        b.tokens = {200, 300};
        b.vec = pomai::VectorView(vec_b);

        POMAI_EXPECT_OK(db->PutChunk("rag", a));
        POMAI_EXPECT_OK(db->PutChunk("rag", b));

        pomai::RagQuery query;
        std::vector<pomai::TokenId> tokens = {200};
        query.tokens = tokens;
        query.vec = pomai::VectorView(vec_a);
        query.topk = 2;

        pomai::RagSearchResult out;
        pomai::RagSearchOptions opts;
        opts.candidate_budget = 10;
        opts.token_budget = 0;
        POMAI_EXPECT_OK(db->SearchRag("rag", query, opts, &out));
        POMAI_EXPECT_TRUE(!out.hits.empty());
        POMAI_EXPECT_EQ(out.hits[0].chunk_id, static_cast<pomai::ChunkId>(10));
    }
} // namespace
