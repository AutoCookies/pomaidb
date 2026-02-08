#include "tests/common/test_main.h"

#include <memory>
#include <vector>

#include "pomai/pomai.h"
#include "pomai/rag.h"
#include "tests/common/test_tmpdir.h"

namespace
{
    POMAI_TEST(RagContract_PutVectorRejectedOnRag)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-rag-contract");
        opt.dim = 4;
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

        std::vector<float> vec{1.0f, 2.0f, 3.0f, 4.0f};
        auto st = db->PutVector("rag", 1, vec);
        POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
    }

    POMAI_TEST(RagContract_PutChunkRejectedOnVector)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-rag-contract-vec");
        opt.dim = 4;
        opt.shard_count = 2;

        std::unique_ptr<pomai::DB> db;
        POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

        pomai::MembraneSpec vec;
        vec.name = "vec";
        vec.dim = opt.dim;
        vec.shard_count = opt.shard_count;
        vec.kind = pomai::MembraneKind::kVector;
        POMAI_EXPECT_OK(db->CreateMembrane(vec));
        POMAI_EXPECT_OK(db->OpenMembrane("vec"));

        pomai::RagChunk chunk;
        chunk.chunk_id = 1;
        chunk.doc_id = 2;
        chunk.tokens = {10, 11};
        auto st = db->PutChunk("vec", chunk);
        POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
    }

    POMAI_TEST(RagContract_TokenBlobRequired)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-rag-contract-token");
        opt.dim = 4;
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

        pomai::RagChunk chunk;
        chunk.chunk_id = 7;
        chunk.doc_id = 8;
        auto st = db->PutChunk("rag", chunk);
        POMAI_EXPECT_TRUE(st.code() == pomai::ErrorCode::kInvalidArgument);
    }

    POMAI_TEST(RagContract_ValidChunkAccepted)
    {
        pomai::DBOptions opt;
        opt.path = pomai::test::TempDir("pomai-rag-contract-ok");
        opt.dim = 4;
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

        pomai::RagChunk chunk;
        chunk.chunk_id = 7;
        chunk.doc_id = 8;
        chunk.tokens = {42, 43, 44};
        auto st = db->PutChunk("rag", chunk);
        POMAI_EXPECT_OK(st);
    }
} // namespace
