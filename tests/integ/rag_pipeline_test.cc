// Integration tests for Edge RAG pipeline (IngestDocument + RetrieveContext).

#include "tests/common/test_main.h"
#include "pomai/pomai.h"
#include "pomai/rag/embedding_provider.h"
#include "pomai/rag/pipeline.h"
#include "tests/common/test_tmpdir.h"
#include <memory>
#include <string>

namespace pomai {

POMAI_TEST(RagPipeline_IngestAndRetrieve) {
  pomai::DBOptions opt;
  opt.path = pomai::test::TempDir("pomai-rag-pipeline");
  opt.dim = 4;
  opt.shard_count = 2;

  std::unique_ptr<pomai::DB> db;
  POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));

  pomai::MembraneSpec rag;
  rag.name = "rag";
  rag.dim = 4;
  rag.shard_count = 2;
  rag.kind = pomai::MembraneKind::kRag;
  POMAI_EXPECT_OK(db->CreateMembrane(rag));
  POMAI_EXPECT_OK(db->OpenMembrane("rag"));

  MockEmbeddingProvider provider(4);
  RagPipelineOptions pipe_opts;
  pipe_opts.max_chunk_bytes = 64;
  pipe_opts.max_chunks_per_batch = 10;

  RagPipeline pipeline(db.get(), "rag", 4, &provider, pipe_opts);

  std::string doc = "The quick brown fox jumps over the lazy dog. "
                    "RAG pipelines chunk and embed text for retrieval.";
  POMAI_EXPECT_OK(pipeline.IngestDocument(1, doc));

  std::string context;
  POMAI_EXPECT_OK(pipeline.RetrieveContext("quick fox", 5, &context));
  POMAI_EXPECT_TRUE(!context.empty());
}

POMAI_TEST(RagPipeline_EmptyDocumentOk) {
  pomai::DBOptions opt;
  opt.path = pomai::test::TempDir("pomai-rag-pipeline-empty");
  opt.dim = 4;
  opt.shard_count = 2;

  std::unique_ptr<pomai::DB> db;
  POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));
  pomai::MembraneSpec rag;
  rag.name = "rag";
  rag.dim = 4;
  rag.shard_count = 2;
  rag.kind = pomai::MembraneKind::kRag;
  POMAI_EXPECT_OK(db->CreateMembrane(rag));
  POMAI_EXPECT_OK(db->OpenMembrane("rag"));

  MockEmbeddingProvider provider(4);
  RagPipeline pipeline(db.get(), "rag", 4, &provider, {});

  POMAI_EXPECT_OK(pipeline.IngestDocument(2, ""));
  std::string context;
  POMAI_EXPECT_OK(pipeline.RetrieveContext("query", 5, &context));
  POMAI_EXPECT_TRUE(context.empty());
}

POMAI_TEST(RagPipeline_RetrieveContextFormattedFromChunkText) {
  pomai::DBOptions opt;
  opt.path = pomai::test::TempDir("pomai-rag-pipeline-format");
  opt.dim = 4;
  opt.shard_count = 2;

  std::unique_ptr<pomai::DB> db;
  POMAI_EXPECT_OK(pomai::DB::Open(opt, &db));
  pomai::MembraneSpec rag;
  rag.name = "rag";
  rag.dim = 4;
  rag.shard_count = 2;
  rag.kind = pomai::MembraneKind::kRag;
  POMAI_EXPECT_OK(db->CreateMembrane(rag));
  POMAI_EXPECT_OK(db->OpenMembrane("rag"));

  MockEmbeddingProvider provider(4);
  RagPipeline pipeline(db.get(), "rag", 4, &provider, {});

  POMAI_EXPECT_OK(pipeline.IngestDocument(1, "Alpha content here."));
  POMAI_EXPECT_OK(pipeline.IngestDocument(2, "Beta content there."));

  std::string context;
  POMAI_EXPECT_OK(pipeline.RetrieveContext("content", 10, &context));
  POMAI_EXPECT_TRUE(context.find("Alpha") != std::string::npos || context.find("Beta") != std::string::npos);
}

}  // namespace pomai
