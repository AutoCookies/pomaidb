// Minimal RAG benchmark: ingest chunks, lexical query, hybrid query.

#include "pomai/pomai.h"
#include "pomai/rag.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

using namespace std::chrono;

namespace
{
    std::vector<pomai::TokenId> MakeTokens(std::mt19937& rng, std::size_t count, pomai::TokenId vocab)
    {
        std::uniform_int_distribution<pomai::TokenId> dist(1, vocab);
        std::vector<pomai::TokenId> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back(dist(rng));
        }
        return out;
    }
}

int main(int argc, char** argv)
{
    std::uint32_t num_chunks = 20000;
    std::uint32_t tokens_per_chunk = 64;
    std::uint32_t dim = 32;

    if (argc > 1) num_chunks = static_cast<std::uint32_t>(std::atoi(argv[1]));
    if (argc > 2) tokens_per_chunk = static_cast<std::uint32_t>(std::atoi(argv[2]));
    if (argc > 3) dim = static_cast<std::uint32_t>(std::atoi(argv[3]));

    printf("=============================================================\n");
    printf("                      RAG Benchmark\n");
    printf("=============================================================\n");
    printf("Chunks:          %u\n", num_chunks);
    printf("Tokens/chunk:    %u\n", tokens_per_chunk);
    printf("Embedding dim:   %u\n", dim);
    printf("=============================================================\n\n");

    std::filesystem::remove_all("/tmp/rag_bench");

    pomai::DBOptions opts;
    opts.path = "/tmp/rag_bench";
    opts.dim = dim;
    opts.shard_count = std::max(1u, static_cast<unsigned>(std::thread::hardware_concurrency()));
    opts.fsync = pomai::FsyncPolicy::kNever;

    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opts, &db);
    if (!st.ok()) {
        fprintf(stderr, "Open failed: %s\n", st.message());
        return 1;
    }

    pomai::MembraneSpec rag;
    rag.name = "rag";
    rag.dim = dim;
    rag.shard_count = opts.shard_count;
    rag.kind = pomai::MembraneKind::kRag;
    st = db->CreateMembrane(rag);
    if (!st.ok() && st.code() != pomai::ErrorCode::kAlreadyExists) {
        fprintf(stderr, "CreateMembrane failed: %s\n", st.message());
        return 1;
    }
    st = db->OpenMembrane("rag");
    if (!st.ok()) {
        fprintf(stderr, "OpenMembrane failed: %s\n", st.message());
        return 1;
    }

    printf("[1/3] Ingesting chunks...\n");
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vec(dim);
    for (std::uint32_t i = 0; i < num_chunks; ++i) {
        for (auto& v : vec) v = dist(rng);
        pomai::RagChunk chunk;
        chunk.chunk_id = i + 1;
        chunk.doc_id = (i / 10) + 1;
        chunk.tokens = MakeTokens(rng, tokens_per_chunk, 1000);
        chunk.vec = pomai::VectorView(vec);

        st = db->PutChunk("rag", chunk);
        if (!st.ok()) {
            fprintf(stderr, "PutChunk failed: %s\n", st.message());
            return 1;
        }
    }

    printf("[2/3] Lexical query...\n");
    auto query_tokens = MakeTokens(rng, 4, 1000);
    pomai::RagQuery lexical_query;
    lexical_query.tokens = query_tokens;
    lexical_query.topk = 10;

    pomai::RagSearchOptions opts_lex;
    opts_lex.candidate_budget = 200;

    pomai::RagSearchResult lex_out;
    auto t0 = high_resolution_clock::now();
    st = db->SearchRag("rag", lexical_query, opts_lex, &lex_out);
    auto t1 = high_resolution_clock::now();
    if (!st.ok()) {
        fprintf(stderr, "SearchRag (lexical) failed: %s\n", st.message());
        return 1;
    }

    printf("[3/3] Hybrid query...\n");
    pomai::RagQuery hybrid_query = lexical_query;
    hybrid_query.vec = pomai::VectorView(vec);
    pomai::RagSearchOptions opts_hybrid = opts_lex;

    pomai::RagSearchResult hybrid_out;
    auto t2 = high_resolution_clock::now();
    st = db->SearchRag("rag", hybrid_query, opts_hybrid, &hybrid_out);
    auto t3 = high_resolution_clock::now();
    if (!st.ok()) {
        fprintf(stderr, "SearchRag (hybrid) failed: %s\n", st.message());
        return 1;
    }

    printf("\n=============================================================\n");
    printf("                      RESULTS\n");
    printf("=============================================================\n");
    const double lex_ms = duration<double, std::milli>(t1 - t0).count();
    const double hyb_ms = duration<double, std::milli>(t3 - t2).count();
    printf("Lexical latency:     %.3f ms\n", lex_ms);
    printf("Hybrid latency:      %.3f ms\n", hyb_ms);
    printf("Lexical candidates:  %u\n", lex_out.explain.lexical_candidates);
    printf("Vector candidates:   %u\n", hybrid_out.explain.vector_candidates);
    printf("Candidate reduction: %u -> %zu\n",
           lex_out.explain.lexical_candidates,
           hybrid_out.hits.size());
    printf("Memory usage:        (not measured)\n");
    printf("=============================================================\n");

    db->Close();
    std::filesystem::remove_all("/tmp/rag_bench");
    return 0;
}
