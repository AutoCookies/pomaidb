// Unit tests for EmbeddingProvider (MockEmbeddingProvider).

#include "tests/common/test_main.h"
#include "pomai/rag/embedding_provider.h"
#include <cmath>
#include <string_view>

namespace pomai {

POMAI_TEST(Embedding_Mock_Dim) {
  MockEmbeddingProvider p(64);
  POMAI_EXPECT_EQ(p.Dim(), 64u);
}

POMAI_TEST(Embedding_Mock_SingleText) {
  MockEmbeddingProvider p(8);
  std::vector<std::string_view> texts = {"hello"};
  std::vector<std::vector<float>> out;
  POMAI_EXPECT_OK(p.Embed(texts, &out, 8));
  POMAI_EXPECT_EQ(out.size(), 1u);
  POMAI_EXPECT_EQ(out[0].size(), 8u);
}

POMAI_TEST(Embedding_Mock_Deterministic) {
  MockEmbeddingProvider p(4);
  std::vector<std::string_view> texts = {"same"};
  std::vector<std::vector<float>> a, b;
  POMAI_EXPECT_OK(p.Embed(texts, &a, 4));
  POMAI_EXPECT_OK(p.Embed(texts, &b, 4));
  POMAI_EXPECT_EQ(a.size(), 1u);
  POMAI_EXPECT_EQ(b.size(), 1u);
  for (size_t i = 0; i < 4; ++i)
    POMAI_EXPECT_EQ(a[0][i], b[0][i]);
}

POMAI_TEST(Embedding_Mock_Batch) {
  MockEmbeddingProvider p(4);
  std::vector<std::string_view> texts = {"a", "b", "c"};
  std::vector<std::vector<float>> out;
  POMAI_EXPECT_OK(p.Embed(texts, &out, 4));
  POMAI_EXPECT_EQ(out.size(), 3u);
  for (const auto& v : out)
    POMAI_EXPECT_EQ(v.size(), 4u);
}

POMAI_TEST(Embedding_Mock_DifferentTextsDifferentVectors) {
  MockEmbeddingProvider p(8);
  std::vector<std::string_view> texts = {"x", "y"};
  std::vector<std::vector<float>> out;
  POMAI_EXPECT_OK(p.Embed(texts, &out, 8));
  POMAI_EXPECT_EQ(out.size(), 2u);
  bool same = true;
  for (size_t i = 0; i < 8; ++i)
    if (out[0][i] != out[1][i]) { same = false; break; }
  POMAI_EXPECT_TRUE(!same);
}

POMAI_TEST(Embedding_Mock_EmptyInputFails) {
  MockEmbeddingProvider p(4);
  std::vector<std::string_view> texts;
  std::vector<std::vector<float>> out;
  Status st = p.Embed(texts, &out, 4);
  POMAI_EXPECT_TRUE(!st.ok());
}

POMAI_TEST(Embedding_Mock_DimMismatchFails) {
  MockEmbeddingProvider p(4);
  std::vector<std::string_view> texts = {"hi"};
  std::vector<std::vector<float>> out;
  Status st = p.Embed(texts, &out, 8);
  POMAI_EXPECT_TRUE(!st.ok());
}

POMAI_TEST(Embedding_Mock_NullOutFails) {
  MockEmbeddingProvider p(4);
  std::vector<std::string_view> texts = {"hi"};
  Status st = p.Embed(texts, nullptr, 4);
  POMAI_EXPECT_TRUE(!st.ok());
}

}  // namespace pomai
