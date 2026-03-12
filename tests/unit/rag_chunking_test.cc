// Unit tests for zero-copy RAG chunking (recursive splitter + sliding window).

#include "tests/common/test_main.h"
#include "pomai/rag/chunking.h"
#include <string>
#include <vector>

namespace pomai {

POMAI_TEST(Chunking_Recursive_EmptyInput) {
  RecursiveChunkOptions opts;
  opts.max_chunk_bytes = 100;
  RecursiveChunkSplitter splitter(opts);
  std::vector<ChunkView> out;
  splitter.Chunk("", &out);
  POMAI_EXPECT_TRUE(out.empty());
}

POMAI_TEST(Chunking_Recursive_SingleChunk) {
  std::string doc = "short";
  RecursiveChunkOptions opts;
  opts.max_chunk_bytes = 100;
  RecursiveChunkSplitter splitter(opts);
  std::vector<ChunkView> out;
  splitter.Chunk(doc, &out);
  POMAI_EXPECT_EQ(out.size(), 1u);
  POMAI_EXPECT_EQ(out[0].text, "short");
  POMAI_EXPECT_EQ(out[0].start_offset, 0u);
  // Views refer into original buffer
  POMAI_EXPECT_TRUE(out[0].text.data() == doc.data());
}

POMAI_TEST(Chunking_Recursive_SplitOnNewline) {
  std::string doc = "a\nb\nc\nd\n";
  RecursiveChunkOptions opts;
  opts.max_chunk_bytes = 4;
  opts.separators = {"\n"};
  RecursiveChunkSplitter splitter(opts);
  std::vector<ChunkView> out;
  splitter.Chunk(doc, &out);
  POMAI_EXPECT_TRUE(out.size() >= 2u);
  POMAI_EXPECT_TRUE(out[0].text.data() >= doc.data() && out[0].text.data() < doc.data() + doc.size());
  for (const auto& c : out) {
    POMAI_EXPECT_TRUE(c.text.size() <= 4u);
  }
}

POMAI_TEST(Chunking_Recursive_DefaultSeparators) {
  std::string doc = "first paragraph.\n\nSecond paragraph.\n\nThird.";
  RecursiveChunkOptions opts;
  opts.max_chunk_bytes = 64;
  RecursiveChunkSplitter splitter(opts);
  std::vector<ChunkView> out;
  splitter.Chunk(doc, &out);
  POMAI_EXPECT_TRUE(out.size() >= 1u);
  for (const auto& c : out) {
    POMAI_EXPECT_TRUE(c.text.data() >= doc.data() && c.text.data() + c.text.size() <= doc.data() + doc.size());
  }
}

POMAI_TEST(Chunking_Recursive_ExactBoundary) {
  std::string doc = "1234567890";  // 10 chars, no separator
  RecursiveChunkOptions opts;
  opts.max_chunk_bytes = 5;
  opts.separators = {" "};
  RecursiveChunkSplitter splitter(opts);
  std::vector<ChunkView> out;
  splitter.Chunk(doc, &out);
  // No space: splitter respects max_chunk_bytes and emits fixed-size chunks
  POMAI_EXPECT_EQ(out.size(), 2u);
  POMAI_EXPECT_EQ(out[0].text, "12345");
  POMAI_EXPECT_EQ(out[1].text, "67890");
  POMAI_EXPECT_TRUE(out[0].text.data() == doc.data());
}

POMAI_TEST(Chunking_Recursive_ViewsReferIntoDocument) {
  std::string doc = "one two three four five";
  RecursiveChunkOptions opts;
  opts.max_chunk_bytes = 8;
  opts.separators = {" "};
  RecursiveChunkSplitter splitter(opts);
  std::vector<ChunkView> out;
  splitter.Chunk(doc, &out);
  for (const auto& c : out) {
    POMAI_EXPECT_TRUE(c.text.data() >= doc.data());
    POMAI_EXPECT_TRUE(c.text.data() + c.text.size() <= doc.data() + doc.size());
  }
}

POMAI_TEST(Chunking_SlidingWindow_Empty) {
  SlidingWindowOptions opts;
  opts.chunk_bytes = 10;
  opts.stride_bytes = 10;
  SlidingWindowChunker chunker(opts);
  std::vector<ChunkView> out;
  chunker.Chunk("", &out);
  POMAI_EXPECT_TRUE(out.empty());
}

POMAI_TEST(Chunking_SlidingWindow_SingleChunk) {
  std::string doc = "hello";
  SlidingWindowOptions opts;
  opts.chunk_bytes = 20;
  opts.stride_bytes = 20;
  SlidingWindowChunker chunker(opts);
  std::vector<ChunkView> out;
  chunker.Chunk(doc, &out);
  POMAI_EXPECT_EQ(out.size(), 1u);
  POMAI_EXPECT_EQ(out[0].text, "hello");
  POMAI_EXPECT_EQ(out[0].start_offset, 0u);
  POMAI_EXPECT_TRUE(out[0].text.data() == doc.data());
}

POMAI_TEST(Chunking_SlidingWindow_MultipleChunks) {
  std::string doc = "0123456789";  // 10 chars
  SlidingWindowOptions opts;
  opts.chunk_bytes = 4;
  opts.stride_bytes = 4;
  SlidingWindowChunker chunker(opts);
  std::vector<ChunkView> out;
  chunker.Chunk(doc, &out);
  POMAI_EXPECT_EQ(out.size(), 3u);  // 0-4, 4-8, 8-10
  POMAI_EXPECT_EQ(out[0].text, "0123");
  POMAI_EXPECT_EQ(out[0].start_offset, 0u);
  POMAI_EXPECT_EQ(out[1].text, "4567");
  POMAI_EXPECT_EQ(out[1].start_offset, 4u);
  POMAI_EXPECT_EQ(out[2].text, "89");
  POMAI_EXPECT_EQ(out[2].start_offset, 8u);
  for (const auto& c : out) {
    POMAI_EXPECT_TRUE(c.text.data() >= doc.data() && c.text.data() + c.text.size() <= doc.data() + doc.size());
  }
}

POMAI_TEST(Chunking_SlidingWindow_Overlap) {
  std::string doc = "abcdefghij";  // 10 chars
  SlidingWindowOptions opts;
  opts.chunk_bytes = 4;
  opts.stride_bytes = 2;
  SlidingWindowChunker chunker(opts);
  std::vector<ChunkView> out;
  chunker.Chunk(doc, &out);
  POMAI_EXPECT_TRUE(out.size() >= 4u);
  POMAI_EXPECT_EQ(out[0].text, "abcd");
  POMAI_EXPECT_EQ(out[0].start_offset, 0u);
  POMAI_EXPECT_EQ(out[1].text, "cdef");
  POMAI_EXPECT_EQ(out[1].start_offset, 2u);
}

}  // namespace pomai
