#include "pomai/write_batch.h"
#include <gtest/gtest.h>

using namespace pomai;

TEST(WriteBatchTest, DefaultConstructor)
{
    WriteBatch batch;
    EXPECT_TRUE(batch.Empty());
    EXPECT_EQ(batch.Count(), 0u);
}

TEST(WriteBatchTest, SinglePut)
{
    WriteBatch batch;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    batch.Put(100, vec);

    EXPECT_FALSE(batch.Empty());
    EXPECT_EQ(batch.Count(), 1u);

    const auto &ops = batch.Ops();
    ASSERT_EQ(ops.size(), 1u);
    EXPECT_EQ(ops[0].type, WriteBatch::OpType::kPut);
    EXPECT_EQ(ops[0].id, 100u);
    EXPECT_EQ(ops[0].vec, vec);
}

TEST(WriteBatchTest, SingleDelete)
{
    WriteBatch batch;
    batch.Delete(200);

    EXPECT_FALSE(batch.Empty());
    EXPECT_EQ(batch.Count(), 1u);

    const auto &ops = batch.Ops();
    ASSERT_EQ(ops.size(), 1u);
    EXPECT_EQ(ops[0].type, WriteBatch::OpType::kDelete);
    EXPECT_EQ(ops[0].id, 200u);
    EXPECT_TRUE(ops[0].vec.empty());
}

TEST(WriteBatchTest, MixedOperations)
{
    WriteBatch batch;
    std::vector<float> vec1 = {1.0f, 2.0f};
    std::vector<float> vec2 = {3.0f, 4.0f};

    batch.Put(1, vec1);
    batch.Delete(2);
    batch.Put(3, vec2);
    batch.Delete(4);

    EXPECT_EQ(batch.Count(), 4u);

    const auto &ops = batch.Ops();
    ASSERT_EQ(ops.size(), 4u);

    EXPECT_EQ(ops[0].type, WriteBatch::OpType::kPut);
    EXPECT_EQ(ops[0].id, 1u);

    EXPECT_EQ(ops[1].type, WriteBatch::OpType::kDelete);
    EXPECT_EQ(ops[1].id, 2u);

    EXPECT_EQ(ops[2].type, WriteBatch::OpType::kPut);
    EXPECT_EQ(ops[2].id, 3u);

    EXPECT_EQ(ops[3].type, WriteBatch::OpType::kDelete);
    EXPECT_EQ(ops[3].id, 4u);
}

TEST(WriteBatchTest, Clear)
{
    WriteBatch batch;
    std::vector<float> vec = {1.0f, 2.0f};

    batch.Put(1, vec);
    batch.Delete(2);
    EXPECT_EQ(batch.Count(), 2u);

    batch.Clear();
    EXPECT_TRUE(batch.Empty());
    EXPECT_EQ(batch.Count(), 0u);
}

TEST(WriteBatchTest, ClearAndReuse)
{
    WriteBatch batch;
    std::vector<float> vec1 = {1.0f, 2.0f};
    std::vector<float> vec2 = {3.0f, 4.0f};

    batch.Put(1, vec1);
    batch.Clear();

    batch.Put(2, vec2);
    EXPECT_EQ(batch.Count(), 1u);

    const auto &ops = batch.Ops();
    ASSERT_EQ(ops.size(), 1u);
    EXPECT_EQ(ops[0].id, 2u);
    EXPECT_EQ(ops[0].vec, vec2);
}

TEST(WriteBatchTest, LargeBatch)
{
    WriteBatch batch;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};

    for (VectorId i = 0; i < 1000; ++i)
    {
        batch.Put(i, vec);
    }

    EXPECT_EQ(batch.Count(), 1000u);
    EXPECT_FALSE(batch.Empty());
}

TEST(WriteBatchTest, EmptyVector)
{
    WriteBatch batch;
    std::vector<float> empty_vec;

    batch.Put(1, empty_vec);
    EXPECT_EQ(batch.Count(), 1u);

    const auto &ops = batch.Ops();
    ASSERT_EQ(ops.size(), 1u);
    EXPECT_TRUE(ops[0].vec.empty());
}

TEST(WriteBatchTest, DuplicateIds)
{
    WriteBatch batch;
    std::vector<float> vec1 = {1.0f, 2.0f};
    std::vector<float> vec2 = {3.0f, 4.0f};

    // Same ID multiple times
    batch.Put(100, vec1);
    batch.Put(100, vec2);
    batch.Delete(100);

    EXPECT_EQ(batch.Count(), 3u);

    const auto &ops = batch.Ops();
    ASSERT_EQ(ops.size(), 3u);
    // All three operations should be present (batch doesn't deduplicate)
    EXPECT_EQ(ops[0].id, 100u);
    EXPECT_EQ(ops[1].id, 100u);
    EXPECT_EQ(ops[2].id, 100u);
}
