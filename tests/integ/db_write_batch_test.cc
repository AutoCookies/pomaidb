#include "pomai/pomai.h"
#include "pomai/write_batch.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <memory>

using namespace pomai;
namespace fs = std::filesystem;

class WriteBatchTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_path_ = "./test_write_batch_" + std::to_string(std::time(nullptr));
        fs::remove_all(test_path_);
    }

    void TearDown() override
    {
        if (db_)
        {
            (void)db_->Close();
            db_.reset();
        }
        fs::remove_all(test_path_);
    }

    std::unique_ptr<DB> OpenDB(uint32_t dim = 3, uint32_t shard_count = 4)
    {
        DBOptions opt;
        opt.path = test_path_;
        opt.dim = dim;
        opt.shard_count = shard_count;
        opt.fsync = FsyncPolicy::kAlways;

        std::unique_ptr<DB> db;
        auto st = DB::Open(opt, &db);
        EXPECT_POMAI_OK(st);
        return db;
    }

    std::string test_path_;
    std::unique_ptr<DB> db_;
};

TEST_F(WriteBatchTest, EmptyBatch)
{
    db_ = OpenDB();

    WriteBatch batch;
    auto st = db_->Write(batch);
    EXPECT_POMAI_OK(st);
}

TEST_F(WriteBatchTest, SinglePut)
{
    db_ = OpenDB();

    WriteBatch batch;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    batch.Put(100, vec);

    auto st = db_->Write(batch);
    EXPECT_POMAI_OK(st);

    // Verify the vector was inserted
    SearchResult result;
    st = db_->Search(vec, 1, &result);
    EXPECT_POMAI_OK(st);
    ASSERT_EQ(result.hits.size(), 1u);
    EXPECT_EQ(result.hits[0].id, 100u);
}

TEST_F(WriteBatchTest, SingleDelete)
{
    db_ = OpenDB();

    std::vector<float> vec = {1.0f, 2.0f, 3.0f};

    // Insert first
    auto st = db_->Put(200, vec);
    EXPECT_POMAI_OK(st);

    // Verify it exists
    SearchResult result;
    st = db_->Search(vec, 1, &result);
    EXPECT_POMAI_OK(st);
    ASSERT_EQ(result.hits.size(), 1u);

    // Delete via batch
    WriteBatch batch;
    batch.Delete(200);
    st = db_->Write(batch);
    EXPECT_POMAI_OK(st);

    // Verify it's gone
    st = db_->Search(vec, 1, &result);
    EXPECT_POMAI_OK(st);
    EXPECT_EQ(result.hits.size(), 0u);
}

TEST_F(WriteBatchTest, MixedOperations)
{
    db_ = OpenDB();

    std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> vec2 = {4.0f, 5.0f, 6.0f};
    std::vector<float> vec3 = {7.0f, 8.0f, 9.0f};

    WriteBatch batch;
    batch.Put(1, vec1);
    batch.Put(2, vec2);
    batch.Put(3, vec3);
    batch.Delete(100); // Delete non-existent (should be no-op)

    auto st = db_->Write(batch);
    EXPECT_POMAI_OK(st);

    // Verify all inserts succeeded
    SearchResult result;
    st = db_->Search(vec1, 1, &result);
    EXPECT_POMAI_OK(st);
    EXPECT_EQ(result.hits.size(), 1u);

    st = db_->Search(vec2, 1, &result);
    EXPECT_POMAI_OK(st);
    EXPECT_EQ(result.hits.size(), 1u);

    st = db_->Search(vec3, 1, &result);
    EXPECT_POMAI_OK(st);
    EXPECT_EQ(result.hits.size(), 1u);
}

TEST_F(WriteBatchTest, LargeBatch)
{
    db_ = OpenDB(3, 4);

    WriteBatch batch;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};

    const size_t count = 100;
    for (size_t i = 0; i < count; ++i)
    {
        batch.Put(i, vec);
    }

    auto st = db_->Write(batch);
    EXPECT_POMAI_OK(st);

    // Verify vectors were inserted
    SearchResult result;
    st = db_->Search(vec, static_cast<uint32_t>(count), &result);
    EXPECT_POMAI_OK(st);
    EXPECT_EQ(result.hits.size(), count);
}

TEST_F(WriteBatchTest, CrossShardBatch)
{
    db_ = OpenDB(3, 4);

    WriteBatch batch;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};

    // Insert vectors that map to different shards
    // With 4 shards: id % 4 determines shard
    batch.Put(0, vec);  // shard 0
    batch.Put(1, vec);  // shard 1
    batch.Put(2, vec);  // shard 2
    batch.Put(3, vec);  // shard 3
    batch.Put(4, vec);  // shard 0
    batch.Put(5, vec);  // shard 1

    auto st = db_->Write(batch);
    EXPECT_POMAI_OK(st);

    // Verify all were inserted
    SearchResult result;
    st = db_->Search(vec, 10, &result);
    EXPECT_POMAI_OK(st);
    EXPECT_EQ(result.hits.size(), 6u);
}

TEST_F(WriteBatchTest, PersistenceAfterBatch)
{
    std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> vec2 = {4.0f, 5.0f, 6.0f};

    {
        db_ = OpenDB();

        WriteBatch batch;
        batch.Put(100, vec1);
        batch.Put(200, vec2);

        auto st = db_->Write(batch);
        EXPECT_POMAI_OK(st);

        st = db_->Flush();
        EXPECT_POMAI_OK(st);

        st = db_->Close();
        EXPECT_POMAI_OK(st);
    }

    // Reopen and verify
    {
        db_ = OpenDB();

        SearchResult result;
        auto st = db_->Search(vec1, 1, &result);
        EXPECT_POMAI_OK(st);
        ASSERT_EQ(result.hits.size(), 1u);
        EXPECT_EQ(result.hits[0].id, 100u);

        st = db_->Search(vec2, 1, &result);
        EXPECT_POMAI_OK(st);
        ASSERT_EQ(result.hits.size(), 1u);
        EXPECT_EQ(result.hits[0].id, 200u);
    }
}

TEST_F(WriteBatchTest, DimensionMismatch)
{
    db_ = OpenDB(3, 2);

    WriteBatch batch;
    std::vector<float> vec_wrong = {1.0f, 2.0f}; // dim=2, expected dim=3
    batch.Put(100, vec_wrong);

    auto st = db_->Write(batch);
    EXPECT_FALSE(st.ok());
    EXPECT_EQ(st.code(), ErrorCode::kInvalidArgument);
}

TEST_F(WriteBatchTest, BatchWithMembranes)
{
    db_ = OpenDB(3, 2);

    // Create a custom membrane
    MembraneSpec spec;
    spec.name = "test_membrane";
    spec.dim = 3;
    spec.shard_count = 2;

    auto st = db_->CreateMembrane(spec);
    EXPECT_POMAI_OK(st);

    st = db_->OpenMembrane("test_membrane");
    EXPECT_POMAI_OK(st);

    // Write batch to custom membrane
    WriteBatch batch;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    batch.Put(100, vec);
    batch.Put(200, vec);

    st = db_->Write("test_membrane", batch);
    EXPECT_POMAI_OK(st);

    // Verify in custom membrane
    SearchResult result;
    st = db_->Search("test_membrane", vec, 10, &result);
    EXPECT_POMAI_OK(st);
    EXPECT_EQ(result.hits.size(), 2u);

    // Verify NOT in default membrane
    st = db_->Search(vec, 10, &result);
    EXPECT_POMAI_OK(st);
    EXPECT_EQ(result.hits.size(), 0u);
}

TEST_F(WriteBatchTest, AtomicityPerShard)
{
    db_ = OpenDB(3, 2);

    WriteBatch batch;
    std::vector<float> vec_good = {1.0f, 2.0f, 3.0f};
    std::vector<float> vec_bad = {1.0f, 2.0f}; // wrong dim

    // Mix good and bad operations
    batch.Put(100, vec_good);
    batch.Put(200, vec_bad); // This should cause failure

    auto st = db_->Write(batch);
    EXPECT_FALSE(st.ok());

    // Due to shard routing, behavior may vary
    // This test documents current behavior rather than prescribing it
    // In the future, may want true cross-shard atomicity
}
