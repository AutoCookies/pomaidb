#include "tests/common/test_main.h"

#include <filesystem>
#include <string>
#include <vector>

#include "pomai/status.h"
#include "table/segment.h"
#include "tests/common/test_tmpdir.h"

namespace
{
    namespace fs = std::filesystem;

    POMAI_TEST(Segment_BuildAndGet)
    {
        const std::string root = pomai::test::TempDir("pomai-seg-build");
        const std::string path = (fs::path(root) / "seg_0.dat").string();
        
        const uint32_t dim = 4;
        pomai::table::SegmentBuilder builder(path, dim);
        
        std::vector<float> v1 = {1.0, 2.0, 3.0, 4.0};
        std::vector<float> v2 = {5.0, 6.0, 7.0, 8.0};
        std::vector<float> v3 = {9.0, 10.0, 11.0, 12.0};
        
        // Add out of order to test sorting
        POMAI_EXPECT_OK(builder.Add(20, std::span<const float>(v2), /*is_deleted=*/false));
        POMAI_EXPECT_OK(builder.Add(10, std::span<const float>(v1), /*is_deleted=*/false));
        POMAI_EXPECT_OK(builder.Add(30, std::span<const float>(v3), /*is_deleted=*/true)); // Tombstone
        
        POMAI_EXPECT_OK(builder.Finish());
        
        POMAI_EXPECT_EQ(builder.Count(), 3);
        POMAI_EXPECT_TRUE(fs::exists(path));
        
        // Re-open
        std::unique_ptr<pomai::table::SegmentReader> reader;
        POMAI_EXPECT_OK(pomai::table::SegmentReader::Open(path, &reader));
        
        POMAI_EXPECT_EQ(reader->Count(), 3);
        POMAI_EXPECT_EQ(reader->Dim(), 4);
        
        // Get existing
        std::span<const float> out;
        std::vector<float> decoded;
        POMAI_EXPECT_TRUE(reader->FindAndDecode(10, &out, &decoded, nullptr) == pomai::table::SegmentReader::FindResult::kFound);
        POMAI_EXPECT_EQ(out.size(), 4);
        POMAI_EXPECT_TRUE(std::abs(out[0] - 1.0f) < 0.1f);
        
        POMAI_EXPECT_TRUE(reader->FindAndDecode(20, &out, &decoded, nullptr) == pomai::table::SegmentReader::FindResult::kFound);
        POMAI_EXPECT_TRUE(std::abs(out[0] - 5.0f) < 0.1f);
        
        // Get Tombstone
        auto res = reader->FindAndDecode(30, &out, &decoded, nullptr);
        POMAI_EXPECT_TRUE(res == pomai::table::SegmentReader::FindResult::kFoundTombstone);
        
        // Find API
        POMAI_EXPECT_TRUE(reader->FindAndDecode(10, nullptr, nullptr, nullptr) == pomai::table::SegmentReader::FindResult::kFound);
        POMAI_EXPECT_TRUE(reader->FindAndDecode(20, nullptr, nullptr, nullptr) == pomai::table::SegmentReader::FindResult::kFound);
        POMAI_EXPECT_TRUE(reader->FindAndDecode(30, nullptr, nullptr, nullptr) == pomai::table::SegmentReader::FindResult::kFoundTombstone);
        POMAI_EXPECT_TRUE(reader->FindAndDecode(99, nullptr, nullptr, nullptr) == pomai::table::SegmentReader::FindResult::kNotFound);

        // Get non-existing
        res = reader->FindAndDecode(15, &out, &decoded, nullptr);
        POMAI_EXPECT_TRUE(res == pomai::table::SegmentReader::FindResult::kNotFound);
        
        // ForEach
        int count = 0;
        int tombstones = 0;
        reader->ForEach([&](pomai::VectorId id, std::span<const float> /*vec*/, bool is_deleted, const pomai::Metadata*) {
             count++;
             if (is_deleted) tombstones++;
             if (id == 30) POMAI_EXPECT_TRUE(is_deleted);
             if (id == 10 || id == 20) POMAI_EXPECT_TRUE(!is_deleted);
        });
        POMAI_EXPECT_EQ(count, 3);
        POMAI_EXPECT_EQ(tombstones, 1);
        
        res = reader->FindAndDecode(5, &out, &decoded, nullptr); // Check before first
        POMAI_EXPECT_TRUE(res == pomai::table::SegmentReader::FindResult::kNotFound);
        
        res = reader->FindAndDecode(40, &out, &decoded, nullptr); // Check after last
        POMAI_EXPECT_TRUE(res == pomai::table::SegmentReader::FindResult::kNotFound);
    }

} // namespace
