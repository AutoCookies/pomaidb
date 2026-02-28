#include "tests/common/test_main.h"
#include <filesystem>
#include <string>
#include <vector>
#include <cmath>

#include "pomai/status.h"
#include "table/segment.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/options.h"

namespace
{
    namespace fs = std::filesystem;

    POMAI_TEST(FP16_Quantization_BuildAndSearch)
    {
        const std::string root = pomai::test::TempDir("pomai-fp16-test");
        const std::string path = (fs::path(root) / "seg_fp16.dat").string();
        
        const uint32_t dim = 8;
        pomai::IndexParams params;
        params.quant_type = pomai::QuantizationType::kFp16;
        
        pomai::table::SegmentBuilder builder(path, dim, params);
        
        std::vector<float> v1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        std::vector<float> v2 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
        
        POMAI_EXPECT_OK(builder.Add(1, std::span<const float>(v1), false));
        POMAI_EXPECT_OK(builder.Add(2, std::span<const float>(v2), false));
        POMAI_EXPECT_OK(builder.Finish());
        
        // Re-open reader
        std::unique_ptr<pomai::table::SegmentReader> reader;
        POMAI_EXPECT_OK(pomai::table::SegmentReader::Open(path, &reader));
        
        POMAI_EXPECT_EQ(reader->GetQuantType(), pomai::QuantizationType::kFp16);
        
        // Find and Decode
        std::span<const float> out_span;
        std::vector<float> decoded;
        auto res = reader->FindAndDecode(1, &out_span, &decoded, nullptr);
        POMAI_EXPECT_TRUE(res == pomai::table::SegmentReader::FindResult::kFound);
        POMAI_EXPECT_EQ(decoded.size(), dim);
        
        // Precision check: FP16 should be very close to FP32 for these small values
        for (uint32_t i = 0; i < dim; ++i) {
            POMAI_EXPECT_TRUE(std::abs(decoded[i] - v1[i]) < 0.01f);
        }
        
        // Test ComputeDistance via Quantizer
        const pomai::core::VectorQuantizer<float>* q = reader->GetQuantizer();
        POMAI_EXPECT_TRUE(q != nullptr);
        
        std::span<const uint8_t> codes;
        POMAI_EXPECT_OK(reader->GetQuantized(1, &codes, nullptr));
        POMAI_EXPECT_EQ(codes.size(), dim * 2);
        
        float dist = q->ComputeDistance(std::span<const float>(v1), codes);
        // Dot product of v1 with itself should be roughly sum(i^2) = 1+4+9+16+25+36+49+64 = 204
        POMAI_EXPECT_TRUE(std::abs(dist - 204.0f) < 0.1f);
    }

} // namespace
