#include "tests/common/test_main.h"

#include "pomai/membrane_capabilities.h"

#include <cstdint>
#include <cstring>

namespace {

POMAI_TEST(MembraneCapabilities_VectorSnapshot) {
    const auto c = pomai::GetMembraneKindCapabilities(pomai::MembraneKind::kVector);
    POMAI_EXPECT_EQ(static_cast<int>(c.kind), static_cast<int>(pomai::MembraneKind::kVector));
    POMAI_EXPECT_TRUE(c.read_path);
    POMAI_EXPECT_TRUE(c.write_path);
    POMAI_EXPECT_TRUE(c.unified_scan);
    POMAI_EXPECT_TRUE(c.snapshot_isolated_scan);
    POMAI_EXPECT_EQ(c.stability, pomai::MembraneStability::kStable);
}

POMAI_TEST(MembraneCapabilities_KvNoSnapshotScan) {
    const auto c = pomai::GetMembraneKindCapabilities(pomai::MembraneKind::kKeyValue);
    POMAI_EXPECT_TRUE(c.unified_scan);
    POMAI_EXPECT_TRUE(!c.snapshot_isolated_scan);
}

POMAI_TEST(MembraneCapabilities_ExperimentalSimd) {
    const auto sp = pomai::GetMembraneKindCapabilities(pomai::MembraneKind::kSpatial);
    POMAI_EXPECT_EQ(sp.stability, pomai::MembraneStability::kExperimental);
    const auto mesh = pomai::GetMembraneKindCapabilities(pomai::MembraneKind::kMesh);
    POMAI_EXPECT_EQ(mesh.stability, pomai::MembraneStability::kExperimental);
}

POMAI_TEST(MembraneCapabilities_AllKindsValid) {
    for (uint8_t v = 0; v <= static_cast<uint8_t>(pomai::MembraneKind::kMeta); ++v) {
        POMAI_EXPECT_TRUE(pomai::IsValidMembraneKindValue(v));
        const auto k = static_cast<pomai::MembraneKind>(v);
        const auto c = pomai::GetMembraneKindCapabilities(k);
        POMAI_EXPECT_EQ(static_cast<uint8_t>(c.kind), v);
        POMAI_EXPECT_TRUE(c.read_path);
        POMAI_EXPECT_TRUE(c.write_path);
        POMAI_EXPECT_TRUE(c.unified_scan);
        if (k == pomai::MembraneKind::kVector) {
            POMAI_EXPECT_TRUE(c.snapshot_isolated_scan);
        } else {
            POMAI_EXPECT_TRUE(!c.snapshot_isolated_scan);
        }
    }
}

POMAI_TEST(MembraneCapabilities_ApiNames) {
    POMAI_EXPECT_TRUE(std::strcmp(pomai::MembraneKindApiName(pomai::MembraneKind::kVector), "vector") == 0);
    POMAI_EXPECT_TRUE(std::strcmp(pomai::MembraneKindApiName(pomai::MembraneKind::kMeta), "meta") == 0);
}

} // namespace
