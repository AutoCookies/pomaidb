#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <chrono>
#include <thread>

#include "pomai/pomai.h"

namespace {
using namespace pomai;

POMAI_TEST(NewMembrane_TypedApis_Basic) {
    auto path = pomai::test::TempDir("typed_api_basic");
    DBOptions opt;
    opt.path = path;
    opt.dim = 4;
    opt.shard_count = 1;
    opt.max_kv_entries = 8;
    opt.max_sketch_entries = 8;
    opt.max_blob_bytes_mb = 1;

    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));

    MembraneSpec ts; ts.name = "ts"; ts.kind = MembraneKind::kTimeSeries; ts.shard_count = 1;
    MembraneSpec kv; kv.name = "kv"; kv.kind = MembraneKind::kKeyValue; kv.shard_count = 1;
    MembraneSpec mt; mt.name = "meta"; mt.kind = MembraneKind::kMeta; mt.shard_count = 1;
    MembraneSpec sk; sk.name = "sk"; sk.kind = MembraneKind::kSketch; sk.shard_count = 1;
    MembraneSpec bl; bl.name = "bl"; bl.kind = MembraneKind::kBlob; bl.shard_count = 1;
    POMAI_EXPECT_OK(db->CreateMembrane(ts)); POMAI_EXPECT_OK(db->OpenMembrane("ts"));
    POMAI_EXPECT_OK(db->CreateMembrane(kv)); POMAI_EXPECT_OK(db->OpenMembrane("kv"));
    POMAI_EXPECT_OK(db->CreateMembrane(mt)); POMAI_EXPECT_OK(db->OpenMembrane("meta"));
    POMAI_EXPECT_OK(db->CreateMembrane(sk)); POMAI_EXPECT_OK(db->OpenMembrane("sk"));
    POMAI_EXPECT_OK(db->CreateMembrane(bl)); POMAI_EXPECT_OK(db->OpenMembrane("bl"));

    POMAI_EXPECT_OK(db->TsPut("ts", 1, 100, 42.5));
    std::vector<TimeSeriesPoint> points;
    POMAI_EXPECT_OK(db->TsRange("ts", 1, 99, 101, &points));
    POMAI_EXPECT_EQ(points.size(), 1u);
    POMAI_EXPECT_EQ(points[0].timestamp, 100u);

    POMAI_EXPECT_OK(db->KvPut("kv", "dev.mode", "eco"));
    std::string out;
    POMAI_EXPECT_OK(db->KvGet("kv", "dev.mode", &out));
    POMAI_EXPECT_EQ(out, std::string("eco"));

    POMAI_EXPECT_OK(db->MetaPut("meta", "gid:42", "{\"status\":\"active\",\"location\":\"factory-A\"}"));
    POMAI_EXPECT_OK(db->MetaGet("meta", "gid:42", &out));
    POMAI_EXPECT_EQ(out, std::string("{\"status\":\"active\",\"location\":\"factory-A\"}"));

    POMAI_EXPECT_OK(db->SketchAdd("sk", "id-1", 3));
    uint64_t est = 0;
    POMAI_EXPECT_OK(db->SketchEstimate("sk", "id-1", &est));
    POMAI_EXPECT_EQ(est, 3u);

    std::vector<uint8_t> blob = {1,2,3,4,5};
    POMAI_EXPECT_OK(db->BlobPut("bl", 7, blob));
    std::vector<uint8_t> got;
    POMAI_EXPECT_OK(db->BlobGet("bl", 7, &got));
    POMAI_EXPECT_EQ(got.size(), 5u);
    POMAI_EXPECT_OK(db->Close());
}

POMAI_TEST(NewMembrane_MetaRetention_TtlAndCount) {
    auto path = pomai::test::TempDir("meta_retention");
    DBOptions opt;
    opt.path = path;
    opt.dim = 4;
    opt.shard_count = 1;
    opt.max_kv_entries = 64;

    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));

    MembraneSpec mt;
    mt.name = "meta_ttl";
    mt.kind = MembraneKind::kMeta;
    mt.shard_count = 1;
    mt.ttl_sec = 1;
    POMAI_EXPECT_OK(db->CreateMembrane(mt));
    POMAI_EXPECT_OK(db->OpenMembrane("meta_ttl"));

    std::string out;
    POMAI_EXPECT_OK(db->MetaPut("meta_ttl", "gid:1", "{\"v\":1}"));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    POMAI_EXPECT_TRUE(!db->MetaGet("meta_ttl", "gid:1", &out).ok());

    POMAI_EXPECT_OK(db->MetaPut("meta_ttl", "gid:2", "{\"v\":2}"));
    POMAI_EXPECT_OK(db->MetaGet("meta_ttl", "gid:2", &out));
    POMAI_EXPECT_OK(db->Compact("meta_ttl"));
    POMAI_EXPECT_OK(db->Close());
}

} // namespace

