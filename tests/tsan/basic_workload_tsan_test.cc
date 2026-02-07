#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"

#include <memory>
#include <vector>

#include "pomai/iterator.h"
#include "pomai/pomai.h"

namespace {
using namespace pomai;

POMAI_TEST(Tsan_BasicOpenPutBatchSearchScanClose) {
    DBOptions opt;
    opt.path = pomai::test::TempDir("tsan_basic_workload");
    opt.dim = 8;
    opt.shard_count = 1;
    opt.fsync = FsyncPolicy::kNever;

    std::unique_ptr<DB> db;
    POMAI_EXPECT_OK(DB::Open(opt, &db));

    std::vector<std::vector<float>> owned_vecs(32, std::vector<float>(8, 0.0f));
    std::vector<VectorId> ids;
    std::vector<std::span<const float>> spans;
    ids.reserve(32);
    spans.reserve(32);
    for (std::size_t i = 0; i < owned_vecs.size(); ++i) {
        owned_vecs[i][0] = static_cast<float>(i);
        ids.push_back(static_cast<VectorId>(i + 1));
        spans.push_back(owned_vecs[i]);
    }

    POMAI_EXPECT_OK(db->PutBatch(ids, spans));
    POMAI_EXPECT_OK(db->Freeze("__default__"));

    SearchResult out;
    std::vector<float> q(8, 0.0f);
    q[0] = 12.0f;
    POMAI_EXPECT_OK(db->Search(q, 5, &out));
    POMAI_EXPECT_TRUE(!out.hits.empty());

    std::unique_ptr<SnapshotIterator> it;
    POMAI_EXPECT_OK(db->NewIterator("__default__", &it));
    std::size_t count = 0;
    while (it->Valid()) {
        ++count;
        it->Next();
    }
    POMAI_EXPECT_TRUE(count > 0);

    POMAI_EXPECT_OK(db->Close());
}

} // namespace
