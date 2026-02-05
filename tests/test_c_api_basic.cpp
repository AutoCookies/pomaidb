#include "tests/common/test_main.h"

#include <array>
#include <cstdlib>
#include <string>
#include <vector>

#include "pomai/c_api.h"

namespace {

class CApiFixture {
public:
    CApiFixture() {
        pomai_options_init(&opts_);
        dir_ = "test_db_capi_" + std::to_string(std::rand());
        opts_.path = dir_.c_str();
        opts_.shards = 2;
        opts_.dim = 8;
        std::string cmd = "rm -rf " + dir_;
        std::system(cmd.c_str());
        auto* st = pomai_open(&opts_, &db_);
        if (st != nullptr) {
            std::abort();
        }
    }

    ~CApiFixture() {
        if (db_ != nullptr) {
            pomai_status_t* st = pomai_close(db_);
            if (st != nullptr) {
                pomai_status_free(st);
            }
        }
        std::string cmd = "rm -rf " + dir_;
        std::system(cmd.c_str());
    }

    pomai_db_t* db() { return db_; }

    std::array<float, 8> Vec(float v0) {
        std::array<float, 8> out{};
        out[0] = v0;
        return out;
    }

private:
    std::string dir_;
    pomai_options_t opts_{};
    pomai_db_t* db_ = nullptr;
};

#define CAPI_EXPECT_OK(expr)                                            \
    do {                                                                 \
        pomai_status_t* _st = (expr);                                    \
        if (_st != nullptr) {                                            \
            std::cerr << "status=" << pomai_status_code(_st) << " msg=" \
                      << pomai_status_message(_st) << "\n";             \
            pomai_status_free(_st);                                      \
            std::abort();                                                \
        }                                                                \
    } while (0)

POMAI_TEST(CApiOpenCloseAndVersion) {
    CApiFixture fx;
    POMAI_EXPECT_TRUE(pomai_abi_version() > 0);
    POMAI_EXPECT_TRUE(std::string(pomai_version_string()).size() > 0);
}

POMAI_TEST(CApiPutGetExistsDelete) {
    CApiFixture fx;
    const auto v = fx.Vec(1.5f);
    const char tenant[] = "tenantA";

    pomai_upsert_t up{};
    up.struct_size = sizeof(pomai_upsert_t);
    up.id = 42;
    up.vector = v.data();
    up.dim = static_cast<uint32_t>(v.size());
    up.metadata = reinterpret_cast<const uint8_t*>(tenant);
    up.metadata_len = static_cast<uint32_t>(sizeof(tenant) - 1);

    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    bool exists = false;
    CAPI_EXPECT_OK(pomai_exists(fx.db(), 42, &exists));
    POMAI_EXPECT_TRUE(exists);

    pomai_record_t* rec = nullptr;
    CAPI_EXPECT_OK(pomai_get(fx.db(), 42, &rec));
    POMAI_EXPECT_TRUE(rec != nullptr);
    POMAI_EXPECT_EQ(rec->id, 42u);
    POMAI_EXPECT_EQ(rec->dim, 8u);
    POMAI_EXPECT_EQ(std::string(reinterpret_cast<const char*>(rec->metadata), rec->metadata_len), "tenantA");
    pomai_record_free(rec);

    CAPI_EXPECT_OK(pomai_delete(fx.db(), 42));
    CAPI_EXPECT_OK(pomai_exists(fx.db(), 42, &exists));
    POMAI_EXPECT_TRUE(!exists);
}

POMAI_TEST(CApiPutBatchSearchAndSnapshotScan) {
    CApiFixture fx;
    std::vector<std::array<float, 8>> vectors(16);
    std::vector<pomai_upsert_t> batch(16);
    for (size_t i = 0; i < batch.size(); ++i) {
        vectors[i][0] = static_cast<float>(i);
        batch[i].struct_size = sizeof(pomai_upsert_t);
        batch[i].id = static_cast<uint64_t>(i + 1);
        batch[i].vector = vectors[i].data();
        batch[i].dim = 8;
    }
    CAPI_EXPECT_OK(pomai_put_batch(fx.db(), batch.data(), batch.size()));

    auto qv = fx.Vec(11.0f);
    pomai_query_t query{};
    query.struct_size = sizeof(pomai_query_t);
    query.vector = qv.data();
    query.dim = 8;
    query.topk = 5;

    pomai_search_results_t* results = nullptr;
    CAPI_EXPECT_OK(pomai_search(fx.db(), &query, &results));
    POMAI_EXPECT_TRUE(results != nullptr);
    POMAI_EXPECT_TRUE(results->count >= 1);
    bool found = false;
    for (size_t i = 0; i < results->count; ++i) {
        if (results->ids[i] == 12u) {
            found = true;
        }
    }
    POMAI_EXPECT_TRUE(found);
    pomai_search_results_free(results);

    auto v1 = fx.Vec(100.0f);
    pomai_upsert_t up{};
    up.struct_size = sizeof(pomai_upsert_t);
    up.id = 100;
    up.vector = v1.data();
    up.dim = 8;
    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    pomai_snapshot_t* snap = nullptr;
    CAPI_EXPECT_OK(pomai_get_snapshot(fx.db(), &snap));

    auto v2 = fx.Vec(200.0f);
    up.id = 200;
    up.vector = v2.data();
    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    pomai_scan_options_t scan_opts{};
    pomai_scan_options_init(&scan_opts);
    pomai_iter_t* it = nullptr;
    CAPI_EXPECT_OK(pomai_scan(fx.db(), &scan_opts, snap, &it));

    bool saw200 = false;
    while (pomai_iter_valid(it)) {
        pomai_record_view_t view{};
        view.struct_size = sizeof(pomai_record_view_t);
        CAPI_EXPECT_OK(pomai_iter_get_record(it, &view));
        if (view.id == 200) saw200 = true;
        pomai_iter_next(it);
    }
    CAPI_EXPECT_OK(pomai_iter_status(it));
    pomai_iter_free(it);
    pomai_snapshot_free(snap);

    POMAI_EXPECT_TRUE(!saw200);
}


POMAI_TEST(CApiDeadlines) {
    CApiFixture fx;
    auto qv = fx.Vec(1.0f);
    pomai_query_t query{};
    query.struct_size = sizeof(pomai_query_t);
    query.vector = qv.data();
    query.dim = 8;
    query.topk = 1;
    query.deadline_ms = 1;

    pomai_search_results_t* results = nullptr;
    pomai_status_t* st = pomai_search(fx.db(), &query, &results);
    POMAI_EXPECT_TRUE(st != nullptr);
    POMAI_EXPECT_EQ(pomai_status_code(st), static_cast<int>(POMAI_STATUS_DEADLINE_EXCEEDED));
    pomai_status_free(st);
}

POMAI_TEST(CApiInvalidArgs) {
    pomai_status_t* st = pomai_put(nullptr, nullptr);
    POMAI_EXPECT_TRUE(st != nullptr);
    POMAI_EXPECT_EQ(pomai_status_code(st), static_cast<int>(POMAI_STATUS_INVALID_ARGUMENT));
    pomai_status_free(st);
}

}  // namespace
