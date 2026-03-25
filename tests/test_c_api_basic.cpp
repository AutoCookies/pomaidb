#include "tests/common/test_main.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
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
        const int rc = std::system(cmd.c_str());
        (void)rc;
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
        const int rc = std::system(cmd.c_str());
        (void)rc;
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

POMAI_TEST(CApiMetaMembraneCrud) {
    CApiFixture fx;
    constexpr uint32_t kMetaKind = 12u; // MembraneKind::kMeta
    CAPI_EXPECT_OK(pomai_create_membrane_kind(fx.db(), "meta", 1, 1, kMetaKind));

    CAPI_EXPECT_OK(pomai_meta_put(fx.db(), "meta", "gid:7", "{\"status\":\"active\"}"));

    char* out_value = nullptr;
    size_t out_len = 0;
    CAPI_EXPECT_OK(pomai_meta_get(fx.db(), "meta", "gid:7", &out_value, &out_len));
    POMAI_EXPECT_TRUE(out_value != nullptr);
    POMAI_EXPECT_EQ(std::string(out_value, out_len), std::string("{\"status\":\"active\"}"));
    pomai_free(out_value);

    CAPI_EXPECT_OK(pomai_meta_delete(fx.db(), "meta", "gid:7"));
    pomai_status_t* st = pomai_meta_get(fx.db(), "meta", "gid:7", &out_value, &out_len);
    POMAI_EXPECT_TRUE(st != nullptr);
    pomai_status_free(st);
}

POMAI_TEST(CApiMetaMembraneRetentionTtl) {
    CApiFixture fx;
    constexpr uint32_t kMetaKind = 12u; // MembraneKind::kMeta
    CAPI_EXPECT_OK(pomai_create_membrane_kind_with_retention(
        fx.db(), "meta_ttl", 1, 1, kMetaKind, 1, 0, 0));

    CAPI_EXPECT_OK(pomai_meta_put(fx.db(), "meta_ttl", "gid:9", "{\"status\":\"hot\"}"));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    char* out_value = nullptr;
    size_t out_len = 0;
    pomai_status_t* st = pomai_meta_get(fx.db(), "meta_ttl", "gid:9", &out_value, &out_len);
    POMAI_EXPECT_TRUE(st != nullptr);
    pomai_status_free(st);
}

POMAI_TEST(CApiUpdateMembraneRetention) {
    CApiFixture fx;
    constexpr uint32_t kMetaKind = 12u; // MembraneKind::kMeta
    CAPI_EXPECT_OK(pomai_create_membrane_kind(fx.db(), "meta_upd", 1, 1, kMetaKind));
    CAPI_EXPECT_OK(pomai_update_membrane_retention(fx.db(), "meta_upd", 1, 0, 0));
    CAPI_EXPECT_OK(pomai_meta_put(fx.db(), "meta_upd", "gid:11", "{\"status\":\"warm\"}"));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    char* out_value = nullptr;
    size_t out_len = 0;
    pomai_status_t* st = pomai_meta_get(fx.db(), "meta_upd", "gid:11", &out_value, &out_len);
    POMAI_EXPECT_TRUE(st != nullptr);
    pomai_status_free(st);

    char* json = nullptr;
    size_t json_len = 0;
    CAPI_EXPECT_OK(pomai_get_membrane_retention_json(fx.db(), "meta_upd", &json, &json_len));
    POMAI_EXPECT_TRUE(json != nullptr);
    std::string s(json, json_len);
    POMAI_EXPECT_TRUE(s.find("\"ttl_sec\":1") != std::string::npos);
    pomai_free(json);
}

POMAI_TEST(CApiObjectLinkerAndGatewayLifecycle) {
    CApiFixture fx;
    CAPI_EXPECT_OK(pomai_link_objects(fx.db(), "gid:demo-1", 10, 20, 30));
    CAPI_EXPECT_OK(pomai_unlink_objects(fx.db(), "gid:demo-1"));
    CAPI_EXPECT_OK(pomai_edge_gateway_start(fx.db(), 18080, 18090));
    CAPI_EXPECT_OK(pomai_edge_gateway_stop(fx.db()));
    CAPI_EXPECT_OK(pomai_edge_gateway_start_secure(fx.db(), 18082, 18092, "demo-token"));
    CAPI_EXPECT_OK(pomai_edge_gateway_stop(fx.db()));
}


POMAI_TEST(CApiFreezePublishesToSnapshotScan) {
    CApiFixture fx;

    auto v = fx.Vec(3.0f);
    pomai_upsert_t up{};
    up.struct_size = sizeof(pomai_upsert_t);
    up.id = 778;
    up.vector = v.data();
    up.dim = 8;

    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    pomai_snapshot_t* before = nullptr;
    CAPI_EXPECT_OK(pomai_get_snapshot(fx.db(), &before));

    CAPI_EXPECT_OK(pomai_freeze(fx.db()));

    pomai_snapshot_t* after = nullptr;
    CAPI_EXPECT_OK(pomai_get_snapshot(fx.db(), &after));

    auto scan_has_id = [&](pomai_snapshot_t* snap, uint64_t id) {
        pomai_scan_options_t scan_opts{};
        pomai_scan_options_init(&scan_opts);
        pomai_iter_t* it = nullptr;
        CAPI_EXPECT_OK(pomai_scan(fx.db(), &scan_opts, snap, &it));

        bool found = false;
        while (pomai_iter_valid(it)) {
            pomai_record_view_t view{};
            view.struct_size = sizeof(pomai_record_view_t);
            CAPI_EXPECT_OK(pomai_iter_get_record(it, &view));
            if (view.id == id) {
                found = true;
            }
            pomai_iter_next(it);
        }
        CAPI_EXPECT_OK(pomai_iter_status(it));
        pomai_iter_free(it);
        return found;
    };

    POMAI_EXPECT_TRUE(!scan_has_id(before, 778));
    POMAI_EXPECT_TRUE(scan_has_id(after, 778));

    pomai_snapshot_free(before);
    pomai_snapshot_free(after);
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

POMAI_TEST(CApiMembraneScanVector) {
    CApiFixture fx;
    const auto v = fx.Vec(3.0f);
    pomai_upsert_t up{};
    up.struct_size = sizeof(pomai_upsert_t);
    up.id = 7;
    up.vector = v.data();
    up.dim = 8;
    CAPI_EXPECT_OK(pomai_put(fx.db(), &up));

    pomai_membrane_scan_options_t mopts{};
    pomai_membrane_scan_options_init(&mopts);
    mopts.max_records = 100;

    pomai_membrane_iter_t* mit = nullptr;
    CAPI_EXPECT_OK(pomai_membrane_scan(fx.db(), "__default__", &mopts, &mit));

    bool saw7 = false;
    while (pomai_membrane_iter_valid(mit)) {
        pomai_membrane_record_view_t view{};
        view.struct_size = sizeof(view);
        CAPI_EXPECT_OK(pomai_membrane_iter_get_record(mit, &view));
        POMAI_EXPECT_EQ(view.membrane_kind, 0u);
        if (view.id == 7u) {
            saw7 = true;
            POMAI_EXPECT_EQ(view.vector_dim, 8u);
        }
        pomai_membrane_iter_next(mit);
    }
    POMAI_EXPECT_TRUE(saw7);
    CAPI_EXPECT_OK(pomai_membrane_iter_status(mit));
    pomai_membrane_iter_free(mit);
}

POMAI_TEST(CApiMembraneKindCapabilities) {
    pomai_membrane_capabilities_t caps{};
    pomai_membrane_capabilities_init(&caps);
    caps.struct_size = sizeof(caps);
    CAPI_EXPECT_OK(pomai_membrane_kind_capabilities(POMAI_MEMBRANE_KIND_VECTOR, &caps));
    POMAI_EXPECT_EQ(static_cast<unsigned>(caps.kind), static_cast<unsigned>(POMAI_MEMBRANE_KIND_VECTOR));
    POMAI_EXPECT_TRUE(caps.snapshot_isolated_scan);
    CAPI_EXPECT_OK(pomai_membrane_kind_capabilities(POMAI_MEMBRANE_KIND_SPATIAL, &caps));
    POMAI_EXPECT_EQ(caps.stability, POMAI_MEMBRANE_STABILITY_EXPERIMENTAL);
    pomai_status_t* bad = pomai_membrane_kind_capabilities(200, &caps);
    POMAI_EXPECT_TRUE(bad != nullptr);
    pomai_status_free(bad);
}

}  // namespace
