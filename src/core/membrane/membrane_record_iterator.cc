#include "core/membrane/manager.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/blob/blob_engine.h"
#include "core/bitset/bitset_engine.h"
#include "core/graph/graph_membrane_impl.h"
#include "core/keyvalue/keyvalue_engine.h"
#include "core/mesh/mesh_engine.h"
#include "core/rag/rag_engine.h"
#include "core/sketch/sketch_engine.h"
#include "core/sparse/sparse_engine.h"
#include "core/spatial/spatial_engine.h"
#include "core/text/text_membrane.h"
#include "core/timeseries/timeseries_engine.h"
#include "core/vector_engine/vector_engine.h"
#include "pomai/iterator.h"
#include "pomai/membrane_iterator.h"

namespace pomai::core {

namespace {

using pomai::MembraneRecord;
using pomai::MembraneRecordIterator;
using pomai::MembraneScanOptions;
using pomai::Status;

MembraneRecord kEmptyMembraneRecord{};

static bool ScanDeadlineExceeded(const std::chrono::steady_clock::time_point& t0, uint32_t deadline_ms) {
    if (deadline_ms == 0) return false;
    const auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count() >= static_cast<int64_t>(deadline_ms);
}

static void TruncateField(std::string* s, size_t max_bytes) {
    if (!s || max_bytes == 0 || s->size() <= max_bytes) return;
    s->resize(max_bytes);
    *s += "...";
}

static std::string PreviewText(const std::string& t, std::size_t max_n) {
    std::string o;
    for (char c : t) {
        if (o.size() >= max_n) {
            o += "...";
            break;
        }
        if (c == '\n' || c == '\r' || c == '\t') {
            o += ' ';
        } else {
            o.push_back(c);
        }
    }
    return o;
}

static uint64_t HashKey(std::string_view k) {
    return static_cast<uint64_t>(std::hash<std::string>{}(std::string(k)));
}

static uint64_t EffectiveMaxRecords(const MembraneScanOptions& o) {
    return o.max_records == 0 ? UINT64_MAX : o.max_records;
}

class VectorMembraneScanIterator final : public MembraneRecordIterator {
    std::unique_ptr<pomai::SnapshotIterator> it_;
    MembraneScanOptions opts_;
    MembraneRecord rec_{};
    uint64_t row_index_ = 0;
    bool truncated_ = false;
    Status st_ = Status::Ok();
    std::chrono::steady_clock::time_point t0_;
    uint32_t vec_deadline_check_counter_ = 0;

    void Refresh() {
        rec_.vector.clear();
        rec_.key.clear();
        rec_.value.clear();
        rec_.id = 0;
        if (!it_ || !it_->Valid()) return;
        rec_.kind = pomai::MembraneKind::kVector;
        rec_.id = it_->id();
        const auto sp = it_->vector();
        rec_.vector.assign(sp.begin(), sp.end());
        rec_.value = "dim=" + std::to_string(rec_.vector.size());
    }

public:
    VectorMembraneScanIterator(std::unique_ptr<pomai::SnapshotIterator> it, MembraneScanOptions opts)
        : it_(std::move(it)), opts_(opts), t0_(std::chrono::steady_clock::now()) {
        rec_.kind = pomai::MembraneKind::kVector;
        Refresh();
    }
    bool Valid() const override { return st_.ok() && it_ && it_->Valid(); }
    void Next() override {
        if (!st_.ok() || !it_ || !it_->Valid()) return;
        if (opts_.deadline_ms) {
            if (++vec_deadline_check_counter_ >= 4096) {
                vec_deadline_check_counter_ = 0;
                if (ScanDeadlineExceeded(t0_, opts_.deadline_ms)) {
                    st_ = Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
                    it_.reset();
                    return;
                }
            }
        }
        if (opts_.max_records > 0 && row_index_ + 1 >= opts_.max_records) {
            const bool more = it_->Next();
            truncated_ = more && it_->Valid();
            it_.reset();
            return;
        }
        if (!it_->Next()) {
            it_.reset();
            return;
        }
        ++row_index_;
        Refresh();
        if (opts_.deadline_ms && ScanDeadlineExceeded(t0_, opts_.deadline_ms)) {
            st_ = Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
            it_.reset();
        }
    }
    const MembraneRecord& Record() const override {
        if (Valid()) return rec_;
        return kEmptyMembraneRecord;
    }
    Status ScanStatus() const override { return st_; }
    bool Truncated() const override { return truncated_; }
};

class ListMembraneIterator final : public MembraneRecordIterator {
    std::vector<MembraneRecord> rows_;
    std::size_t i_ = 0;
    bool truncated_ = false;
    Status st_ = Status::Ok();

public:
    ListMembraneIterator(std::vector<MembraneRecord> rows, bool truncated, Status st = Status::Ok())
        : rows_(std::move(rows)), truncated_(truncated), st_(st) {}
    bool Valid() const override { return st_.ok() && i_ < rows_.size(); }
    void Next() override {
        if (i_ < rows_.size()) ++i_;
    }
    const MembraneRecord& Record() const override {
        if (Valid()) return rows_[i_];
        return kEmptyMembraneRecord;
    }
    Status ScanStatus() const override { return st_; }
    bool Truncated() const override { return truncated_; }
};

class RagLazyIterator final : public MembraneRecordIterator {
    RagEngine* eng_;
    std::vector<pomai::ChunkId> ids_;
    std::size_t i_ = 0;
    MembraneScanOptions opts_;
    MembraneRecord rec_{};
    bool truncated_ = false;
    Status st_ = Status::Ok();

    void LoadCurrent() {
        rec_ = MembraneRecord{};
        rec_.kind = pomai::MembraneKind::kRag;
        if (i_ >= ids_.size()) return;
        pomai::DocId did = 0;
        std::string text;
        std::size_t ntok = 0;
        bool hv = false;
        if (!eng_->TryGetChunkExport(ids_[i_], &did, &text, &ntok, &hv)) {
            return;
        }
        rec_.id = ids_[i_];
        rec_.key = "doc_id=" + std::to_string(did);
        std::ostringstream oss;
        oss << "tokens=" << ntok << " embedding=" << (hv ? "yes" : "no") << " text=" << PreviewText(text, 480);
        rec_.value = std::move(oss).str();
        TruncateField(&rec_.key, opts_.max_field_bytes);
        TruncateField(&rec_.value, opts_.max_field_bytes);
    }

public:
    RagLazyIterator(RagEngine* eng, std::vector<pomai::ChunkId> ids, MembraneScanOptions opts, bool truncated, Status st)
        : eng_(eng), ids_(std::move(ids)), opts_(opts), truncated_(truncated), st_(st) {
        rec_.kind = pomai::MembraneKind::kRag;
        if (st_.ok() && !ids_.empty()) LoadCurrent();
    }
    bool Valid() const override { return st_.ok() && i_ < ids_.size(); }
    void Next() override {
        if (!st_.ok() || i_ >= ids_.size()) return;
        ++i_;
        if (i_ < ids_.size()) LoadCurrent();
    }
    const MembraneRecord& Record() const override {
        if (Valid()) return rec_;
        return kEmptyMembraneRecord;
    }
    Status ScanStatus() const override { return st_; }
    bool Truncated() const override { return truncated_; }
};

class TextLazyIterator final : public MembraneRecordIterator {
    TextMembrane* eng_;
    std::vector<pomai::VectorId> ids_;
    std::size_t i_ = 0;
    MembraneScanOptions opts_;
    MembraneRecord rec_{};
    bool truncated_ = false;
    Status st_ = Status::Ok();

    void LoadCurrent() {
        rec_ = MembraneRecord{};
        rec_.kind = pomai::MembraneKind::kText;
        if (i_ >= ids_.size()) return;
        std::string t;
        if (!eng_->Get(ids_[i_], &t).ok()) return;
        rec_.id = ids_[i_];
        rec_.value = PreviewText(t, 480);
        TruncateField(&rec_.value, opts_.max_field_bytes);
    }

public:
    TextLazyIterator(TextMembrane* eng, std::vector<pomai::VectorId> ids, MembraneScanOptions opts, bool truncated, Status st)
        : eng_(eng), ids_(std::move(ids)), opts_(opts), truncated_(truncated), st_(st) {
        rec_.kind = pomai::MembraneKind::kText;
        if (st_.ok() && !ids_.empty()) LoadCurrent();
    }
    bool Valid() const override { return st_.ok() && i_ < ids_.size(); }
    void Next() override {
        if (!st_.ok() || i_ >= ids_.size()) return;
        ++i_;
        if (i_ < ids_.size()) LoadCurrent();
    }
    const MembraneRecord& Record() const override {
        if (Valid()) return rec_;
        return kEmptyMembraneRecord;
    }
    Status ScanStatus() const override { return st_; }
    bool Truncated() const override { return truncated_; }
};

class KvLazyIterator final : public MembraneRecordIterator {
    KeyValueEngine* eng_;
    pomai::MembraneKind kind_;
    std::vector<std::string> keys_;
    std::size_t i_ = 0;
    MembraneScanOptions opts_;
    MembraneRecord rec_{};
    bool truncated_ = false;
    Status st_ = Status::Ok();

    void LoadCurrent() {
        rec_ = MembraneRecord{};
        rec_.kind = kind_;
        if (i_ >= keys_.size()) return;
        std::string v;
        if (!eng_->Get(keys_[i_], &v).ok()) {
            rec_.id = HashKey(keys_[i_]);
            rec_.key = keys_[i_];
            rec_.value = "";
            return;
        }
        rec_.id = HashKey(keys_[i_]);
        rec_.key = keys_[i_];
        rec_.value = std::move(v);
        TruncateField(&rec_.key, opts_.max_field_bytes);
        TruncateField(&rec_.value, opts_.max_field_bytes);
    }

public:
    KvLazyIterator(KeyValueEngine* eng, pomai::MembraneKind k, std::vector<std::string> keys, MembraneScanOptions opts,
                   bool truncated, Status st)
        : eng_(eng), kind_(k), keys_(std::move(keys)), opts_(opts), truncated_(truncated), st_(st) {
        if (st_.ok() && !keys_.empty()) LoadCurrent();
    }
    bool Valid() const override { return st_.ok() && i_ < keys_.size(); }
    void Next() override {
        if (!st_.ok() || i_ >= keys_.size()) return;
        ++i_;
        if (i_ < keys_.size()) LoadCurrent();
    }
    const MembraneRecord& Record() const override {
        if (Valid()) return rec_;
        return kEmptyMembraneRecord;
    }
    Status ScanStatus() const override { return st_; }
    bool Truncated() const override { return truncated_; }
};

} // namespace

Status MembraneManager::NewMembraneRecordIterator(std::string_view membrane,
                                                  std::unique_ptr<MembraneRecordIterator>* out) {
    return NewMembraneRecordIterator(membrane, MembraneScanOptions{}, out);
}

Status MembraneManager::NewMembraneRecordIterator(std::string_view membrane, const MembraneScanOptions& scan_opts,
                                                  std::unique_ptr<MembraneRecordIterator>* out) {
    if (!out) return Status::InvalidArgument("out is null");
    out->reset();
    auto* state = GetMembraneOrNull(membrane);
    if (!state) return Status::NotFound("membrane not found");
    PollMaintenance();

    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t cap = EffectiveMaxRecords(scan_opts);

    switch (state->spec.kind) {
    case pomai::MembraneKind::kVector: {
        if (!state->vector_engine) return Status::InvalidArgument("vector engine missing");
        std::unique_ptr<pomai::SnapshotIterator> it;
        const auto st = state->vector_engine->NewIterator(&it);
        if (!st.ok()) return st;
        *out = std::make_unique<VectorMembraneScanIterator>(std::move(it), scan_opts);
        return Status::Ok();
    }
    case pomai::MembraneKind::kRag: {
        if (!state->rag_engine) return Status::InvalidArgument("rag engine missing");
        std::vector<pomai::ChunkId> ids;
        ids.reserve(256);
        bool key_overflow = false;
        bool deadline_hit = false;
        state->rag_engine->ForEachChunk([&](pomai::ChunkId cid, pomai::DocId, const std::string&, std::size_t, bool) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
                deadline_hit = true;
                return;
            }
            if (scan_opts.max_materialized_keys > 0 && ids.size() >= scan_opts.max_materialized_keys) {
                key_overflow = true;
                return;
            }
            ids.push_back(cid);
        });
        if (deadline_hit || ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        if (key_overflow) {
            return Status(pomai::ErrorCode::kResourceExhausted, "membrane scan chunk id materialization cap exceeded");
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        const bool truncated = (scan_opts.max_records > 0 && ids.size() > cap);
        if (truncated) ids.resize(static_cast<std::size_t>(cap));
        *out = std::make_unique<RagLazyIterator>(state->rag_engine.get(), std::move(ids), scan_opts, truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kMeta:
    case pomai::MembraneKind::kKeyValue: {
        KeyValueEngine* eng =
            (state->spec.kind == pomai::MembraneKind::kMeta) ? state->meta_engine.get() : state->keyvalue_engine.get();
        if (!eng) return Status::InvalidArgument("kv/meta engine missing");
        std::vector<std::string> keys;
        keys.reserve(256);
        bool key_overflow = false;
        bool deadline_hit = false;
        eng->ForEach([&](std::string_view k, std::string_view) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
                deadline_hit = true;
                return;
            }
            if (scan_opts.max_materialized_keys > 0 && keys.size() >= scan_opts.max_materialized_keys) {
                key_overflow = true;
                return;
            }
            keys.emplace_back(k);
        });
        if (deadline_hit || ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        if (key_overflow) {
            return Status(pomai::ErrorCode::kResourceExhausted, "membrane scan key materialization cap exceeded");
        }
        std::sort(keys.begin(), keys.end());
        const bool truncated = (scan_opts.max_records > 0 && keys.size() > cap);
        if (truncated) keys.resize(static_cast<std::size_t>(cap));
        *out = std::make_unique<KvLazyIterator>(eng, state->spec.kind, std::move(keys), scan_opts, truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kText: {
        if (!state->text_engine) return Status::InvalidArgument("text engine missing");
        std::vector<pomai::VectorId> ids;
        ids.reserve(256);
        bool key_overflow = false;
        bool deadline_hit = false;
        state->text_engine->ForEach([&](pomai::VectorId id, std::string_view) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
                deadline_hit = true;
                return;
            }
            if (scan_opts.max_materialized_keys > 0 && ids.size() >= scan_opts.max_materialized_keys) {
                key_overflow = true;
                return;
            }
            ids.push_back(id);
        });
        if (deadline_hit || ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        if (key_overflow) {
            return Status(pomai::ErrorCode::kResourceExhausted, "membrane scan doc id materialization cap exceeded");
        }
        std::sort(ids.begin(), ids.end());
        const bool truncated = (scan_opts.max_records > 0 && ids.size() > cap);
        if (truncated) ids.resize(static_cast<std::size_t>(cap));
        *out = std::make_unique<TextLazyIterator>(state->text_engine.get(), std::move(ids), scan_opts, truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kSketch: {
        if (!state->sketch_engine) return Status::InvalidArgument("sketch engine missing");
        std::vector<std::string> keys;
        keys.reserve(256);
        bool key_overflow = false;
        bool deadline_hit = false;
        state->sketch_engine->ForEach([&](std::string_view k, std::uint64_t) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
                deadline_hit = true;
                return;
            }
            if (scan_opts.max_materialized_keys > 0 && keys.size() >= scan_opts.max_materialized_keys) {
                key_overflow = true;
                return;
            }
            keys.emplace_back(k);
        });
        if (deadline_hit || ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        if (key_overflow) {
            return Status(pomai::ErrorCode::kResourceExhausted, "membrane scan key materialization cap exceeded");
        }
        std::sort(keys.begin(), keys.end());
        const bool truncated = (scan_opts.max_records > 0 && keys.size() > cap);
        const std::size_t n = truncated ? static_cast<std::size_t>(cap) : keys.size();
        std::vector<MembraneRecord> rows;
        rows.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
                return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
            }
            uint64_t c = 0;
            (void)state->sketch_engine->Estimate(keys[j], &c);
            MembraneRecord r;
            r.kind = pomai::MembraneKind::kSketch;
            r.id = HashKey(keys[j]);
            r.key = keys[j];
            r.value = "count=" + std::to_string(c);
            TruncateField(&r.key, scan_opts.max_field_bytes);
            TruncateField(&r.value, scan_opts.max_field_bytes);
            rows.push_back(std::move(r));
        }
        *out = std::make_unique<ListMembraneIterator>(std::move(rows), truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kTimeSeries: {
        if (!state->timeseries_engine) return Status::InvalidArgument("timeseries engine missing");
        std::vector<MembraneRecord> rows;
        bool truncated = false;
        state->timeseries_engine->ForEach([&](std::uint64_t sid, const pomai::TimeSeriesPoint& p) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) return;
            if (scan_opts.max_records > 0 && rows.size() >= cap) {
                truncated = true;
                return;
            }
            MembraneRecord r;
            r.kind = pomai::MembraneKind::kTimeSeries;
            r.id = p.timestamp;
            r.key = "series_id=" + std::to_string(sid);
            r.value = "ts=" + std::to_string(p.timestamp) + " value=" + std::to_string(p.value);
            TruncateField(&r.key, scan_opts.max_field_bytes);
            TruncateField(&r.value, scan_opts.max_field_bytes);
            rows.push_back(std::move(r));
        });
        if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            if (a.key != b.key) return a.key < b.key;
            return a.id < b.id;
        });
        *out = std::make_unique<ListMembraneIterator>(std::move(rows), truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kBlob: {
        if (!state->blob_engine) return Status::InvalidArgument("blob engine missing");
        std::vector<std::pair<uint64_t, std::size_t>> items;
        items.reserve(256);
        bool id_overflow = false;
        bool deadline_hit = false;
        state->blob_engine->ForEach([&](uint64_t id, std::size_t nbytes) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
                deadline_hit = true;
                return;
            }
            if (scan_opts.max_materialized_keys > 0 && items.size() >= scan_opts.max_materialized_keys) {
                id_overflow = true;
                return;
            }
            items.emplace_back(id, nbytes);
        });
        if (deadline_hit || ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        if (id_overflow) {
            return Status(pomai::ErrorCode::kResourceExhausted, "membrane scan id materialization cap exceeded");
        }
        std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const bool truncated = (scan_opts.max_records > 0 && items.size() > cap);
        const std::size_t n = truncated ? static_cast<std::size_t>(cap) : items.size();
        std::vector<MembraneRecord> rows;
        rows.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            MembraneRecord r;
            r.kind = pomai::MembraneKind::kBlob;
            r.id = items[j].first;
            r.value = "bytes=" + std::to_string(items[j].second);
            rows.push_back(std::move(r));
        }
        *out = std::make_unique<ListMembraneIterator>(std::move(rows), truncated, Status::Ok());
        return Status::Ok();
    }
    default:
        break;
    }

    switch (state->spec.kind) {
    case pomai::MembraneKind::kSpatial: {
        if (!state->spatial_engine) return Status::InvalidArgument("spatial engine missing");
        std::vector<MembraneRecord> rows;
        bool truncated = false;
        state->spatial_engine->ForEach([&](std::uint64_t id, double lat, double lon) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) return;
            if (scan_opts.max_records > 0 && rows.size() >= cap) {
                truncated = true;
                return;
            }
            MembraneRecord r;
            r.kind = pomai::MembraneKind::kSpatial;
            r.id = id;
            r.value = "lat=" + std::to_string(lat) + " lon=" + std::to_string(lon);
            TruncateField(&r.value, scan_opts.max_field_bytes);
            rows.push_back(std::move(r));
        });
        if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        *out = std::make_unique<ListMembraneIterator>(std::move(rows), truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kMesh: {
        if (!state->mesh_engine) return Status::InvalidArgument("mesh engine missing");
        std::vector<MembraneRecord> rows;
        bool truncated = false;
        state->mesh_engine->ForEach([&](std::uint64_t id, std::size_t nfloats) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) return;
            if (scan_opts.max_records > 0 && rows.size() >= cap) {
                truncated = true;
                return;
            }
            MembraneRecord r;
            r.kind = pomai::MembraneKind::kMesh;
            r.id = id;
            r.value = "base_xyz_floats=" + std::to_string(nfloats);
            rows.push_back(std::move(r));
        });
        if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        *out = std::make_unique<ListMembraneIterator>(std::move(rows), truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kSparse: {
        if (!state->sparse_engine) return Status::InvalidArgument("sparse engine missing");
        std::vector<MembraneRecord> rows;
        bool truncated = false;
        state->sparse_engine->ForEach([&](std::uint64_t id, std::size_t nnz) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) return;
            if (scan_opts.max_records > 0 && rows.size() >= cap) {
                truncated = true;
                return;
            }
            MembraneRecord r;
            r.kind = pomai::MembraneKind::kSparse;
            r.id = id;
            r.value = "nnz=" + std::to_string(nnz);
            rows.push_back(std::move(r));
        });
        if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        *out = std::make_unique<ListMembraneIterator>(std::move(rows), truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kBitset: {
        if (!state->bitset_engine) return Status::InvalidArgument("bitset engine missing");
        std::vector<MembraneRecord> rows;
        bool truncated = false;
        state->bitset_engine->ForEach([&](std::uint64_t id, std::size_t nbytes) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) return;
            if (scan_opts.max_records > 0 && rows.size() >= cap) {
                truncated = true;
                return;
            }
            MembraneRecord r;
            r.kind = pomai::MembraneKind::kBitset;
            r.id = id;
            r.value = "bytes=" + std::to_string(nbytes);
            rows.push_back(std::move(r));
        });
        if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        *out = std::make_unique<ListMembraneIterator>(std::move(rows), truncated, Status::Ok());
        return Status::Ok();
    }
    case pomai::MembraneKind::kGraph: {
        auto* g = dynamic_cast<GraphMembraneImpl*>(state->graph_engine.get());
        if (!g) return Status::InvalidArgument("graph engine implementation unavailable");
        std::vector<MembraneRecord> rows;
        bool truncated = false;
        g->ForEachVertex([&](pomai::VertexId vid, std::size_t deg) {
            if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) return;
            if (scan_opts.max_records > 0 && rows.size() >= cap) {
                truncated = true;
                return;
            }
            MembraneRecord r;
            r.kind = pomai::MembraneKind::kGraph;
            r.id = vid;
            r.value = "out_degree=" + std::to_string(deg);
            rows.push_back(std::move(r));
        });
        if (ScanDeadlineExceeded(t0, scan_opts.deadline_ms)) {
            return Status(pomai::ErrorCode::kAborted, "membrane scan deadline exceeded");
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
        *out = std::make_unique<ListMembraneIterator>(std::move(rows), truncated, Status::Ok());
        return Status::Ok();
    }
    default:
        *out = std::make_unique<ListMembraneIterator>(std::vector<MembraneRecord>{}, false, Status::Ok());
        return Status::Ok();
    }
}

} // namespace pomai::core
