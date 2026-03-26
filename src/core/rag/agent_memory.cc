#include "pomai/agent_memory.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>

#include "pomai/metadata.h"

namespace pomai
{

namespace {

// Simple escaping: '\' -> '\\', '|' -> '\|'.
std::string EscapeField(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
    {
        if (c == '\\' || c == '|')
        {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

bool UnescapeField(std::string_view s, std::string* out)
{
    out->clear();
    out->reserve(s.size());
    bool escape = false;
    for (char c : s)
    {
        if (!escape && c == '\\')
        {
            escape = true;
            continue;
        }
        out->push_back(c);
        escape = false;
    }
    if (escape)
    {
        // Trailing backslash – treat as malformed.
        return false;
    }
    return true;
}

// Parse a metadata payload of the form:
//   "am|key=value|key2=value2|..."
// with '|' and '\' escaped inside values.
bool ParseEncodedMetadata(std::string_view encoded,
                          AgentMemoryRecord* out_record)
{
    if (!out_record)
        return false;
    if (!encoded.starts_with("am|"))
        return false;

    AgentMemoryRecord rec;

    std::size_t pos = 3; // skip "am|"
    std::string field_buf;
    std::string key;
    std::string value;

    auto flush_field = [&](std::string_view field) -> bool {
        if (field.empty())
        {
            return true;
        }
        auto eq = field.find('=');
        if (eq == std::string_view::npos)
        {
            return false;
        }
        std::string_view k = field.substr(0, eq);
        std::string_view v = field.substr(eq + 1);
        if (!UnescapeField(k, &key))
            return false;
        if (!UnescapeField(v, &value))
            return false;

        if (key == "agent")
        {
            rec.agent_id = value;
        }
        else if (key == "session")
        {
            rec.session_id = value;
        }
        else if (key == "kind")
        {
            if (value == "message")
            {
                rec.kind = AgentMemoryKind::kMessage;
            }
            else if (value == "summary")
            {
                rec.kind = AgentMemoryKind::kSummary;
            }
            else if (value == "knowledge")
            {
                rec.kind = AgentMemoryKind::kKnowledge;
            }
        }
        else if (key == "ts")
        {
            std::int64_t ts = 0;
            auto res = std::from_chars(value.data(), value.data() + value.size(), ts);
            if (res.ec == std::errc())
            {
                rec.logical_ts = ts;
            }
        }
        else if (key == "text")
        {
            rec.text = value;
        }
        return true;
    };

    bool escape = false;
    field_buf.clear();
    for (; pos < encoded.size(); ++pos)
    {
        char c = encoded[pos];
        if (!escape && c == '\\')
        {
            escape = true;
            continue;
        }
        if (!escape && c == '|')
        {
            if (!flush_field(std::string_view(field_buf))
                )
            {
                return false;
            }
            field_buf.clear();
            continue;
        }
        field_buf.push_back(c);
        escape = false;
    }
    if (!field_buf.empty())
    {
        if (!flush_field(std::string_view(field_buf)))
        {
            return false;
        }
    }

    *out_record = std::move(rec);
    return true;
}

std::string EncodeKind(AgentMemoryKind kind)
{
    switch (kind)
    {
    case AgentMemoryKind::kMessage:
        return "message";
    case AgentMemoryKind::kSummary:
        return "summary";
    case AgentMemoryKind::kKnowledge:
        return "knowledge";
    }
    return "message";
}

float ComputeFallbackScore(std::span<const float> query,
                           std::span<const float> candidate,
                           MetricType metric)
{
    if (query.size() != candidate.size())
    {
        return -std::numeric_limits<float>::infinity();
    }

    float dot = 0.0f;
    float l2 = 0.0f;
    float qn = 0.0f;
    float cn = 0.0f;
    for (std::size_t i = 0; i < query.size(); ++i)
    {
        const float qv = query[i];
        const float cv = candidate[i];
        dot += qv * cv;
        const float d = qv - cv;
        l2 += d * d;
        qn += qv * qv;
        cn += cv * cv;
    }

    switch (metric)
    {
    case MetricType::kInnerProduct:
        return dot;
    case MetricType::kCosine:
    {
        const float denom = std::sqrt(qn) * std::sqrt(cn);
        if (denom <= 0.0f)
        {
            return -std::numeric_limits<float>::infinity();
        }
        return dot / denom;
    }
    case MetricType::kL2:
    default:
        // CTest ordering expects larger score == better hit.
        return -l2;
    }
}

} // namespace

AgentMemory::AgentMemory(AgentMemoryOptions opts, std::unique_ptr<Database> db)
    : options_(std::move(opts)), db_(std::move(db))
{
    dim_ = options_.dim;
    opened_ = (db_ != nullptr);
}

AgentMemory::~AgentMemory() = default;

Status AgentMemory::Open(const AgentMemoryOptions& options,
                         std::unique_ptr<AgentMemory>* out)
{
    if (!out)
        return Status::InvalidArgument("out is null");
    if (options.path.empty())
        return Status::InvalidArgument("AgentMemoryOptions.path is empty");
    if (options.dim == 0)
        return Status::InvalidArgument("AgentMemoryOptions.dim must be > 0");

    EmbeddedOptions emb;
    emb.path = options.path;
    emb.dim = options.dim;
    emb.metric = options.metric;
    emb.fsync = FsyncPolicy::kNever;
    emb.max_memtable_mb = options.max_memtable_mb;
    emb.memtable_flush_threshold_mb = options.memtable_flush_threshold_mb;
    emb.auto_freeze_on_pressure = options.auto_freeze_on_pressure;

    auto db = std::make_unique<Database>();
    auto st = db->Open(emb);
    if (!st.ok())
        return st;

    auto ptr = std::make_unique<AgentMemory>(options, std::move(db));

    {
        // Initialize next_id_ by scanning for the max existing id.
        std::shared_ptr<Snapshot> snap;
        st = ptr->db_->GetSnapshot(&snap);
        if (!st.ok())
        {
            return st;
        }
        std::unique_ptr<SnapshotIterator> it;
        st = ptr->db_->NewIterator(snap, &it);
        if (!st.ok())
        {
            return st;
        }
        VectorId max_id = 0;
        while (it->Next())
        {
            if (it->id() > max_id)
            {
                max_id = it->id();
            }
        }
        ptr->next_id_ = max_id + 1;
    }

    *out = std::move(ptr);
    return Status::Ok();
}

Status AgentMemory::EnsureOpen() const
{
    if (!opened_ || !db_ || !db_->IsOpen())
    {
        return Status(ErrorCode::kFailedPrecondition, "AgentMemory is not open");
    }
    return Status::Ok();
}

std::string AgentMemory::EncodeMetadata(const AgentMemoryRecord& record)
{
    // Compact, delimiter-based encoding specialized for AgentMemory.
    // Format: "am|agent=...|session=...|kind=...|ts=...|text=..."
    std::string out = "am|";
    out.append("agent=");
    out.append(EscapeField(record.agent_id));
    out.push_back('|');
    out.append("session=");
    out.append(EscapeField(record.session_id));
    out.push_back('|');
    out.append("kind=");
    out.append(EncodeKind(record.kind));
    out.push_back('|');
    out.append("ts=");
    out.append(std::to_string(static_cast<long long>(record.logical_ts)));
    out.push_back('|');
    out.append("text=");
    out.append(EscapeField(record.text));
    return out;
}

bool AgentMemory::DecodeMetadata(const std::string& encoded,
                                 AgentMemoryRecord* out_record)
{
    return ParseEncodedMetadata(encoded, out_record);
}

Status AgentMemory::AppendMessage(const AgentMemoryRecord& record,
                                  VectorId* out_id)
{
    auto st = EnsureOpen();
    if (!st.ok())
        return st;
    if (record.embedding.size() != dim_)
    {
        return Status::InvalidArgument("embedding dimension mismatch");
    }
    const VectorId id = next_id_++;

    const std::string encoded = EncodeMetadata(record);
    Metadata meta(encoded);

    st = db_->AddVector(id, std::span<const float>(record.embedding.data(), record.embedding.size()), meta);
    if (!st.ok())
        return st;

    // Best-effort backpressure coordination.
    (void)db_->TryFreezeIfPressured();

    if (out_id)
    {
        *out_id = id;
    }

    // Lazy per-agent pruning based on soft cap.
    if (options_.max_messages_per_agent > 0)
    {
        (void)PruneOld(record.agent_id,
                       options_.max_messages_per_agent,
                       std::numeric_limits<std::int64_t>::min());
    }
    if (options_.max_device_bytes > 0)
    {
        (void)PruneDeviceWide(options_.max_device_bytes);
    }

    return Status::Ok();
}

Status AgentMemory::AppendBatch(const std::vector<AgentMemoryRecord>& records,
                                std::vector<VectorId>* out_ids)
{
    auto st = EnsureOpen();
    if (!st.ok())
        return st;
    if (records.empty())
    {
        if (out_ids)
            out_ids->clear();
        return Status::Ok();
    }

    std::vector<VectorId> ids;
    std::vector<std::span<const float>> vecs;
    ids.reserve(records.size());
    vecs.reserve(records.size());

    for (const auto& r : records)
    {
        if (r.embedding.size() != dim_)
        {
            return Status::InvalidArgument("embedding dimension mismatch in batch");
        }
    }

    for (std::size_t i = 0; i < records.size(); ++i)
    {
        VectorId id = next_id_++;
        ids.push_back(id);
        vecs.emplace_back(records[i].embedding.data(), records[i].embedding.size());

        const std::string encoded = EncodeMetadata(records[i]);
        Metadata meta(encoded);
        // Use single AddVector path for now so metadata is persisted per-record.
        // We don't have a metadata-aware AddVectorBatch in the embedded API yet,
        // so we fall back to individual calls here.
        st = db_->AddVector(id, vecs.back(), meta);
        if (!st.ok())
            return st;
    }

    if (out_ids)
    {
        *out_ids = std::move(ids);
    }

    (void)db_->TryFreezeIfPressured();

    // Soft caps checked against the last record's agent / device totals.
    if (!records.empty() && options_.max_messages_per_agent > 0)
    {
        (void)PruneOld(records.back().agent_id,
                       options_.max_messages_per_agent,
                       std::numeric_limits<std::int64_t>::min());
    }
    if (options_.max_device_bytes > 0)
    {
        (void)PruneDeviceWide(options_.max_device_bytes);
    }

    return Status::Ok();
}

Status AgentMemory::GetRecent(std::string_view agent_id,
                              std::string_view session_id,
                              std::size_t limit,
                              std::vector<AgentMemoryRecord>* out)
{
    if (!out)
        return Status::InvalidArgument("out is null");
    auto st = EnsureOpen();
    if (!st.ok())
        return st;

    std::shared_ptr<Snapshot> snap;
    st = db_->GetSnapshot(&snap);
    if (!st.ok())
        return st;
    std::unique_ptr<SnapshotIterator> it;
    st = db_->NewIterator(snap, &it);
    if (!st.ok())
        return st;

    std::vector<AgentMemoryRecord> all;
    std::vector<float> tmp_vec;
    Metadata meta;

    while (it->Next())
    {
        const VectorId id = it->id();
        tmp_vec.clear();
        meta = Metadata();
        auto gst = db_->Get(id, &tmp_vec, &meta);
        if (!gst.ok())
        {
            continue;
        }
        AgentMemoryRecord rec;
        if (!DecodeMetadata(meta.tenant, &rec))
        {
            continue;
        }
        if (rec.agent_id != agent_id)
        {
            continue;
        }
        if (!session_id.empty() && rec.session_id != session_id)
        {
            continue;
        }
        all.push_back(std::move(rec));
    }

    std::sort(all.begin(), all.end(),
              [](const AgentMemoryRecord& a, const AgentMemoryRecord& b) {
                  if (a.logical_ts != b.logical_ts)
                      return a.logical_ts < b.logical_ts;
                  return a.text < b.text;
              });

    // Temporary debug: observe how many records are returned for tests.
    // This will be removed once AgentMemory behavior is validated.
    std::cerr << "[AgentMemory::GetRecent] agent=" << agent_id
              << " session=" << session_id
              << " total=" << all.size()
              << " limit=" << limit << std::endl;

    if (all.size() > limit)
    {
        out->assign(all.end() - static_cast<std::ptrdiff_t>(limit), all.end());
    }
    else
    {
        *out = std::move(all);
    }
    return Status::Ok();
}

Status AgentMemory::SearchAndFilter(const AgentMemoryQuery& query,
                                          AgentMemorySearchResult* out)
{
    if (!out)
        return Status::InvalidArgument("out is null");
    if (query.embedding.size() != dim_)
    {
        return Status::InvalidArgument("query.embedding dimension mismatch");
    }

    // Over-fetch to compensate for post-filtering by agent/session/kind/time.
    std::uint32_t internal_topk = query.topk;
    if (internal_topk == 0)
    {
        internal_topk = 1;
    }
    internal_topk = std::min<std::uint32_t>(internal_topk * 4, 1024);

    SearchResult raw;
    auto st = db_->Search(std::span<const float>(query.embedding.data(), query.embedding.size()),
                          internal_topk, &raw);
    if (!st.ok())
        return st;

    if (raw.hits.empty())
    {
        // Fallback: ANN index may temporarily have no visible hits while writes
        // are still resident in the active memtable. Probe known ids directly.
        std::vector<float> probe_vec;
        Metadata probe_meta;
        for (VectorId id = 1; id < next_id_; ++id)
        {
            probe_vec.clear();
            probe_meta = Metadata();
            if (!db_->Get(id, &probe_vec, &probe_meta).ok())
            {
                continue;
            }
            raw.hits.push_back(SearchHit{id, 0.0f, {}});
        }
    }

    out->Clear();
    std::vector<float> tmp_vec;
    Metadata meta;

    for (const auto& h : raw.hits)
    {
        tmp_vec.clear();
        meta = Metadata();
        auto gst = db_->Get(h.id, &tmp_vec, &meta);
        if (!gst.ok())
        {
            continue;
        }
        AgentMemoryRecord rec;
        if (!DecodeMetadata(meta.tenant, &rec))
        {
            continue;
        }
        if (rec.agent_id != query.agent_id)
        {
            continue;
        }
        if (query.has_session_filter && rec.session_id != query.session_id)
        {
            continue;
        }
        if (query.has_kind_filter && rec.kind != query.kind)
        {
            continue;
        }
        if (rec.logical_ts < query.min_ts || rec.logical_ts > query.max_ts)
        {
            continue;
        }
        AgentMemoryHit hit;
        hit.record = std::move(rec);
        hit.score = (h.score != 0.0f)
                        ? h.score
                        : ComputeFallbackScore(
                              std::span<const float>(query.embedding.data(), query.embedding.size()),
                              std::span<const float>(tmp_vec.data(), tmp_vec.size()),
                              options_.metric);
        out->hits.push_back(std::move(hit));
    }

    std::sort(out->hits.begin(), out->hits.end(),
              [](const AgentMemoryHit& a, const AgentMemoryHit& b) {
                  return a.score > b.score;
              });

    if (out->hits.size() > query.topk)
    {
        out->hits.resize(query.topk);
    }

    if (query.topk == 0)
    {
        out->hits.clear();
    }

    return Status::Ok();
}

Status AgentMemory::SemanticSearch(const AgentMemoryQuery& query,
                                   AgentMemorySearchResult* out)
{
    auto st = EnsureOpen();
    if (!st.ok())
        return st;
    return SearchAndFilter(query, out);
}

Status AgentMemory::CollectPruneEntries(
    std::vector<PruneEntry>* out_entries) const
{
    if (!out_entries)
        return Status::InvalidArgument("out_entries is null");
    auto st = EnsureOpen();
    if (!st.ok())
        return st;

    out_entries->clear();
    std::shared_ptr<Snapshot> snap;
    st = db_->GetSnapshot(&snap);
    if (!st.ok())
        return st;
    std::unique_ptr<SnapshotIterator> it;
    st = db_->NewIterator(snap, &it);
    if (!st.ok())
        return st;

    std::vector<float> tmp_vec;
    Metadata meta;

    while (it->Next())
    {
        const VectorId id = it->id();
        tmp_vec.clear();
        meta = Metadata();
        auto gst = db_->Get(id, &tmp_vec, &meta);
        if (!gst.ok())
        {
            continue;
        }
        AgentMemoryRecord rec;
        if (!DecodeMetadata(meta.tenant, &rec))
        {
            continue;
        }
        PruneEntry e;
        e.id = id;
        e.agent_id = rec.agent_id;
        e.session_id = rec.session_id;
        e.kind = rec.kind;
        e.logical_ts = rec.logical_ts;
        e.approx_bytes = tmp_vec.size() * sizeof(float) + meta.tenant.size();
        out_entries->push_back(std::move(e));
    }
    return Status::Ok();
}

Status AgentMemory::PruneOld(std::string_view agent_id,
                             std::size_t keep_last_n,
                             std::int64_t min_ts_to_keep)
{
    std::vector<PruneEntry> entries;
    auto st = CollectPruneEntries(&entries);
    if (!st.ok())
        return st;

    std::vector<PruneEntry*> filtered;
    for (auto& e : entries)
    {
        if (e.agent_id == agent_id)
        {
            filtered.push_back(&e);
        }
    }
    if (filtered.empty())
        return Status::Ok();

    std::sort(filtered.begin(), filtered.end(),
              [](const PruneEntry* a, const PruneEntry* b) {
                  if (a->logical_ts != b->logical_ts)
                      return a->logical_ts < b->logical_ts;
                  return a->id < b->id;
              });

    std::size_t to_keep = keep_last_n;
    if (to_keep > filtered.size())
    {
        to_keep = filtered.size();
    }
    const std::size_t cutoff_index = filtered.size() - to_keep;

    for (std::size_t i = 0; i < filtered.size(); ++i)
    {
        const PruneEntry* e = filtered[i];
        if (e->logical_ts >= min_ts_to_keep)
        {
            continue;
        }
        if (i >= cutoff_index)
        {
            continue;
        }
        (void)db_->Delete(e->id);
    }
    return Status::Ok();
}

Status AgentMemory::PruneDeviceWide(std::size_t target_total_bytes)
{
    std::vector<PruneEntry> entries;
    auto st = CollectPruneEntries(&entries);
    if (!st.ok())
        return st;
    if (entries.empty())
        return Status::Ok();

    std::size_t total = 0;
    for (const auto& e : entries)
    {
        total += e.approx_bytes;
    }
    if (total <= target_total_bytes)
    {
        return Status::Ok();
    }

    std::sort(entries.begin(), entries.end(),
              [](const PruneEntry& a, const PruneEntry& b) {
                  if (a.logical_ts != b.logical_ts)
                      return a.logical_ts < b.logical_ts;
                  return a.id < b.id;
              });

    for (const auto& e : entries)
    {
        if (total <= target_total_bytes)
        {
            break;
        }
        (void)db_->Delete(e.id);
        if (total >= e.approx_bytes)
        {
            total -= e.approx_bytes;
        }
        else
        {
            total = 0;
        }
    }
    return Status::Ok();
}

Status AgentMemory::FreezeIfNeeded()
{
    auto st = EnsureOpen();
    if (!st.ok())
        return st;
    return db_->TryFreezeIfPressured();
}

} // namespace pomai

