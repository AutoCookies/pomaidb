#include "pomai/core/shard.h"
#include "pomai/util/crc32c.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace pomai::core
{
    static constexpr std::uint32_t kSnapshotMagic = 0x4E534D50; // "PMSN"
    static constexpr std::uint32_t kSnapshotVersion = 2;        // v2: includes checkpoint_seq

    std::uint64_t Shard::NowSteadyNs()
    {
        using clock = std::chrono::steady_clock;
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count());
    }

    Shard::Shard(std::uint32_t shard_id, ShardOptions opt)
        : shard_id_(shard_id), opt_(std::move(opt))
    {
        std::uint32_t dim = opt_.vector_dim > 0 ? opt_.vector_dim : 128;
        arena_ = std::make_unique<PagedVectorArena>(dim);

        pomai::index::HnswIndex::Config idx_cfg;
        idx_cfg.dim = dim;
        idx_cfg.max_elements = 1000000;
        idx_cfg.m = 16;
        idx_cfg.ef_construction = 200;
        index_ = std::make_unique<pomai::index::HnswIndex>(idx_cfg);
    }

    std::filesystem::path Shard::MetaDir() const
    {
        // Keep backward compatibility: derive a directory from wal_path.
        // Example: /data/shard0.wal -> /data/shard0.wal.d/
        std::filesystem::path d = opt_.wal_path;
        d += ".d";
        return d;
    }

    std::filesystem::path Shard::WalDir() const { return MetaDir() / "wal"; }
    std::filesystem::path Shard::CheckpointsDir() const { return MetaDir() / "checkpoints"; }
    std::filesystem::path Shard::ManifestPath() const { return MetaDir() / "MANIFEST"; }

    pomai::Status Shard::OpenActiveWal(std::uint64_t wal_id)
    {
        active_wal_id_ = wal_id;
        std::error_code ec;
        std::filesystem::create_directories(WalDir(), ec);

        const auto path = WalFilePath(WalDir(), wal_id);
        return wal_.Open(path, opt_.fsync_policy);
    }

    void Shard::PruneOldWalFiles(std::uint64_t keep_from_id)
    {
        std::vector<std::uint64_t> ids;
        if (!ListWalFileIds(WalDir(), ids).ok())
            return;

        std::error_code ec;
        for (auto id : ids)
        {
            if (id < keep_from_id)
            {
                std::filesystem::remove(WalFilePath(WalDir(), id), ec);
            }
        }
    }

    pomai::Status Shard::Start()
    {
        published_.store(std::make_shared<ReadSnapshot>(), std::memory_order_release);

        // BlobStore stays in legacy location (same directory as wal_path), for now.
        std::filesystem::path blob_path = opt_.wal_path;
        blob_path.replace_extension(".blob");
        blob_store_ = std::make_unique<BlobStore>(blob_path);

        // Step2: if MANIFEST exists, recover from it; else fall back to legacy.
        Manifest m;
        auto mst = Manifest::Load(ManifestPath(), m);
        if (mst.ok())
        {
            auto st = RecoverFromManifest(m);
            if (!st.ok())
                return st;

            // After a successful recovery from manifest, last checkpoint time is "now" for scheduling.
            last_checkpoint_time_ns_.store(NowSteadyNs(), std::memory_order_release);

            // Continue writing from wal_start_id (first WAL after checkpoint)
            return OpenActiveWal(m.wal_start_id);
        }

        // Legacy load: blob index + snapshot + hnsw + single wal replay.
        std::filesystem::path blob_idx_path = opt_.wal_path;
        blob_idx_path.replace_extension(".blob.idx");
        if (std::filesystem::exists(blob_idx_path))
        {
            blob_store_->LoadIndexSnapshot(blob_idx_path);
        }

        std::filesystem::path snap_path = opt_.wal_path;
        snap_path.replace_extension(".snap");
        if (std::filesystem::exists(snap_path))
        {
            std::uint64_t ignored_seq = 0;
            auto st = LoadSnapshot(snap_path, &ignored_seq);
            if (!st.ok())
                return pomai::Status::Internal("load snapshot failed: " + st.message);
        }

        std::filesystem::path idx_path = opt_.wal_path;
        idx_path.replace_extension(".hnsw");
        if (std::filesystem::exists(idx_path))
        {
            try
            {
                index_->Load(idx_path.string());
            }
            catch (...)
            {
                return pomai::Status::Internal("load hnsw index failed");
            }
        }
        else if (arena_->Size() > 0)
        {
            const auto &ids = arena_->Ids();
            for (size_t i = 0; i < ids.size(); ++i)
            {
                std::vector<float> vec(arena_->Dim());
                const float *raw = arena_->GetVector(static_cast<std::uint32_t>(i));
                std::memcpy(vec.data(), raw, arena_->Dim() * sizeof(float));
                index_->AddPoint(vec, ids[i]);
            }
        }

        if (std::filesystem::exists(opt_.wal_path))
        {
            auto st = RecoverLegacySingleWal();
            if (!st.ok())
                return pomai::Status::Internal("wal recovery failed: " + st.message);
        }

        // Keep legacy: write to single wal file when MANIFEST not present.
        return wal_.Open(opt_.wal_path, opt_.fsync_policy);
    }

    void Shard::Stop() { wal_.Close(); }

    pomai::Status Shard::Flush() { return wal_.Flush(); }

    pomai::Status Shard::CreateCheckpoint()
    {
        // Step2: write checkpoint artifacts under MetaDir(), then publish MANIFEST, then rotate WAL.
        std::error_code ec;
        std::filesystem::create_directories(CheckpointsDir(), ec);
        std::filesystem::create_directories(WalDir(), ec);

        const std::uint64_t cp_seq = upsert_count_.load(std::memory_order_relaxed);

        // Snapshot
        char snap_name[128];
        std::snprintf(snap_name, sizeof(snap_name), "cp_%06llu.snap",
                      static_cast<unsigned long long>(cp_seq));
        auto snap_rel = std::string("checkpoints/") + snap_name;

        const auto snap_path = MetaDir() / snap_rel;
        const auto snap_tmp = snap_path.string() + ".tmp";

        auto st = SaveSnapshot(snap_tmp, cp_seq);
        if (!st.ok())
            return st;

        std::filesystem::rename(snap_tmp, snap_path, ec);
        if (ec)
            return pomai::Status::IO("rename checkpoint snapshot failed");

        // Blob index snapshot
        char blob_idx_name[128];
        std::snprintf(blob_idx_name, sizeof(blob_idx_name), "cp_%06llu.blob.idx",
                      static_cast<unsigned long long>(cp_seq));
        auto blob_idx_rel = std::string("checkpoints/") + blob_idx_name;
        blob_store_->SaveIndexSnapshot(MetaDir() / blob_idx_rel);

        // Optional HNSW persistence (expensive)
        std::string hnsw_rel;
        if (opt_.persist_hnsw_on_checkpoint)
        {
            char hnsw_name[128];
            std::snprintf(hnsw_name, sizeof(hnsw_name), "cp_%06llu.hnsw",
                          static_cast<unsigned long long>(cp_seq));
            hnsw_rel = std::string("checkpoints/") + hnsw_name;
            try
            {
                index_->Save((MetaDir() / hnsw_rel).string());
            }
            catch (...)
            {
                return pomai::Status::IO("save hnsw index failed");
            }
        }

        // Rotate WAL: next id
        const std::uint64_t next_wal_id = (active_wal_id_ == 0) ? 1 : (active_wal_id_ + 1);
        wal_.Close();

        // Publish MANIFEST first so recovery can find the checkpoint + wal_start_id.
        Manifest m;
        m.checkpoint_seq = cp_seq;
        m.wal_start_id = next_wal_id;
        m.snapshot_rel = snap_rel;
        m.blob_idx_rel = blob_idx_rel;
        m.hnsw_rel = hnsw_rel;

        st = m.SaveAtomic(ManifestPath());
        if (!st.ok())
            return st;

        // Now open new WAL file.
        st = OpenActiveWal(next_wal_id);
        if (!st.ok())
            return st;

        // Best-effort prune old WAL files (safe because checkpoint is published).
        PruneOldWalFiles(next_wal_id);

        ops_since_checkpoint_ = 0;
        last_checkpoint_time_ns_.store(NowSteadyNs(), std::memory_order_release);
        return pomai::Status::OK();
    }

    pomai::Status Shard::SaveSnapshot(const std::filesystem::path &path, std::uint64_t checkpoint_seq)
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open())
            return pomai::Status::IO("create snapshot file failed");

        std::uint32_t magic = kSnapshotMagic;
        std::uint32_t ver = kSnapshotVersion;
        std::uint32_t dim = arena_->Dim();
        std::uint64_t num_vecs = arena_->Size();

        ofs.write(reinterpret_cast<const char *>(&magic), 4);
        ofs.write(reinterpret_cast<const char *>(&ver), 4);
        ofs.write(reinterpret_cast<const char *>(&dim), 4);
        ofs.write(reinterpret_cast<const char *>(&num_vecs), 8);
        ofs.write(reinterpret_cast<const char *>(&checkpoint_seq), 8);

        const auto &ids = arena_->Ids();
        if (!ids.empty())
        {
            ofs.write(reinterpret_cast<const char *>(ids.data()),
                      static_cast<std::streamsize>(ids.size() * sizeof(pomai::VectorId)));
        }

        const std::size_t num_pages = arena_->NumPages();
        const std::size_t page_sz = PagedVectorArena::kPageSizeBytes;
        for (std::size_t i = 0; i < num_pages; ++i)
        {
            const float *page = arena_->GetPage(static_cast<std::uint32_t>(i));
            ofs.write(reinterpret_cast<const char *>(page), static_cast<std::streamsize>(page_sz));
        }

        if (ofs.bad())
            return pomai::Status::IO("write snapshot data failed");
        return pomai::Status::OK();
    }

    pomai::Status Shard::LoadSnapshot(const std::filesystem::path &path, std::uint64_t *out_checkpoint_seq)
    {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open())
            return pomai::Status::IO("open snapshot failed");

        std::uint32_t magic = 0, ver = 0, dim = 0;
        std::uint64_t num_vecs = 0;
        std::uint64_t checkpoint_seq = 0;

        ifs.read(reinterpret_cast<char *>(&magic), 4);
        if (magic != kSnapshotMagic)
            return pomai::Status::Internal("invalid snapshot magic");

        ifs.read(reinterpret_cast<char *>(&ver), 4);
        if (ver != kSnapshotVersion)
            return pomai::Status::Internal("unsupported snapshot version");

        ifs.read(reinterpret_cast<char *>(&dim), 4);
        if (dim != arena_->Dim())
            return pomai::Status::Internal("snapshot dimension mismatch");

        ifs.read(reinterpret_cast<char *>(&num_vecs), 8);
        ifs.read(reinterpret_cast<char *>(&checkpoint_seq), 8);

        std::vector<pomai::VectorId> ids(num_vecs);
        if (num_vecs > 0)
        {
            ifs.read(reinterpret_cast<char *>(ids.data()),
                     static_cast<std::streamsize>(num_vecs * sizeof(pomai::VectorId)));
        }
        arena_->SetIds(std::move(ids));

        const std::size_t vec_size = static_cast<std::size_t>(dim) * sizeof(float);
        std::size_t vecs_per_page = PagedVectorArena::kPageSizeBytes / vec_size;
        if (vecs_per_page < 1)
            vecs_per_page = 1;

        const std::size_t num_pages =
            (static_cast<std::size_t>(num_vecs) + vecs_per_page - 1) / vecs_per_page;
        arena_->AllocatePages(num_pages);

        for (std::size_t i = 0; i < num_pages; ++i)
        {
            float *page = arena_->GetMutablePage(static_cast<std::uint32_t>(i));
            ifs.read(reinterpret_cast<char *>(page),
                     static_cast<std::streamsize>(PagedVectorArena::kPageSizeBytes));
        }

        upsert_count_.store(num_vecs, std::memory_order_relaxed);
        ops_since_checkpoint_ = 0;

        if (out_checkpoint_seq)
            *out_checkpoint_seq = checkpoint_seq;
        return pomai::Status::OK();
    }

    void Shard::MaybePublishSnapshot()
    {
        if (ops_since_publish_ < 4096)
            return;
        ops_since_publish_ = 0;

        auto snap = std::make_shared<ReadSnapshot>();
        snap->num_vectors = arena_->Size();
        published_.store(snap, std::memory_order_release);
    }

    std::shared_ptr<const ReadSnapshot> Shard::GetSnapshot() const
    {
        return published_.load(std::memory_order_acquire);
    }

    pomai::Status Shard::RecoverLegacySingleWal()
    {
        WalReader reader(opt_.wal_path);
        auto ost = reader.Open();
        if (!ost.ok())
            return pomai::Status::OK();

        std::vector<std::byte> payload;
        while (true)
        {
            auto st = reader.ReadNext(payload);
            if (st.code == pomai::StatusCode::NotFound)
                break;
            if (!st.ok())
                return st;

            st = ReplayPayload(payload);
            if (!st.ok())
                return st;
        }
        return pomai::Status::OK();
    }

    pomai::Status Shard::RecoverFromManifest(const Manifest &m)
    {
        const auto base = MetaDir();

        // Load checkpoint artifacts
        const auto snap_path = base / m.snapshot_rel;
        const auto blob_idx_path = base / m.blob_idx_rel;

        if (std::filesystem::exists(blob_idx_path))
        {
            blob_store_->LoadIndexSnapshot(blob_idx_path);
        }

        std::uint64_t snap_seq = 0;
        if (std::filesystem::exists(snap_path))
        {
            auto st = LoadSnapshot(snap_path, &snap_seq);
            if (!st.ok())
                return st;
        }

        // Optional HNSW load
        if (!m.hnsw_rel.empty())
        {
            const auto hnsw_path = base / m.hnsw_rel;
            if (std::filesystem::exists(hnsw_path))
            {
                try
                {
                    index_->Load(hnsw_path.string());
                }
                catch (...)
                {
                    return pomai::Status::Internal("load hnsw failed");
                }
            }
        }
        else if (arena_->Size() > 0)
        {
            // Rebuild index from vectors if not persisted.
            const auto &ids = arena_->Ids();
            for (size_t i = 0; i < ids.size(); ++i)
            {
                std::vector<float> vec(arena_->Dim());
                const float *raw = arena_->GetVector(static_cast<std::uint32_t>(i));
                std::memcpy(vec.data(), raw, arena_->Dim() * sizeof(float));
                index_->AddPoint(vec, ids[i]);
            }
        }

        // Replay WAL files with id >= wal_start_id
        std::vector<std::uint64_t> ids;
        auto st = ListWalFileIds(WalDir(), ids);
        if (!st.ok() && st.code != pomai::StatusCode::NotFound)
            return st;

        for (auto id : ids)
        {
            if (id < m.wal_start_id)
                continue;

            WalReader reader(WalFilePath(WalDir(), id));
            auto ost = reader.Open();
            if (!ost.ok())
                continue; // missing is fine

            std::vector<std::byte> payload;
            while (true)
            {
                auto rs = reader.ReadNext(payload);
                if (rs.code == pomai::StatusCode::NotFound)
                    break;
                if (!rs.ok())
                    return rs;

                auto ap = ReplayPayload(payload);
                if (!ap.ok())
                    return ap;
            }
        }

        return pomai::Status::OK();
    }

    pomai::Status Shard::ReplayPayload(const std::vector<std::byte> &payload)
    {
        const std::byte *ptr = payload.data();
        const std::byte *end = payload.data() + payload.size();

        if (ptr + sizeof(std::uint32_t) > end)
            return pomai::Status::IO("bad payload");
        std::uint32_t count = 0;
        std::memcpy(&count, ptr, sizeof(std::uint32_t));
        ptr += sizeof(std::uint32_t);

        for (std::uint32_t i = 0; i < count; ++i)
        {
            if (ptr + sizeof(pomai::VectorId) > end)
                return pomai::Status::IO("bad payload");
            pomai::VectorId id = 0;
            std::memcpy(&id, ptr, sizeof(pomai::VectorId));
            ptr += sizeof(pomai::VectorId);

            if (ptr + sizeof(std::uint32_t) > end)
                return pomai::Status::IO("bad payload");
            std::uint32_t dim = 0;
            std::memcpy(&dim, ptr, sizeof(std::uint32_t));
            ptr += sizeof(std::uint32_t);

            if (ptr + static_cast<std::size_t>(dim) * sizeof(float) > end)
                return pomai::Status::IO("bad payload");
            std::vector<float> vec(dim);
            std::memcpy(vec.data(), ptr, static_cast<std::size_t>(dim) * sizeof(float));
            ptr += static_cast<std::size_t>(dim) * sizeof(float);

            if (ptr + sizeof(std::uint32_t) > end)
                return pomai::Status::IO("bad payload");
            std::uint32_t plen = 0;
            std::memcpy(&plen, ptr, sizeof(std::uint32_t));
            ptr += sizeof(std::uint32_t);

            std::string payload_str;
            if (plen > 0)
            {
                if (ptr + plen > end)
                    return pomai::Status::IO("bad payload");
                payload_str.assign(reinterpret_cast<const char *>(ptr), plen);
                ptr += plen;
            }

            arena_->Add(id, vec);
            try
            {
                index_->AddPoint(vec, id);
            }
            catch (...)
            {
            }

            if (!payload_str.empty())
                blob_store_->Append(id, payload_str);
        }

        upsert_count_.store(arena_->Size(), std::memory_order_relaxed);
        return pomai::Status::OK();
    }

    PayloadBatchReply Shard::GetPayloadBatch(const std::vector<pomai::VectorId> &ids)
    {
        PayloadBatchReply rep;
        rep.payloads.reserve(ids.size());
        for (auto id : ids)
        {
            rep.payloads.push_back(blob_store_ ? blob_store_->Get(id) : std::string{});
        }
        rep.status = pomai::Status::OK();
        return rep;
    }

    pomai::Status Shard::ApplyUpsert(std::vector<pomai::UpsertItem> &&items)
    {
        auto t0 = std::chrono::steady_clock::now();
        if (items.empty())
            return pomai::Status::OK();

        const std::uint32_t dim = arena_->Dim();
        for (const auto &it : items)
        {
            if (it.vec.values.size() != dim)
                return pomai::Status::Invalid("dimension mismatch");
        }

        // Frame: [payload_len u32][crc u32][payload...]
        std::size_t payload_size = sizeof(std::uint32_t);
        for (const auto &it : items)
        {
            payload_size += sizeof(pomai::VectorId);
            payload_size += sizeof(std::uint32_t);
            payload_size += it.vec.values.size() * sizeof(float);
            payload_size += sizeof(std::uint32_t);
            payload_size += it.payload.size();
        }

        std::size_t total_frame_size = sizeof(std::uint32_t) * 2 + payload_size;
        std::vector<std::byte> buffer(total_frame_size);
        std::byte *ptr = buffer.data();

        std::uint32_t payload_sz_u32 = static_cast<std::uint32_t>(payload_size);
        std::memcpy(ptr, &payload_sz_u32, sizeof(std::uint32_t));
        ptr += sizeof(std::uint32_t);

        std::byte *crc_ptr = ptr;
        ptr += sizeof(std::uint32_t);

        std::byte *payload_start = ptr;

        std::uint32_t count = static_cast<std::uint32_t>(items.size());
        std::memcpy(ptr, &count, sizeof(std::uint32_t));
        ptr += sizeof(std::uint32_t);

        for (const auto &it : items)
        {
            std::memcpy(ptr, &it.id, sizeof(pomai::VectorId));
            ptr += sizeof(pomai::VectorId);

            std::uint32_t d = static_cast<std::uint32_t>(it.vec.values.size());
            std::memcpy(ptr, &d, sizeof(std::uint32_t));
            ptr += sizeof(std::uint32_t);

            std::size_t data_bytes = static_cast<std::size_t>(d) * sizeof(float);
            std::memcpy(ptr, it.vec.values.data(), data_bytes);
            ptr += data_bytes;

            std::uint32_t p_len = static_cast<std::uint32_t>(it.payload.size());
            std::memcpy(ptr, &p_len, sizeof(std::uint32_t));
            ptr += sizeof(std::uint32_t);

            if (p_len > 0)
            {
                std::memcpy(ptr, it.payload.data(), p_len);
                ptr += p_len;
            }
        }

        std::uint32_t checksum = pomai::util::Crc32c(payload_start, payload_size);
        std::memcpy(crc_ptr, &checksum, sizeof(std::uint32_t));

        auto st = wal_.Append(buffer.data(), buffer.size());
        if (!st.ok())
            return st;

        for (const auto &it : items)
        {
            arena_->Add(it.id, it.vec.values);
            try
            {
                index_->AddPoint(it.vec.values, it.id);
            }
            catch (...)
            {
            }
            if (!it.payload.empty())
                blob_store_->Append(it.id, it.payload);
        }

        upsert_count_.fetch_add(items.size(), std::memory_order_relaxed);
        ops_since_publish_ += items.size();
        ops_since_checkpoint_ += items.size();

        MaybePublishSnapshot();

        // Step3 production: DO NOT checkpoint inline here.
        // MaintenanceScheduler will enqueue CmdCheckpoint when WAL bytes / time thresholds are met.
        // If you still want legacy ops-triggered checkpointing, enable it explicitly outside hot path.

        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        upsert_lat_us_.Add(static_cast<std::uint64_t>(us));

        return pomai::Status::OK();
    }

    SearchReply Shard::ExecuteSearch(const SearchRequest &req)
    {
        auto t0 = std::chrono::steady_clock::now();

        SearchReply rep;
        const std::uint32_t dim = arena_->Dim();
        if (req.query.values.size() != dim)
        {
            rep.status = pomai::Status::Invalid("query dimension mismatch");
            return rep;
        }

        auto hits = index_->Search(req.query.values, req.topk);
        rep.hits.reserve(hits.size());

        // Step1+Step2: keep search hot path IO-free by default.
        for (auto &h : hits)
        {
            rep.hits.push_back({h.second, h.first, ""});
        }
        rep.status = pomai::Status::OK();

        search_count_.fetch_add(1, std::memory_order_relaxed);
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        search_lat_us_.Add(static_cast<std::uint64_t>(us));

        return rep;
    }

    // ---- Step3 signals ----

    std::uint64_t Shard::WalBytes() const
    {
        return wal_.BytesWritten();
    }

    std::uint64_t Shard::MsSinceLastCheckpoint() const
    {
        const std::uint64_t last = last_checkpoint_time_ns_.load(std::memory_order_acquire);
        if (last == 0)
            return 0;

        const std::uint64_t now = NowSteadyNs();
        if (now <= last)
            return 0;

        return (now - last) / 1000000ull;
    }

} // namespace pomai::core
