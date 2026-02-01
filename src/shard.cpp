#include "pomai/core/shard.h"
#include "pomai/util/crc32c.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace pomai::core
{
    static constexpr std::uint32_t kSnapshotMagic = 0x4E534D50;
    static constexpr std::uint32_t kSnapshotVersion = 1;
    static constexpr std::size_t kCheckpointInterval = 50000;

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

    pomai::Status Shard::Start()
    {
        published_.store(std::make_shared<ReadSnapshot>(), std::memory_order_release);

        std::filesystem::path blob_path = opt_.wal_path;
        blob_path.replace_extension(".blob");
        blob_store_ = std::make_unique<BlobStore>(blob_path);

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
            auto st = LoadSnapshot(snap_path);
            if (!st.ok())
            {
                return pomai::Status::Internal("load snapshot failed: " + st.message);
            }
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
                const float *raw = arena_->GetVector(i);
                std::memcpy(vec.data(), raw, arena_->Dim() * sizeof(float));
                index_->AddPoint(vec, ids[i]);
            }
        }

        if (std::filesystem::exists(opt_.wal_path))
        {
            auto st = Recover();
            if (!st.ok())
            {
                return pomai::Status::Internal("wal recovery failed: " + st.message);
            }
        }

        return wal_.Open(opt_.wal_path, opt_.fsync_policy);
    }

    void Shard::Stop()
    {
        wal_.Close();
    }

    pomai::Status Shard::Flush()
    {
        return wal_.Flush();
    }

    pomai::Status Shard::CreateCheckpoint()
    {
        std::filesystem::path snap_path = opt_.wal_path;
        snap_path.replace_extension(".snap");
        std::filesystem::path tmp_path = snap_path;
        tmp_path.replace_extension(".snap.tmp");

        auto st = SaveSnapshot(tmp_path);
        if (!st.ok())
            return st;

        std::filesystem::path idx_path = opt_.wal_path;
        idx_path.replace_extension(".hnsw");
        try
        {
            index_->Save(idx_path.string());
        }
        catch (...)
        {
            return pomai::Status::IO("save hnsw index failed");
        }

        std::filesystem::path blob_idx_path = opt_.wal_path;
        blob_idx_path.replace_extension(".blob.idx");
        blob_store_->SaveIndexSnapshot(blob_idx_path);

        std::error_code ec;
        std::filesystem::rename(tmp_path, snap_path, ec);
        if (ec)
            return pomai::Status::IO("rename snapshot failed");

        wal_.Close();

#if defined(__linux__) || defined(__APPLE__)
        int fd = ::open(opt_.wal_path.c_str(), O_TRUNC | O_WRONLY, 0644);
        if (fd >= 0)
            ::close(fd);
#endif

        st = wal_.Open(opt_.wal_path, opt_.fsync_policy);
        if (!st.ok())
            return st;

        ops_since_checkpoint_ = 0;
        return pomai::Status::OK();
    }

    pomai::Status Shard::SaveSnapshot(const std::filesystem::path &path)
    {
        std::ofstream ofs(path, std::ios::binary | std::ios::out | std::ios::trunc);
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

        const auto &ids = arena_->Ids();
        if (!ids.empty())
        {
            ofs.write(reinterpret_cast<const char *>(ids.data()), ids.size() * sizeof(pomai::VectorId));
        }

        std::size_t num_pages = arena_->NumPages();
        std::size_t page_sz = PagedVectorArena::kPageSizeBytes;

        for (std::size_t i = 0; i < num_pages; ++i)
        {
            const float *page = arena_->GetPage(static_cast<std::uint32_t>(i));
            ofs.write(reinterpret_cast<const char *>(page), page_sz);
        }

        if (ofs.bad())
            return pomai::Status::IO("write snapshot data failed");

        return pomai::Status::OK();
    }

    pomai::Status Shard::LoadSnapshot(const std::filesystem::path &path)
    {
        std::ifstream ifs(path, std::ios::binary | std::ios::in);
        if (!ifs.is_open())
            return pomai::Status::IO("open snapshot failed");

        std::uint32_t magic, ver, dim;
        std::uint64_t num_vecs;

        ifs.read(reinterpret_cast<char *>(&magic), 4);
        if (magic != kSnapshotMagic)
            return pomai::Status::Internal("invalid snapshot magic");

        ifs.read(reinterpret_cast<char *>(&ver), 4);
        ifs.read(reinterpret_cast<char *>(&dim), 4);

        if (dim != arena_->Dim())
            return pomai::Status::Internal("snapshot dimension mismatch");

        ifs.read(reinterpret_cast<char *>(&num_vecs), 8);

        std::vector<pomai::VectorId> ids(num_vecs);
        ifs.read(reinterpret_cast<char *>(ids.data()), num_vecs * sizeof(pomai::VectorId));
        arena_->SetIds(std::move(ids));

        std::size_t vec_size = dim * sizeof(float);
        std::size_t vecs_per_page = PagedVectorArena::kPageSizeBytes / vec_size;
        if (vecs_per_page < 1)
            vecs_per_page = 1;

        std::size_t num_pages = (num_vecs + vecs_per_page - 1) / vecs_per_page;
        arena_->AllocatePages(num_pages);

        for (std::size_t i = 0; i < num_pages; ++i)
        {
            float *page = arena_->GetMutablePage(static_cast<std::uint32_t>(i));
            ifs.read(reinterpret_cast<char *>(page), PagedVectorArena::kPageSizeBytes);
        }

        upsert_count_.store(num_vecs, std::memory_order_relaxed);
        ops_since_checkpoint_ = 0;

        return pomai::Status::OK();
    }

    pomai::Status Shard::ApplyUpsert(std::vector<pomai::UpsertItem> &&items)
    {
        auto t0 = std::chrono::steady_clock::now();

        if (items.empty())
            return pomai::Status::OK();

        std::uint32_t dim = arena_->Dim();
        for (const auto &it : items)
        {
            if (it.vec.values.size() != dim)
                return pomai::Status::Invalid("dimension mismatch");
        }

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

            std::size_t data_bytes = d * sizeof(float);
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
            {
                blob_store_->Append(it.id, it.payload);
            }
        }

        upsert_count_.fetch_add(items.size(), std::memory_order_relaxed);
        ops_since_publish_ += items.size();
        ops_since_checkpoint_ += items.size();

        MaybePublishSnapshot();

        if (ops_since_checkpoint_ >= kCheckpointInterval)
        {
            CreateCheckpoint();
        }

        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        upsert_lat_us_.Add(static_cast<std::uint64_t>(us));

        return pomai::Status::OK();
    }

    SearchReply Shard::ExecuteSearch(const SearchRequest &req)
    {
        auto t0 = std::chrono::steady_clock::now();

        SearchReply rep;
        std::uint32_t dim = arena_->Dim();

        if (req.query.values.size() != dim)
        {
            rep.status = pomai::Status::Invalid("query dimension mismatch");
            return rep;
        }

        auto hits = index_->Search(req.query.values, req.topk);

        rep.hits.reserve(hits.size());
        for (auto &h : hits)
        {
            std::string payload = blob_store_->Get(h.second);
            rep.hits.push_back({h.second, h.first, std::move(payload)});
        }
        rep.status = pomai::Status::OK();

        search_count_.fetch_add(1, std::memory_order_relaxed);
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        search_lat_us_.Add(static_cast<std::uint64_t>(us));

        return rep;
    }

    float Shard::Score(const float *a, const float *b, std::uint32_t dim) const
    {
        float sum = 0.0f;
        for (std::uint32_t i = 0; i < dim; ++i)
        {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return -sum;
    }

    void Shard::MaybePublishSnapshot()
    {
        if (ops_since_publish_ < 1024)
            return;
        ops_since_publish_ = 0;

        auto snap = std::make_shared<ReadSnapshot>();
        snap->ids = arena_->Ids();
        published_.store(snap, std::memory_order_release);
    }

    std::shared_ptr<const ReadSnapshot> Shard::GetSnapshot() const
    {
        return published_.load(std::memory_order_acquire);
    }

    pomai::Status Shard::Recover()
    {
        WalReader reader(opt_.wal_path);
        if (!reader.Open().ok())
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

        MaybePublishSnapshot();
        return pomai::Status::OK();
    }

    pomai::Status Shard::ReplayPayload(const std::vector<std::byte> &payload)
    {
        if (payload.size() < sizeof(std::uint32_t))
            return pomai::Status::Internal("bad payload size");

        const std::byte *ptr = payload.data();
        std::uint32_t count;
        std::memcpy(&count, ptr, sizeof(std::uint32_t));
        ptr += sizeof(std::uint32_t);

        arena_->Reserve(arena_->Size() + count);

        for (std::uint32_t i = 0; i < count; ++i)
        {
            pomai::VectorId id;
            std::memcpy(&id, ptr, sizeof(pomai::VectorId));
            ptr += sizeof(pomai::VectorId);

            std::uint32_t dim;
            std::memcpy(&dim, ptr, sizeof(std::uint32_t));
            ptr += sizeof(std::uint32_t);

            std::vector<float> tmp(dim);
            std::memcpy(tmp.data(), ptr, dim * sizeof(float));
            ptr += dim * sizeof(float);

            std::uint32_t p_len;
            std::memcpy(&p_len, ptr, sizeof(std::uint32_t));
            ptr += sizeof(std::uint32_t);

            std::string payload_str;
            if (p_len > 0)
            {
                payload_str.resize(p_len);
                std::memcpy(payload_str.data(), ptr, p_len);
                ptr += p_len;
            }

            arena_->Add(id, tmp);
            try
            {
                index_->AddPoint(tmp, id);
            }
            catch (...)
            {
            }
            if (!payload_str.empty())
            {
                blob_store_->Append(id, payload_str);
            }
        }

        upsert_count_.fetch_add(count, std::memory_order_relaxed);
        return pomai::Status::OK();
    }

} // namespace pomai::core