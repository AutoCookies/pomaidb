#include "core/engine/engine.h"

#include <algorithm>
#include <filesystem>

#include "core/shard/runtime.h"

#include "storage/manifest/manifest.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"

namespace fs = std::filesystem;

namespace pomai::core
{

    Engine::Engine(pomai::DBOptions opt) : opt_(std::move(opt)) {}

    std::uint32_t Engine::ShardOf(VectorId id) const noexcept
    {
        return static_cast<std::uint32_t>(id % opt_.shard_count);
    }

    pomai::Status Engine::Open()
    {
        if (opt_.path.empty())
            return pomai::Status::InvalidArgument("path empty");
        if (opt_.dim == 0)
            return pomai::Status::InvalidArgument("dim=0");
        if (opt_.shard_count == 0)
            return pomai::Status::InvalidArgument("shard_count=0");

        fs::create_directories(opt_.path);

        auto st = pomai::storage::Manifest::EnsureInitialized(
            opt_.path, opt_.shard_count, opt_.dim);
        if (!st.ok())
            return st;

        shards_.reserve(opt_.shard_count);

        for (std::uint32_t sid = 0; sid < opt_.shard_count; ++sid)
        {
            auto wal = std::make_unique<pomai::storage::Wal>(
                opt_.path, sid, opt_.wal_segment_bytes, opt_.fsync);
            st = wal->Open();
            if (!st.ok())
                return st;

            auto mem = std::make_unique<pomai::table::MemTable>(
                opt_.dim, opt_.arena_block_bytes);

            st = wal->ReplayInto(*mem);
            if (!st.ok())
                return st;

            // 🔥 ShardRuntime complete type is known here
            auto rt = std::make_unique<ShardRuntime>(
                sid, opt_.dim, std::move(wal), std::move(mem), 1u << 16);

            auto shard = std::make_unique<Shard>(std::move(rt));
            st = shard->Start();
            if (!st.ok())
                return st;

            shards_.push_back(std::move(shard));
        }

        return pomai::Status::Ok();
    }

    pomai::Status Engine::Close()
    {
        shards_.clear(); // RAII -> Shard -> ShardRuntime stop clean
        return pomai::Status::Ok();
    }

    pomai::Status Engine::Put(VectorId id, std::span<const float> vec)
    {
        if (vec.size() != opt_.dim)
            return pomai::Status::InvalidArgument("dim mismatch");
        return shards_[ShardOf(id)]->Put(id, vec);
    }

    pomai::Status Engine::Delete(VectorId id)
    {
        return shards_[ShardOf(id)]->Delete(id);
    }

    pomai::Status Engine::Flush()
    {
        for (auto &s : shards_)
        {
            auto st = s->Flush();
            if (!st.ok())
                return st;
        }
        return pomai::Status::Ok();
    }

    pomai::Status Engine::Search(std::span<const float> query,
                                 std::uint32_t topk,
                                 std::vector<pomai::SearchHit> *out)
    {
        if (!out)
            return pomai::Status::InvalidArgument("out null");
        if (query.size() != opt_.dim)
            return pomai::Status::InvalidArgument("dim mismatch");

        out->clear();
        if (topk == 0)
            return pomai::Status::Ok();

        std::vector<pomai::SearchHit> merged;
        merged.reserve(topk * shards_.size());

        for (auto &shard : shards_)
        {
            std::vector<pomai::SearchHit> local;
            auto st = shard->Search(query, topk, &local);
            if (!st.ok())
                return st;

            merged.insert(merged.end(),
                          std::make_move_iterator(local.begin()),
                          std::make_move_iterator(local.end()));
        }

        std::sort(merged.begin(), merged.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a.score > b.score;
                  });

        if (merged.size() > topk)
            merged.resize(topk);

        *out = std::move(merged);
        return pomai::Status::Ok();
    }

} // namespace pomai::core
