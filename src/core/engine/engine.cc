#include "core/engine/engine.h"

#include <algorithm>
#include <filesystem>
#include <thread>
#include <utility>

#include "core/shard/runtime.h"
#include "core/shard/shard.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"

namespace pomai::core
{
    namespace
    {
        constexpr std::size_t kMailboxCap = 4096;
        constexpr std::size_t kArenaBlockBytes = 1u << 20;  // 1 MiB
        constexpr std::size_t kWalSegmentBytes = 64u << 20; // 64 MiB

        static void MergeTopK(std::vector<pomai::SearchHit> *all, std::uint32_t k)
        {
            if (!all)
                return;
            if (all->size() <= k)
            {
                std::sort(all->begin(), all->end(),
                          [](const auto &a, const auto &b)
                          { return a.score > b.score; });
                return;
            }

            std::nth_element(all->begin(), all->begin() + static_cast<std::ptrdiff_t>(k), all->end(),
                             [](const auto &a, const auto &b)
                             { return a.score > b.score; });
            all->resize(k);

            std::sort(all->begin(), all->end(),
                      [](const auto &a, const auto &b)
                      { return a.score > b.score; });
        }
    } // namespace

    Engine::Engine(pomai::DBOptions opt) : opt_(std::move(opt)) {}
    Engine::~Engine() = default;

    std::uint32_t Engine::ShardOf(VectorId id, std::uint32_t shard_count)
    {
        return shard_count == 0 ? 0u : static_cast<std::uint32_t>(id % shard_count);
    }

    Status Engine::Open()
    {
        if (opened_)
            return Status::Ok();
        return OpenLocked();
    }

    Status Engine::OpenLocked()
    {
        if (opt_.dim == 0)
            return Status::InvalidArgument("dim must be > 0");
        if (opt_.shard_count == 0)
            return Status::InvalidArgument("shard_count must be > 0");

        std::error_code ec;
        std::filesystem::create_directories(opt_.path, ec);
        if (ec)
            return Status::IOError("create_directories failed");

        shards_.clear();
        shards_.reserve(opt_.shard_count);

        for (std::uint32_t i = 0; i < opt_.shard_count; ++i)
        {
            auto wal = std::make_unique<storage::Wal>(opt_.path, i, kWalSegmentBytes, opt_.fsync);
            auto st = wal->Open();
            if (!st.ok())
                return st;

            auto mem = std::make_unique<table::MemTable>(opt_.dim, kArenaBlockBytes);

            st = wal->ReplayInto(*mem);
            if (!st.ok())
                return st;

            auto shard_dir = (std::filesystem::path(opt_.path) / "shards" / std::to_string(i)).string();
            // Create shard dir if not exists (Wal might have created parent, but shards/i might be missing if new logic?)
            std::filesystem::create_directories(shard_dir, ec);

            auto rt = std::make_unique<ShardRuntime>(i, shard_dir, opt_.dim, std::move(wal), std::move(mem), kMailboxCap);
            auto shard = std::make_unique<Shard>(std::move(rt));

            st = shard->Start();
            if (!st.ok())
                return st;

            shards_.push_back(std::move(shard));
        }
        
        // Init thread pool
        size_t threads = std::thread::hardware_concurrency();
        if (threads < 4) threads = 4;
        search_pool_ = std::make_unique<util::ThreadPool>(threads);

        opened_ = true;
        return Status::Ok();
    }

    Status Engine::Close()
    {
        if (!opened_)
            return Status::Ok();
        search_pool_.reset(); // Stop pool first
        shards_.clear();
        opened_ = false;
        return Status::Ok();
    }

    Status Engine::Put(VectorId id, std::span<const float> vec)
    {
        if (!opened_)
            return Status::InvalidArgument("engine not opened");
        if (static_cast<std::uint32_t>(vec.size()) != opt_.dim)
            return Status::InvalidArgument("dim mismatch");
        const auto sid = ShardOf(id, opt_.shard_count);
        return shards_[sid]->Put(id, vec);
    }

    Status Engine::Get(VectorId id, std::vector<float> *out)
    {
        if (!opened_)
            return Status::InvalidArgument("engine not opened");
        if (!out)
            return Status::InvalidArgument("out=null");
        const auto sid = ShardOf(id, opt_.shard_count);
        return shards_[sid]->Get(id, out);
    }

    Status Engine::Exists(VectorId id, bool *exists)
    {
        if (!opened_)
            return Status::InvalidArgument("engine not opened");
        if (!exists)
            return Status::InvalidArgument("exists=null");
        const auto sid = ShardOf(id, opt_.shard_count);
        return shards_[sid]->Exists(id, exists);
    }

    Status Engine::Delete(VectorId id)
    {
        if (!opened_)
            return Status::InvalidArgument("engine not opened");
        const auto sid = ShardOf(id, opt_.shard_count);
        return shards_[sid]->Delete(id);
    }

    Status Engine::Flush()
    {
        if (!opened_)
            return Status::InvalidArgument("engine not opened");
        for (auto &s : shards_)
        {
            auto st = s->Flush();
            if (!st.ok())
                return st;
        }
        return Status::Ok();
    }

    Status Engine::Freeze()
    {
        if (!opened_) return Status::InvalidArgument("engine not opened");
        for (auto &s : shards_) {
            Status st = s->Freeze();
            if (!st.ok()) return st;
        }
        return Status::Ok();
    }

    Status Engine::Compact()
    {
        if (!opened_) return Status::InvalidArgument("engine not opened");
        for (auto &s : shards_) {
            Status st = s->Compact();
            if (!st.ok()) return st;
        }
        return Status::Ok();
    }

    Status Engine::Search(std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out)
    {
        if (!opened_)
            return Status::InvalidArgument("engine not opened");
        if (!out)
            return Status::InvalidArgument("out=null");

        out->Clear();
        if (static_cast<std::uint32_t>(query.size()) != opt_.dim)
            return Status::InvalidArgument("dim mismatch");
        if (topk == 0)
            return Status::Ok();

        std::vector<std::vector<pomai::SearchHit>> per(opt_.shard_count);
        std::vector<std::future<pomai::Status>> futures;
        futures.reserve(opt_.shard_count);

        for (std::uint32_t i = 0; i < opt_.shard_count; ++i)
        {
            futures.push_back(search_pool_->Enqueue([&, i]
                            { return shards_[i]->SearchLocal(query, topk, &per[i]); }));
        }
        
        for (auto &f : futures)
            (void)f.get();

        std::vector<pomai::SearchHit> merged;
        for (auto &v : per)
            merged.insert(merged.end(), v.begin(), v.end());

        MergeTopK(&merged, topk);
        out->hits = std::move(merged);
        return Status::Ok();
    }

} // namespace pomai::core
