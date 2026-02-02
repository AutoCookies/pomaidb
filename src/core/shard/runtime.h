#pragma once

#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <thread>
#include <variant>
#include <vector>

#include "core/shard/mailbox.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/types.h"

namespace pomai::storage
{
    class Wal;
}
namespace pomai::table
{
    class MemTable;
}

// Forward declare IVF (avoid heavy include in header).
namespace pomai::index
{
    class IvfCoarse;
}

namespace pomai::core
{

    struct PutCmd
    {
        VectorId id{};
        const float *vec{};
        std::uint32_t dim{};
        std::promise<pomai::Status> done;
    };

    struct DelCmd
    {
        VectorId id{};
        std::promise<pomai::Status> done;
    };

    struct FlushCmd
    {
        std::promise<pomai::Status> done;
    };

    // MUST be complete before being used in std::promise<SearchReply>.
    struct SearchReply
    {
        pomai::Status st;
        std::vector<pomai::SearchHit> hits;
    };

    struct SearchCmd
    {
        std::vector<float> query;
        std::uint32_t topk{0};
        std::promise<SearchReply> done;
    };

    struct StopCmd
    {
        std::promise<void> done;
    };

    using Command = std::variant<PutCmd, DelCmd, FlushCmd, SearchCmd, StopCmd>;

    class ShardRuntime
    {
    public:
        ShardRuntime(std::uint32_t shard_id,
                     std::uint32_t dim,
                     std::unique_ptr<storage::Wal> wal,
                     std::unique_ptr<table::MemTable> mem,
                     std::size_t mailbox_cap);

        ~ShardRuntime();

        ShardRuntime(const ShardRuntime &) = delete;
        ShardRuntime &operator=(const ShardRuntime &) = delete;

        pomai::Status Start();
        pomai::Status Enqueue(Command &&cmd);

        pomai::Status Put(pomai::VectorId id, std::span<const float> vec);
        pomai::Status Delete(pomai::VectorId id);
        pomai::Status Flush();

        pomai::Status Search(std::span<const float> query,
                             std::uint32_t topk,
                             std::vector<pomai::SearchHit> *out);

    private:
        void RunLoop();

        pomai::Status HandlePut(PutCmd &c);
        pomai::Status HandleDel(DelCmd &c);
        pomai::Status HandleFlush(FlushCmd &c);

        SearchReply HandleSearch(SearchCmd &c);
        pomai::Status SearchLocalInternal(std::span<const float> query,
                                          std::uint32_t topk,
                                          std::vector<pomai::SearchHit> *out);

        const std::uint32_t shard_id_;
        const std::uint32_t dim_;

        std::unique_ptr<storage::Wal> wal_;
        std::unique_ptr<table::MemTable> mem_;

        // IVF coarse index for candidate selection (centroid routing).
        std::unique_ptr<pomai::index::IvfCoarse> ivf_;

        BoundedMpscQueue<Command> mailbox_;

        std::jthread worker_;
        std::atomic<bool> started_{false};
    };

} // namespace pomai::core
