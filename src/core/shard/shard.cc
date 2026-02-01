#include "core/shard/shard.h"
#include "core/shard/runtime.h"

namespace pomai::core
{

    Shard::Shard(std::unique_ptr<ShardRuntime> rt)
        : rt_(std::move(rt)) {}

    Shard::~Shard() = default;

    pomai::Status Shard::Start()
    {
        return rt_->Start();
    }

    pomai::Status Shard::Put(VectorId id, std::span<const float> vec)
    {
        PutCmd c;
        c.id = id;
        c.vec = vec.data();
        c.dim = static_cast<std::uint32_t>(vec.size());
        auto fut = c.done.get_future();
        auto st = rt_->Enqueue(Command{std::move(c)});
        if (!st.ok())
            return st;
        return fut.get();
    }

    pomai::Status Shard::Delete(VectorId id)
    {
        DelCmd c;
        c.id = id;
        auto fut = c.done.get_future();
        auto st = rt_->Enqueue(Command{std::move(c)});
        if (!st.ok())
            return st;
        return fut.get();
    }

    pomai::Status Shard::Flush()
    {
        FlushCmd c;
        auto fut = c.done.get_future();
        auto st = rt_->Enqueue(Command{std::move(c)});
        if (!st.ok())
            return st;
        return fut.get();
    }

    pomai::Status Shard::Search(std::span<const float> query,
                                std::uint32_t topk,
                                std::vector<pomai::SearchHit> *out)
    {
        return rt_->Search(query, topk, out);
    }

} // namespace pomai::core
