#include "core/shard/runtime.h"

#include <algorithm>
#include <cmath>

#include "storage/wal/wal.h"
#include "table/memtable.h"

namespace pomai::core
{

    static float Dot(std::span<const float> a, std::span<const float> b)
    {
        float s = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i)
            s += a[i] * b[i];
        return s;
    }

    ShardRuntime::ShardRuntime(std::uint32_t shard_id,
                               std::uint32_t dim,
                               std::unique_ptr<storage::Wal> wal,
                               std::unique_ptr<table::MemTable> mem,
                               std::size_t mailbox_cap)
        : shard_id_(shard_id),
          dim_(dim),
          wal_(std::move(wal)),
          mem_(std::move(mem)),
          mailbox_(mailbox_cap) {}

    ShardRuntime::~ShardRuntime()
    {
        if (started_.load(std::memory_order_relaxed))
        {
            StopCmd c;
            auto fut = c.done.get_future();
            (void)Enqueue(Command{std::move(c)});
            fut.wait();
        }
    }

    pomai::Status ShardRuntime::Start()
    {
        if (started_.exchange(true))
            return pomai::Status::Busy("shard already started");
        worker_ = std::jthread([this]
                               { RunLoop(); });
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::Enqueue(Command &&cmd)
    {
        if (!started_.load(std::memory_order_relaxed))
            return pomai::Status::Aborted("shard not started");
        if (!mailbox_.PushBlocking(std::move(cmd)))
            return pomai::Status::Aborted("mailbox closed");
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::Search(std::span<const float> query,
                                       std::uint32_t topk,
                                       std::vector<pomai::SearchHit> *out)
    {
        if (!out)
            return pomai::Status::InvalidArgument("out null");
        if (query.size() != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");
        if (topk == 0)
        {
            out->clear();
            return pomai::Status::Ok();
        }

        SearchCmd c;
        c.topk = topk;
        c.query.assign(query.begin(), query.end());
        auto fut = c.done.get_future();

        auto st = Enqueue(Command{std::move(c)});
        if (!st.ok())
            return st;

        auto r = fut.get();
        if (!r.st.ok())
            return r.st;
        *out = std::move(r.hits);
        return pomai::Status::Ok();
    }

    void ShardRuntime::RunLoop()
    {
        for (;;)
        {
            auto opt = mailbox_.PopBlocking();
            if (!opt.has_value())
                break;

            Command cmd = std::move(*opt);

            if (auto *c = std::get_if<PutCmd>(&cmd))
            {
                c->done.set_value(HandlePut(*c));
                continue;
            }
            if (auto *c = std::get_if<DelCmd>(&cmd))
            {
                c->done.set_value(HandleDel(*c));
                continue;
            }
            if (auto *c = std::get_if<FlushCmd>(&cmd))
            {
                c->done.set_value(HandleFlush(*c));
                continue;
            }
            if (auto *c = std::get_if<SearchCmd>(&cmd))
            {
                c->done.set_value(HandleSearch(*c));
                continue;
            }
            if (auto *c = std::get_if<StopCmd>(&cmd))
            {
                mailbox_.Close();
                c->done.set_value();
                break;
            }
        }
    }

    pomai::Status ShardRuntime::HandlePut(PutCmd &c)
    {
        if (c.dim != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");
        auto st = wal_->AppendPut(c.id, {c.vec, c.dim});
        if (!st.ok())
            return st;
        return mem_->Put(c.id, {c.vec, c.dim});
    }

    pomai::Status ShardRuntime::HandleDel(DelCmd &c)
    {
        auto st = wal_->AppendDelete(c.id);
        if (!st.ok())
            return st;
        return mem_->Delete(c.id);
    }

    pomai::Status ShardRuntime::HandleFlush(FlushCmd &)
    {
        return wal_->Flush();
    }

    SearchReply ShardRuntime::HandleSearch(SearchCmd &c)
    {
        SearchReply r;
        r.st = SearchLocalInternal(
            {c.query.data(), c.query.size()},
            c.topk,
            &r.hits);
        return r;
    }

    pomai::Status ShardRuntime::SearchLocalInternal(
        std::span<const float> query,
        std::uint32_t topk,
        std::vector<pomai::SearchHit> *out)
    {
        out->clear();
        out->reserve(topk);

        mem_->ForEach([&](VectorId id, std::span<const float> vec)
                      {
    float score = Dot(query, vec);

    if (out->size() < topk) {
      out->push_back({id, score});
      if (out->size() == topk) {
        std::make_heap(out->begin(), out->end(),
          [](auto& a, auto& b) { return a.score > b.score; });
      }
      return;
    }

    if (score <= (*out)[0].score) return;

    std::pop_heap(out->begin(), out->end(),
      [](auto& a, auto& b) { return a.score > b.score; });
    (*out)[topk - 1] = {id, score};
    std::push_heap(out->begin(), out->end(),
      [](auto& a, auto& b) { return a.score > b.score; }); });

        std::sort(out->begin(), out->end(),
                  [](auto &a, auto &b)
                  { return a.score > b.score; });

        return pomai::Status::Ok();
    }

} // namespace pomai::core
