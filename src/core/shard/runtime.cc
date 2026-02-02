#include "core/shard/runtime.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/index/ivf_coarse.h"
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
          mailbox_(mailbox_cap)
    {
        pomai::index::IvfCoarse::Options opt;
        opt.nlist = 64;
        opt.nprobe = 4;
        opt.warmup = 256;
        opt.ema = 0.05f;
        ivf_ = std::make_unique<pomai::index::IvfCoarse>(dim_, opt);
    }

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

    // -------------------------
    // Sync wrappers
    // -------------------------

    pomai::Status ShardRuntime::Put(pomai::VectorId id, std::span<const float> vec)
    {
        if (vec.size() != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        PutCmd c;
        c.id = id;
        c.vec = vec.data();
        c.dim = static_cast<std::uint32_t>(vec.size());

        auto fut = c.done.get_future();
        auto st = Enqueue(Command{std::move(c)});
        if (!st.ok())
            return st;
        return fut.get();
    }

    pomai::Status ShardRuntime::Delete(pomai::VectorId id)
    {
        DelCmd c;
        c.id = id;
        auto fut = c.done.get_future();

        auto st = Enqueue(Command{std::move(c)});
        if (!st.ok())
            return st;
        return fut.get();
    }

    pomai::Status ShardRuntime::WriteBatch(const std::vector<pomai::WriteBatch::Op> &ops)
    {
        if (ops.empty())
            return pomai::Status::Ok();

        WriteBatchCmd c;
        c.ops = ops; // Copy ops into command
        auto fut = c.done.get_future();

        auto st = Enqueue(Command{std::move(c)});
        if (!st.ok())
            return st;
        return fut.get();
    }

    pomai::Status ShardRuntime::Flush()
    {
        FlushCmd c;
        auto fut = c.done.get_future();

        auto st = Enqueue(Command{std::move(c)});
        if (!st.ok())
            return st;
        return fut.get();
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

    // -------------------------
    // Actor loop
    // -------------------------

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
            if (auto *c = std::get_if<WriteBatchCmd>(&cmd))
            {
                c->done.set_value(HandleWriteBatch(*c));
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

    // -------------------------
    // Handlers
    // -------------------------

    pomai::Status ShardRuntime::HandlePut(PutCmd &c)
    {
        if (c.dim != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        auto st = wal_->AppendPut(c.id, {c.vec, c.dim});
        if (!st.ok())
            return st;

        st = mem_->Put(c.id, {c.vec, c.dim});
        if (!st.ok())
            return st;

        // Update IVF AFTER mem is updated (so candidates rerank pointer exists).
        (void)ivf_->Put(c.id, {c.vec, c.dim});
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::HandleDel(DelCmd &c)
    {
        auto st = wal_->AppendDelete(c.id);
        if (!st.ok())
            return st;

        st = mem_->Delete(c.id);
        if (!st.ok())
            return st;

        (void)ivf_->Delete(c.id);
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::HandleWriteBatch(WriteBatchCmd &c)
    {
        if (c.ops.empty())
            return pomai::Status::Ok();

        // Process each operation in order: WAL first, then memtable, then IVF
        for (const auto &op : c.ops)
        {
            if (op.type == pomai::WriteBatch::OpType::kPut)
            {
                if (op.vec.size() != dim_)
                    return pomai::Status::InvalidArgument("WriteBatch: dim mismatch for vector " + std::to_string(op.id));

                auto st = wal_->AppendPut(op.id, op.vec);
                if (!st.ok())
                    return st;

                st = mem_->Put(op.id, op.vec);
                if (!st.ok())
                    return st;

                (void)ivf_->Put(op.id, op.vec);
            }
            else if (op.type == pomai::WriteBatch::OpType::kDelete)
            {
                auto st = wal_->AppendDelete(op.id);
                if (!st.ok())
                    return st;

                st = mem_->Delete(op.id);
                if (!st.ok())
                    return st;

                (void)ivf_->Delete(op.id);
            }
            else
            {
                return pomai::Status::Internal("WriteBatch: unknown operation type");
            }
        }

        // Single fsync at the end if fsync policy requires it
        // (WAL's AppendPut/AppendDelete will handle fsync based on policy)
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::HandleFlush(FlushCmd &)
    {
        return wal_->Flush();
    }

    SearchReply ShardRuntime::HandleSearch(SearchCmd &c)
    {
        SearchReply r;
        r.st = SearchLocalInternal({c.query.data(), c.query.size()}, c.topk, &r.hits);
        return r;
    }

    // -------------------------
    // SearchLocalInternal: IVF-coarse
    // -------------------------

    pomai::Status ShardRuntime::SearchLocalInternal(std::span<const float> query,
                                                    std::uint32_t topk,
                                                    std::vector<pomai::SearchHit> *out)
    {
        out->clear();
        out->reserve(topk);

        // Try IVF candidate selection.
        std::vector<pomai::VectorId> candidates;
        auto st = ivf_->SelectCandidates(query, &candidates);
        if (!st.ok())
            return st;

        auto push_topk = [&](pomai::VectorId id, float score)
        {
            if (out->size() < topk)
            {
                out->push_back({id, score});
                if (out->size() == topk)
                {
                    std::make_heap(out->begin(), out->end(),
                                   [](const auto &a, const auto &b)
                                   { return a.score > b.score; }); // min-heap by score
                }
                return;
            }

            if (score <= (*out)[0].score)
                return;

            std::pop_heap(out->begin(), out->end(),
                          [](const auto &a, const auto &b)
                          { return a.score > b.score; });
            (*out)[topk - 1] = {id, score};
            std::push_heap(out->begin(), out->end(),
                           [](const auto &a, const auto &b)
                           { return a.score > b.score; });
        };

        if (!candidates.empty())
        {
            // IVF path: rerank candidates only.
            for (pomai::VectorId id : candidates)
            {
                const float *ptr = mem_->GetPtr(id);
                if (!ptr)
                    continue;
                float score = Dot(query, std::span<const float>(ptr, dim_));
                push_topk(id, score);
            }
        }
        else
        {
            // Fallback brute-force: scan whole memtable (current behavior).
            mem_->ForEach([&](VectorId id, std::span<const float> vec)
                          {
                          float score = Dot(query, vec);
                          push_topk(id, score); });
        }

        std::sort(out->begin(), out->end(),
                  [](const auto &a, const auto &b)
                  { return a.score > b.score; });

        return pomai::Status::Ok();
    }

} // namespace pomai::core
