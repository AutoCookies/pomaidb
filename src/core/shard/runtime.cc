#include "core/shard/runtime.h"
#include <filesystem>

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/distance.h"
#include "core/index/ivf_coarse.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"
#include "table/segment.h" // Added

namespace pomai::core
{

    namespace fs = std::filesystem; // Added

    ShardRuntime::ShardRuntime(std::uint32_t shard_id,
                               std::string shard_dir, // Added
                               std::uint32_t dim,
                               std::unique_ptr<storage::Wal> wal,
                               std::unique_ptr<table::MemTable> mem,
                               std::size_t mailbox_cap)
        : shard_id_(shard_id),
          shard_dir_(std::move(shard_dir)), // Added
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
        
        LoadSegments(); // Added

        worker_ = std::jthread([this]
                               { RunLoop(); });
        return pomai::Status::Ok();
    }

    void ShardRuntime::LoadSegments() // Added
    {
        // Scan shard_dir for files "seg_*.dat"
        // Since we don't have manifest tracking segments yet (A3 will add that),
        // we assume files on disk are valid segments to load.
        // A2 added "segments" line to membrane manifest? We didn't implement parsing it though?
        // Wait, A2 implemented WriteMembraneManifest with shards/dim/metric.
        // It did NOT add `segments` list to manifest yet.
        // So we rely on directory scan.
        
        if (!fs::exists(shard_dir_)) return;
        
        std::vector<std::string> seg_files;
        for (const auto& entry : fs::directory_iterator(shard_dir_)) {
             if (entry.is_regular_file()) {
                 std::string name = entry.path().filename().string();
                 if (name.rfind("seg_", 0) == 0 && name.ends_with(".dat")) {
                     seg_files.push_back(entry.path().string());
                 }
             }
        }
        
        // Sort files (e.g. seg_0.dat, seg_1.dat...)
        // Simple string sort works if fixed width, otherwise need number parsing.
        // Let's just sort strings for stability.
        std::sort(seg_files.begin(), seg_files.end());
        // Reverse order? Usually we search newest first.
        // WAL rotation produces increasing sequences.
        // Search: MemTable -> Newest Segment -> Oldest Segment.
        // So we want reverse order in `segments_`.
        // Sort ascending, then reverse? Or `blocks` are usually list of immutable components.
        // Let's store newest first.
        std::sort(seg_files.rbegin(), seg_files.rend());

        for (const auto& path : seg_files) {
            std::unique_ptr<table::SegmentReader> reader;
            auto st = table::SegmentReader::Open(path, &reader);
            if (st.ok()) {
                segments_.push_back(std::move(reader));
            } else {
                // Log warning?
                // For now ignore corrupt segments or fail?
                // Fail loud is better for DB.
                // But LoadSegments returns void.
                // We should probably log.
            }
        }
    }

    pomai::Status ShardRuntime::Enqueue(Command &&cmd)
    {
        if (!started_.load(std::memory_order_relaxed))
            return pomai::Status::Aborted("shard not started");
        if (!mailbox_.PushBlocking(std::move(cmd)))
            return pomai::Status::Aborted("mailbox closed");
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::TryEnqueue(Command &&cmd)
    {
        if (!started_.load(std::memory_order_relaxed))
            return pomai::Status::Aborted("shard not started");
        if (!mailbox_.TryPush(std::move(cmd)))
            return pomai::Status::ResourceExhausted("shard mailbox full");
        return pomai::Status::Ok();
    }

    // -------------------------
    // Sync wrappers
    // -------------------------

    pomai::Status ShardRuntime::Put(pomai::VectorId id, std::span<const float> vec)
    {
        if (vec.size() != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        PutCmd cmd;
        cmd.id = id;
        cmd.vec = vec.data(); 
        cmd.dim = static_cast<std::uint32_t>(vec.size());
        
        auto f = cmd.done.get_future();
        auto st = Enqueue(Command{std::move(cmd)});
        if (!st.ok())
            return st;
        return f.get();
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

    pomai::Status ShardRuntime::Get(pomai::VectorId id, std::vector<float> *out)
    {
        if (!out) return Status::InvalidArgument("out is null");
        GetCmd cmd;
        cmd.id = id;
        auto f = cmd.done.get_future();
        auto st = Enqueue(Command{std::move(cmd)});
        if (!st.ok()) return st; // e.g. mailbox closed
        auto reply = f.get();
        if (reply.st.ok())
        {
            *out = std::move(reply.vec);
        }
        return reply.st;
    }

    pomai::Status ShardRuntime::Exists(pomai::VectorId id, bool *exists)
    {
        if (!exists) return Status::InvalidArgument("exists is null");
        ExistsCmd cmd;
        cmd.id = id;
        auto f = cmd.done.get_future();
        auto st = Enqueue(Command{std::move(cmd)});
        if (!st.ok()) return st;
        auto reply = f.get();
        if (reply.first.ok())
        {
            *exists = reply.second;
        }
        return reply.first;
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
        bool stop_now = false;
        for (;;)
        {
            auto opt = mailbox_.PopBlocking();
            if (!opt.has_value())
                break;

            ops_processed_.fetch_add(1, std::memory_order_relaxed);

            Command cmd = std::move(*opt);

            std::visit(
                [&](auto &arg)
                {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, PutCmd>)
                    {
                        arg.done.set_value(HandlePut(arg));
                    }
                    else if constexpr (std::is_same_v<T, DelCmd>)
                    {
                        arg.done.set_value(HandleDel(arg));
                    }
                    else if constexpr (std::is_same_v<T, FlushCmd>)
                    {
                        arg.done.set_value(HandleFlush(arg));
                    }
                    else if constexpr (std::is_same_v<T, SearchCmd>)
                    {
                        arg.done.set_value(HandleSearch(arg));
                    }
                    else if constexpr (std::is_same_v<T, GetCmd>)
                    {
                        arg.done.set_value(HandleGet(arg));
                    }
                    else if constexpr (std::is_same_v<T, ExistsCmd>)
                    {
                        arg.done.set_value(HandleExists(arg));
                    }
                    else if constexpr (std::is_same_v<T, StopCmd>)
                    {
                        mailbox_.Close();
                        arg.done.set_value();
                        stop_now = true;
                    }
                },
                cmd);

            if (stop_now)
                break;
        }
    }

    // -------------------------
    // Handlers
    // -------------------------

    pomai::Status ShardRuntime::HandlePut(PutCmd &c)
    {
        if (c.dim != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        // 1. Write WAL
        auto st = wal_->AppendPut(c.id, {c.vec, c.dim});
        if (!st.ok())
            return st;

        // 2. Update MemTable
        return mem_->Put(c.id, {c.vec, c.dim});
    }

    GetReply ShardRuntime::HandleGet(GetCmd &c)
    {
        GetReply reply;
        const float *ptr = nullptr;
        // 1. Check MemTable
        auto st = mem_->Get(c.id, &ptr);
        if (st.ok() && ptr)
        {
            reply.st = Status::Ok();
            reply.vec.assign(ptr, ptr + dim_);
            return reply;
        }

        // 2. Check Segments
        for (const auto& seg : segments_) {
            std::span<const float> svec;
            st = seg->Get(c.id, &svec);
            if (st.ok()) {
                reply.st = Status::Ok();
                reply.vec.assign(svec.begin(), svec.end());
                return reply;
            }
        }

        reply.st = Status::NotFound("vector not found");
        return reply;
    }

    std::pair<pomai::Status, bool> ShardRuntime::HandleExists(ExistsCmd &c)
    {
        const float *ptr = nullptr;
        auto st = mem_->Get(c.id, &ptr);
        if (st.ok() && ptr) return {Status::Ok(), true};
        
        for (const auto& seg : segments_) {
            std::span<const float> svec;
            if (seg->Get(c.id, &svec).ok()) {
                return {Status::Ok(), true};
            }
        }
        return {Status::Ok(), false};
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

        out->reserve(topk);

        // Try IVF candidate selection.
        auto &candidates = candidates_scratch_;
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
            // Check MemTable
            for (pomai::VectorId id : candidates)
            {
                const float *ptr = mem_->GetPtr(id);
                if (ptr) {
                    float score = Dot(query, std::span<const float>(ptr, dim_));
                    push_topk(id, score);
                    continue;
                }
                
                // If not in memtable, check segments?
                // Single-point lookup for candidates is potentially slow if many candidates
                // but IVF candidates usually < 1000.
                // Segments are sorted, so O(logN). N segments.
                // Better than scanning all.
                for (const auto& seg : segments_) {
                    std::span<const float> svec;
                    if (seg->Get(id, &svec).ok()) {
                        float score = Dot(query, svec);
                        push_topk(id, score);
                        break; // Found (assume unique ID)
                    }
                }
            }
        }
        else
        {
            // Fallback brute-force: scan whole memtable (current behavior).
            mem_->ForEach([&](VectorId id, std::span<const float> vec)
                          {
                              float score = pomai::core::Dot(query, vec);
                              push_topk(id, score); });
                              
            // Scan Segments
            for (const auto& seg : segments_) {
                seg->ForEach([&](VectorId id, std::span<const float> vec) {
                    float score = pomai::core::Dot(query, vec);
                    push_topk(id, score);
                });
            }
        }

        std::sort(out->begin(), out->end(),
                  [](const auto &a, const auto &b)
                  { return a.score > b.score; });

        return pomai::Status::Ok();
    }

} // namespace pomai::core
