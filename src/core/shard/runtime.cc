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
#include "core/shard/manifest.h"
#include <queue>

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
        
        auto st = LoadSegments();
        if (!st.ok()) return st;

        worker_ = std::jthread([this]
                               { RunLoop(); });
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::LoadSegments()
    {
        std::vector<std::string> seg_names;
        auto st = ShardManifest::Load(shard_dir_, &seg_names);
        if (!st.ok()) return st;
        
        segments_.clear();
        for (const auto& name : seg_names) {
            std::string path = (fs::path(shard_dir_) / name).string();
            std::unique_ptr<table::SegmentReader> reader;
            st = table::SegmentReader::Open(path, &reader);
            if (!st.ok()) return st;
            segments_.push_back(std::move(reader));
        }
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

    pomai::Status ShardRuntime::Freeze()
    {
        FreezeCmd c;
        auto f = c.done.get_future();
        auto st = Enqueue(Command{std::move(c)});
        if (!st.ok()) return st;
        return f.get();
    }

    pomai::Status ShardRuntime::Compact()
    {
        CompactCmd c;
        auto f = c.done.get_future();
        auto st = Enqueue(Command{std::move(c)});
        if (!st.ok()) return st;
        return f.get();
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
                    else if constexpr (std::is_same_v<T, FreezeCmd>)
                    {
                        arg.done.set_value(HandleFreeze(arg));
                    }
                    else if constexpr (std::is_same_v<T, CompactCmd>)
                    {
                        arg.done.set_value(HandleCompact(arg));
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
            // Use Find (handles tombstones)
            auto res = seg->Find(c.id, &svec);
            if (res == table::SegmentReader::FindResult::kFound) {
                reply.st = Status::Ok();
                reply.vec.assign(svec.begin(), svec.end());
                return reply;
            } else if (res == table::SegmentReader::FindResult::kFoundTombstone) {
                reply.st = Status::NotFound("tombstone");
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
            auto res = seg->Find(c.id, &svec);
            if (res == table::SegmentReader::FindResult::kFound) {
                return {Status::Ok(), true};
            } else if (res == table::SegmentReader::FindResult::kFoundTombstone) {
                // If found tombstone in NEWER segment, it is deleted.
                return {Status::Ok(), false};
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

    pomai::Status ShardRuntime::HandleFreeze(FreezeCmd &)
    {
        if (mem_->GetCount() == 0) return pomai::Status::Ok();

        // 1. Build Segment
        auto now = std::chrono::steady_clock::now().time_since_epoch().count(); // Monotonic enough for local?
        // Use system clock for filename readability?
        auto sys_now = std::chrono::system_clock::now().time_since_epoch().count();
        std::string name = "seg_" + std::to_string(sys_now) + ".dat";
        std::string path = (fs::path(shard_dir_) / name).string();

        table::SegmentBuilder builder(path, dim_);
        
        // mem_->IterateWithStatus
        mem_->IterateWithStatus([&](VectorId id, std::span<const float> vec, bool is_deleted) {
             // We can ignore deleted if we are sure no older segments exist?
             // But safely we should persist tombstones.
             (void)builder.Add(id, vec, is_deleted);
        });

        auto st = builder.Finish();
        if (!st.ok()) return st;

        // 2. Update Manifest
        std::vector<std::string> seg_names;
        st = ShardManifest::Load(shard_dir_, &seg_names);
        if (!st.ok()) return st; // Should return empty if new
        
        // Prepend (Assuming Newest First).
        // Manifest order: Newest First?
        // Let's decide: Manifest = [Newest, ..., Oldest]
        seg_names.insert(seg_names.begin(), name);
        
        st = ShardManifest::Commit(shard_dir_, seg_names);
        if (!st.ok()) return st;

        // 3. Clear MemTable & Reset WAL
        mem_->Clear();
        st = wal_->Reset();
        if (!st.ok()) return st; 
        
        // 4. Reload in-memory
        std::unique_ptr<table::SegmentReader> reader;
        st = table::SegmentReader::Open(path, &reader);
        if (!st.ok()) return st;
        segments_.insert(segments_.begin(), std::move(reader));

        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::HandleCompact(CompactCmd &)
    {
        if (segments_.empty()) return pomai::Status::Ok();

        // 1. Prepare Output
        auto sys_now = std::chrono::system_clock::now().time_since_epoch().count();
        std::string name = "seg_" + std::to_string(sys_now) + "_compacted.dat";
        std::string path = (fs::path(shard_dir_) / name).string();

        table::SegmentBuilder builder(path, dim_);

        // 2. K-way Merge
        struct Cursor {
            VectorId id;
            uint32_t seg_idx;
            uint32_t entry_idx;
            bool is_deleted;
            
            // Priority Queue (Min-Heap): Smallest ID first, then Smallest SegIdx (Newest)
            bool operator>(const Cursor& other) const {
                if (id != other.id) return id > other.id;
                return seg_idx > other.seg_idx;
            }
        };

        std::priority_queue<Cursor, std::vector<Cursor>, std::greater<Cursor>> heap;

        // Initialize cursors
        for (uint32_t i = 0; i < segments_.size(); ++i) {
            VectorId id;
            bool del;
            // Read first entry
            if (segments_[i]->ReadAt(0, &id, nullptr, &del).ok()) {
                heap.push({id, i, 0, del});
            }
        }

        VectorId last_id = std::numeric_limits<VectorId>::max(); // Using max as sentinel for "none"?
        bool is_first = true;

        while (!heap.empty()) {
            Cursor top = heap.top();
            heap.pop();
            
            // Check if this ID is new
            if (is_first || top.id != last_id) {
                // This is the winning version (newest due to secondary sort order)
                
                // If it is NOT deleted, add it.
                // If it IS deleted, we can drop it because we are compacting ALL segments (full compaction).
                // So no older version exists that needs shadowing.
                if (!top.is_deleted) {
                    // Need to read vector data now
                    std::span<const float> vec;
                    // We need to read again from segments_[top.seg_idx] at top.entry_idx?
                    // We didn't store vec in Cursor to save heap space/copy.
                    // ReadAt again.
                    if (segments_[top.seg_idx]->ReadAt(top.entry_idx, nullptr, &vec, nullptr).ok()) {
                        builder.Add(top.id, vec, false);
                    }
                }
                
                last_id = top.id;
                is_first = false;
            }
            // Else: this is an older version of same ID (shadowed). Ignore.

            // Advance cursor
            uint32_t next_idx = top.entry_idx + 1;
            VectorId next_id;
            bool next_del;
            if (segments_[top.seg_idx]->ReadAt(next_idx, &next_id, nullptr, &next_del).ok()) {
                heap.push({next_id, top.seg_idx, next_idx, next_del});
            }
        }

        auto st = builder.Finish();
        if (!st.ok()) return st;
        
        // 3. Update Manifest
        std::vector<std::string> seg_names;
        // The new list will contain ONLY the new segment.
        seg_names.push_back(name);
        
        st = ShardManifest::Commit(shard_dir_, seg_names);
        if (!st.ok()) return st;

        // 4. Reload in-memory
        // Clear old segments logic?
        // Delete input files?
        // We should delete old files AFTER commit.
        // Copy old segments to delete them later?
        std::vector<std::shared_ptr<table::SegmentReader>> old_segments = std::move(segments_);
        segments_.clear();

        // Load new
        std::unique_ptr<table::SegmentReader> reader;
        st = table::SegmentReader::Open(path, &reader);
        if (!st.ok()) return st;
        segments_.push_back(std::move(reader));
        
        // 5. Delete old files
        for (const auto& old : old_segments) {
             std::error_code ec;
             fs::remove(old->Path(), ec);
        }

        return pomai::Status::Ok();
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
                
                // Check segments
                for (const auto& seg : segments_) {
                    std::span<const float> svec;
                    auto res = seg->Find(id, &svec);
                    if (res == table::SegmentReader::FindResult::kFound) {
                        float score = Dot(query, svec);
                        push_topk(id, score);
                        break; // Found
                    }
                    if (res == table::SegmentReader::FindResult::kFoundTombstone) {
                        break; // Deleted
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
                seg->ForEach([&](VectorId id, std::span<const float> vec, bool is_deleted) {
                    if (is_deleted) return;
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


