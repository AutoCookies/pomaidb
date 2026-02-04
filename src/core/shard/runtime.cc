#include "core/shard/runtime.h"
#include <filesystem>
#include <cassert>
#include <unordered_set>

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/distance.h"
#include "core/index/ivf_coarse.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"
#include "table/segment.h" // Added
#include "core/shard/iterator.h"
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
                               std::size_t mailbox_cap,
                               const pomai::IndexParams& index_params,
                               pomai::util::ThreadPool* thread_pool,
                               pomai::util::ThreadPool* segment_pool)
        : shard_id_(shard_id),
          shard_dir_(std::move(shard_dir)), // Added
          dim_(dim),
          wal_(std::move(wal)),
          mem_(std::move(mem)),
          mailbox_(mailbox_cap),
          thread_pool_(thread_pool),
          segment_pool_(segment_pool),
          index_params_(index_params)
    {
        pomai::index::IvfCoarse::Options opt;
        opt.nlist = index_params_.nlist;
        opt.nprobe = index_params_.nprobe;
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
        
        // If we have replayed data in MemTable, rotate it to Frozen so it's visible in Snapshot.
        if (mem_->GetCount() > 0) {
            (void)RotateMemTable();
        }

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

        PublishSnapshot();
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
    // Snapshot & Rotation
    // -------------------------
    // -------------------------
    // Snapshot & Rotation
    // -------------------------
    void ShardRuntime::PublishSnapshot()
    {
        auto snap = std::make_shared<ShardSnapshot>();
        snap->version = next_snapshot_version_++;
        snap->created_at = std::chrono::steady_clock::now();
        
        // Copy atomic/shared state
        snap->segments = segments_; // Shared ownership of segments
        snap->frozen_memtables = frozen_mem_; // Shared ownership of frozen tables
        snap->mem = mem_.get(); // Active memtable (raw ptr, owned by ShardRuntime)
        
        // INVARIANT: All frozen memtables are immutable (count fixed)
        for (const auto& fmem : snap->frozen_memtables) {
            // usage count might be > 1 (snapshot + frozen_mem_) or more if multiple snapshots
            assert(fmem.use_count() >= 2); 
            (void)fmem;
        }

        // INVARIANT: All segments are immutable (read-only)
        for (const auto& seg : snap->segments) {
            assert(seg.use_count() >= 2);
            (void)seg;
        }

        current_snapshot_.store(snap, std::memory_order_release);
    }

    pomai::Status ShardRuntime::RotateMemTable()
    {
        // Move mutable mem_ to frozen_mem_
        if (mem_->GetCount() == 0) return pomai::Status::Ok();
        
        // Transfer ownership from unique_ptr to shared_ptr
        std::shared_ptr<table::MemTable> old_mem = std::move(mem_);
        frozen_mem_.push_back(old_mem);
        
        // Allocate new MemTable
        // We reuse the same arena block size constant? Need to expose it or hardcode.
        // engine.cc uses kArenaBlockBytes = 1MB. We should probably pass it or use default.
        // mem_->ArenaBlockBytes? No getter.
        // Let's assume 1MB.
        mem_ = std::make_unique<table::MemTable>(dim_, 1u << 20);
        
        PublishSnapshot();
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

    pomai::Status ShardRuntime::PutBatch(const std::vector<pomai::VectorId>& ids,
                                          const std::vector<std::span<const float>>& vectors)
    {
        // Validation
        if (ids.size() != vectors.size())
            return pomai::Status::InvalidArgument("ids and vectors size mismatch");
        if (ids.empty())
            return pomai::Status::Ok();
        
        // Validate dimensions
        for (const auto& vec : vectors) {
            if (vec.size() != dim_)
                return pomai::Status::InvalidArgument("dim mismatch");
        }
        
        // Deep copy vectors into command (avoid lifetime issues)
        BatchPutCmd cmd;
        cmd.ids = ids;
        cmd.vectors.reserve(vectors.size());
        for (const auto& vec : vectors) {
            cmd.vectors.emplace_back(vec.begin(), vec.end());
        }
        
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
        auto snap = GetSnapshot();
        if (!snap) return Status::Aborted("shard not ready");
        return GetFromSnapshot(snap, id, out);
    }

    pomai::Status ShardRuntime::Exists(pomai::VectorId id, bool *exists)
    {
        if (!exists) return Status::InvalidArgument("exists is null");
        auto snap = GetSnapshot();
        if (!snap) return Status::Aborted("shard not ready");
        auto res = ExistsInSnapshot(snap, id);
        if (res.first.ok()) {
            *exists = res.second;
        }
        return res.first;
    }

    pomai::Status ShardRuntime::GetFromSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id, std::vector<float> *out) {
        // 0. Check Active MemTable
        if (snap->mem) {
            const float* p = snap->mem->GetPtr(id);
            if (p) {
                out->assign(p, p + dim_);
                return Status::Ok();
            }
            if (snap->mem->IsTombstone(id)) {
                return Status::NotFound("tombstone"); 
            }
        }

        // 1. Check Frozen MemTables (Newest -> Oldest)
        
        for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
            if ((*it)->IsTombstone(id)) {
                return Status::NotFound("tombstone");
            }
            const float* p = (*it)->GetPtr(id);
            if (p) {
                out->assign(p, p + dim_);
                return Status::Ok();
            }
        }

        // 2. Check Segments (Newest -> Oldest)
        // segments_ is [Newest, ..., Oldest] (based on HandleFreeze insert(begin))
        
        for (auto it = snap->segments.begin(); it != snap->segments.end(); ++it) {
            std::span<const float> svec;
            auto res = (*it)->Find(id, &svec);
            if (res == table::SegmentReader::FindResult::kFound) {
                out->assign(svec.begin(), svec.end());
                return Status::Ok();
            } else if (res == table::SegmentReader::FindResult::kFoundTombstone) {
                return Status::NotFound("tombstone");
            }
        }
        
        return Status::NotFound("vector not found");
    }

    std::pair<pomai::Status, bool> ShardRuntime::ExistsInSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id) {
        // 1. Check Frozen (Newest -> Oldest)
        for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
             if ((*it)->IsTombstone(id)) return {Status::Ok(), false};
             if ((*it)->GetPtr(id)) return {Status::Ok(), true};
        }

        // 2. Check Segments (Newest -> Oldest)
        for (auto it = snap->segments.begin(); it != snap->segments.end(); ++it) {
            std::span<const float> svec;
            auto res = (*it)->Find(id, &svec);
             if (res == table::SegmentReader::FindResult::kFound) {
                return {Status::Ok(), true};
            } else if (res == table::SegmentReader::FindResult::kFoundTombstone) {
                return {Status::Ok(), false};
            }
        }
        return {Status::Ok(), false};
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

    pomai::Status ShardRuntime::NewIterator(std::unique_ptr<pomai::SnapshotIterator>* out)
    {
        IteratorCmd cmd;
        auto f = cmd.done.get_future();
        auto st = Enqueue(Command{std::move(cmd)});
        if (!st.ok())
            return st;
        
        IteratorReply reply = f.get();
        if (!reply.st.ok())
            return reply.st;
        
        *out = std::move(reply.iterator);
        return pomai::Status::Ok();
    }

    // LOCK-FREE SEARCH
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

        // Get Snapshot
        auto snap = GetSnapshot();
        if (!snap) return pomai::Status::Aborted("shard not ready");

        // Local Search using Snapshot (Bypass Mailbox)
        return SearchLocalInternal(snap, query, topk, out);
    }

    // -------------------------
    // Actor loop
    // -------------------------

    void ShardRuntime::RunLoop()
    {
        // Ensure cleanup on exit (exception or normal)
        struct ScopeGuard {
            ShardRuntime* rt;
            ~ScopeGuard() {
                rt->mailbox_.Close();
                rt->started_.store(false);
            }
        } guard{this};

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
                    else if constexpr (std::is_same_v<T, BatchPutCmd>)
                    {
                        arg.done.set_value(HandleBatchPut(arg));
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
                    else if constexpr (std::is_same_v<T, IteratorCmd>)
                    {
                        arg.done.set_value(HandleIterator(arg));
                    }
                    else if constexpr (std::is_same_v<T, SearchCmd>)
                    {
                        arg.done.set_value(HandleSearch(arg));
                    }
                    else if constexpr (std::is_same_v<T, StopCmd>)
                    {
                        // Mailbox close handled by ScopeGuard or manual?
                        // If we Close here, PopBlocking next loop returns nullopt.
                        // But we want to break immediately.
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
        st = mem_->Put(c.id, {c.vec, c.dim});
        if (!st.ok()) return st;

        // 3. Check Threshold for Soft Freeze (e.g. 5000 items)
        // Hardcoding 5000 for now as per plan
        if (mem_->GetCount() >= 5000) {
            (void)RotateMemTable();
        }
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::HandleBatchPut(BatchPutCmd &c)
    {
        // Validation (already done in PutBatch, but belt-and-suspenders)
        if (c.ids.size() != c.vectors.size())
            return pomai::Status::InvalidArgument("ids and vectors size mismatch");
        
        // Convert owned vectors to spans for WAL/MemTable
        std::vector<std::span<const float>> spans;
        spans.reserve(c.vectors.size());
        for (const auto& vec : c.vectors) {
            spans.emplace_back(vec);
        }
        
        // 1. Batch write to WAL (KEY OPTIMIZATION: single fsync)
        auto st = wal_->AppendBatch(c.ids, spans);
        if (!st.ok())
            return st;
        
        // 2. Batch update MemTable
        st = mem_->PutBatch(c.ids, spans);
        if (!st.ok())
            return st;
        
        // 3. Check threshold for soft freeze (same as single Put)
        if (mem_->GetCount() >= 5000) {
            (void)RotateMemTable();
        }
        
        return pomai::Status::Ok();
    }
    
    // HandleGet and HandleExists removed.

    pomai::Status ShardRuntime::HandleDel(DelCmd &c)
    {
        auto st = wal_->AppendDelete(c.id);
        if (!st.ok())
            return st;

        st = mem_->Delete(c.id);
        if (!st.ok())
            return st;

        // Need to mark delete in frozen?
        // Frozen memtables are immutable. We can't delete in them.
        // We add a "Delete" record to active memtable (tombstone).
        // Since we search Active AFTER Frozen? No, we search Newer first.
        // Order: Active -> Frozen (New->Old) -> Segments.
        // But `Search` using `Snapshot` does NOT see Active.
        // So `Search` will see the OLD value in Frozen/Segments if Active has Tombstone.
        // This is STALENESS. "Reads may observe a slightly stale snapshot".
        // This is consistent.
        
        (void)ivf_->Delete(c.id);
        return pomai::Status::Ok();
    }

    pomai::Status ShardRuntime::HandleFlush(FlushCmd &)
    {
        return wal_->Flush();
    }

    // -------------------------
    // HandleFreeze: Atomic Freeze Pipeline (DB-Grade Durability)
    // -------------------------
    // Guarantees:
    // 1. All segments built and fsynced BEFORE manifest commit
    // 2. Manifest committed ONCE (atomically)
    // 3. WAL reset ONLY after all above succeed
    // 4. Directory fsynced at each durability boundary
    //
    // Crash safety: If crash at any point, recovery sees:
    // - Either old manifest (if crash before manifest commit) → segments ignored, WAL replayed
    // - Or new manifest (if crash after) → segments visible, WAL safe to reset

    pomai::Status ShardRuntime::HandleFreeze(FreezeCmd &)
    {
        // Step 1: Rotate Active → Frozen (idempotent if already empty)
        if (mem_->GetCount() > 0) {
            auto st = RotateMemTable();
            if (!st.ok()) {
                return pomai::Status::Internal("Freeze: RotateMemTable failed: " + st.message());
            }
        }
        
        if (frozen_mem_.empty()) {
            return pomai::Status::Ok(); // Nothing to freeze
        }

        // Step 2: Build ALL segments FIRST (no manifest changes yet)
        struct BuiltSegment {
            std::string filename;
            std::string filepath;
            std::shared_ptr<table::SegmentReader> reader;
        };
        std::vector<BuiltSegment> built_segments;
        built_segments.reserve(frozen_mem_.size());
        
        for (auto& fmem : frozen_mem_) {
            if (fmem->GetCount() == 0) continue; // Skip empty frozen tables
            
            // Generate deterministic filename (timestamp-based, unique)
            auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            std::string filename = "seg_" + std::to_string(now) + "_" + 
                                   std::to_string(reinterpret_cast<uint64_t>(fmem.get())) + ".dat";
            std::string filepath = (fs::path(shard_dir_) / filename).string();
            
            // Build segment to disk
            table::SegmentBuilder builder(filepath, dim_);
            fmem->IterateWithStatus([&](VectorId id, std::span<const float> vec, bool is_deleted) {
                (void)builder.Add(id, vec, is_deleted);
            });
            
            // Build Sidecar Index
            auto st = builder.BuildIndex(index_params_.nlist);
            if (!st.ok()) {
                 return pomai::Status::Internal("Freeze: BuildIndex failed: " + st.message());
            }

            auto st_finish = builder.Finish();
            if (!st_finish.ok()) {
                return pomai::Status::Internal("Freeze: SegmentBuilder::Finish failed: " + st_finish.message());
            }
            
            // Fsync directory after segment file creation (durability boundary)
            st = pomai::util::FsyncDir(shard_dir_);
            if (!st.ok()) {
                return pomai::Status::Internal("Freeze: FsyncDir after segment failed: " + st.message());
            }
            
            // Open reader (will be added to segments_ later, after manifest commit)
            std::unique_ptr<table::SegmentReader> reader;
            st = table::SegmentReader::Open(filepath, &reader);
            if (!st.ok()) {
                return pomai::Status::Internal("Freeze: SegmentReader::Open failed: " + st.message());
            }
            
            built_segments.push_back({filename, filepath, std::move(reader)});
        }

        // Step 3: Commit manifest ATOMICALLY (single write)
        // Load existing manifest
        std::vector<std::string> seg_names;
        auto st = ShardManifest::Load(shard_dir_, &seg_names);
        if (!st.ok()) {
            return pomai::Status::Internal("Freeze: ShardManifest::Load failed: " + st.message());
        }
        
        // Prepend new segments (newest first)
        for (const auto& bs : built_segments) {
            seg_names.insert(seg_names.begin(), bs.filename);
        }
        
        // Atomic commit (write manifest.new → fsync → rename → fsync dir)
        st = ShardManifest::Commit(shard_dir_, seg_names);
        if (!st.ok()) {
            return pomai::Status::Internal("Freeze: ShardManifest::Commit failed: " + st.message());
        }
        
        // Step 4: Update in-memory state (only after manifest persisted)
        for (auto& bs : built_segments) {
            segments_.insert(segments_.begin(), std::move(bs.reader));
        }
        
        // Step 5: Clear frozen memtables and reset WAL (only after all above succeed)
        frozen_mem_.clear();
        st = wal_->Reset();
        if (!st.ok()) {
            // WAL reset failed, but segments are durable. Log and continue?
            // Or return error? For now, return error.
            return pomai::Status::Internal("Freeze: WAL::Reset failed: " + st.message());
        }
        
        // Step 6: Publish new snapshot (make segments visible to readers)
        PublishSnapshot();

        return pomai::Status::Ok();
    }

    // -------------------------
    // HandleCompact: DB-Grade Compaction (Tombstone Purging)
    // -------------------------
    // "Database Moat": Tombstones protect against resurrection in newer segments,
    // but compaction PURGES them to reduce read amplification.
    //
    // Strategy:
    // 1. K-way merge (newest → oldest)
    // 2. Keep ONLY newest version per ID
    // 3. PURGE tombstones entirely (don't write to compacted segment)
    // 4. Result: Compacted segment contains ONLY live data
    
    pomai::Status ShardRuntime::HandleCompact(CompactCmd &c)
    {
        (void)c;
        if (segments_.empty()) return pomai::Status::Ok();

        // Step 1: Prepare output segment
        auto sys_now = std::chrono::system_clock::now().time_since_epoch().count();
        std::string name = "seg_" + std::to_string(sys_now) + "_compacted.dat";
        std::string path = (fs::path(shard_dir_) / name).string();

        table::SegmentBuilder builder(path, dim_);

        // Step 2: K-way merge (newest → oldest)
        struct Cursor {
            VectorId id;
            uint32_t seg_idx;
            uint32_t entry_idx;
            bool is_deleted;
            
            // Min-heap by (ID, segment_index)
            // Smaller ID first; for same ID, smaller seg_idx (newer) first
            bool operator>(const Cursor& other) const {
                if (id != other.id) return id > other.id;
                return seg_idx > other.seg_idx;  // Newer segments have smaller index
            }
        };

        std::priority_queue<Cursor, std::vector<Cursor>, std::greater<Cursor>> heap;
        
        // Initialize heap with first entry from each segment
        for (uint32_t i = 0; i < segments_.size(); ++i) {
            VectorId id;
            bool del;
            if (segments_[i]->ReadAt(0, &id, nullptr, &del).ok()) {
                heap.push({id, i, 0, del});
            }
        }

        VectorId last_id = std::numeric_limits<VectorId>::max();
        bool is_first = true;
        
        uint64_t total_entries_scanned = 0;
        uint64_t tombstones_purged = 0;
        uint64_t old_versions_dropped = 0;
        uint64_t live_entries_kept = 0;

        // Step 3: Merge and purge
        while (!heap.empty()) {
            Cursor top = heap.top();
            heap.pop();
            total_entries_scanned++;
            
            // Process only newest version of each ID
            if (is_first || top.id != last_id) {
                // This is the newest version of this ID
                
                if (top.is_deleted) {
                    // ✅ PURGE TOMBSTONE (don't write to compacted segment)
                    tombstones_purged++;
                } else {
                    // ✅ KEEP LIVE DATA (newest version only)
                    std::span<const float> vec;
                    if (segments_[top.seg_idx]->ReadAt(top.entry_idx, nullptr, &vec, nullptr).ok()) {
                        builder.Add(top.id, vec, false);
                        live_entries_kept++;
                    }
                }
                
                last_id = top.id;
                is_first = false;
            } else {
                // ✅ DROP OLD VERSION (already processed newest)
                old_versions_dropped++;
            }
            
            // Advance cursor in this segment
            uint32_t next_idx = top.entry_idx + 1;
            VectorId next_id;
            bool next_del;
            if (segments_[top.seg_idx]->ReadAt(next_idx, &next_id, nullptr, &next_del).ok()) {
                heap.push({next_id, top.seg_idx, next_idx, next_del});
            }
        }

        // Step 4: Finalize compacted segment
        auto st = builder.BuildIndex(index_params_.nlist); 
        if (!st.ok()) {
             return pomai::Status::Internal("Compact: BuildIndex failed: " + st.message());
        }

        st = builder.Finish();
        if (!st.ok()) {
            return pomai::Status::Internal("Compact: SegmentBuilder::Finish failed: " + st.message());
        }
        
        // Fsync directory after segment creation
        st = pomai::util::FsyncDir(shard_dir_);
        if (!st.ok()) {
            return pomai::Status::Internal("Compact: FsyncDir after segment failed: " + st.message());
        }

        // Step 5: Atomic manifest update (replace all old segments with new one)
        std::vector<std::string> seg_names;
        seg_names.push_back(name);
        st = ShardManifest::Commit(shard_dir_, seg_names);
        if (!st.ok()) {
            return pomai::Status::Internal("Compact: ShardManifest::Commit failed: " + st.message());
        }

        // Step 6: Update in-memory state (swap segments)
        std::vector<std::shared_ptr<table::SegmentReader>> old_segments = std::move(segments_);
        segments_.clear();

        std::unique_ptr<table::SegmentReader> reader;
        st = table::SegmentReader::Open(path, &reader);
        if (!st.ok()) {
            return pomai::Status::Internal("Compact: SegmentReader::Open failed: " + st.message());
        }
        segments_.push_back(std::move(reader));
        
        // Step 7: Delete old segment files
        for (const auto& old : old_segments) {
            std::error_code ec;
            fs::remove(old->Path(), ec);
            // Ignore errors (best-effort cleanup)
        }
        
        // Step 8: Publish new snapshot
        PublishSnapshot();
        
        // TODO: Log compaction metrics
        // std::cerr << "Compaction: scanned=" << total_entries_scanned 
        //           << " purged=" << tombstones_purged 
        //           << " dropped=" << old_versions_dropped 
        //           << " kept=" << live_entries_kept << "\n";

        return pomai::Status::Ok();
    }

    IteratorReply ShardRuntime::HandleIterator(IteratorCmd &c)
    {
        (void)c;  // unused parameter
        
        // Create iterator with current snapshot (point-in-time view)
        auto snapshot = current_snapshot_.load();
        
        if (!snapshot) {
            IteratorReply reply;
            reply.st = pomai::Status::Internal("HandleIterator: snapshot is null");
            return reply;
        }
        
        // Create ShardIterator
        auto shard_iter = std::make_unique<ShardIterator>(snapshot);
        
        IteratorReply reply;
        reply.st = pomai::Status::Ok();
        reply.iterator = std::move(shard_iter);
        return reply;
    }

    SearchReply ShardRuntime::HandleSearch(SearchCmd &c)
    {
        // Legacy fallback
        SearchReply r;
        // Use internal helper but we need to create a snapshot from current state manually?
        // Or just use the Atomic Snapshot.
        // It's safe to use atomic snapshot even from Writer thread.
        auto snap = GetSnapshot();
        if(snap) {
             // We need to pass the snap to SearchLocalInternal
             // But SearchLocalInternal needs to be updated.
             // I'll update SearchLocalInternal signature below.
             r.st = SearchLocalInternal(snap, {c.query.data(), c.query.size()}, c.topk, &r.hits);
        } else {
            r.st = Status::Aborted("no snapshot");
        }
        return r;
    }

// -------------------------
    // SearchLocalInternal: DB-grade 1-pass merge scan
    // -------------------------

    pomai::Status ShardRuntime::SearchLocalInternal(
            std::shared_ptr<ShardSnapshot> snap,
            std::span<const float> query,
            std::uint32_t topk,
            std::vector<pomai::SearchHit> *out)
    {
        out->clear();
        out->reserve(topk);

        // Begin new search iteration (resets seen tracker via generation increment)
        seen_tracker_.BeginSearch();

        // Inline top-K heap maintenance
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

        // -------------------------
        // 0. SCAN Active MemTable (Newest)
        // -------------------------
        // We must check Active MemTable before Frozen/Segments.
        // It uses shared_lock internal to IterateWithStatus, so safe to call concurrently.
        if (snap->mem) {
            snap->mem->IterateWithStatus([&](VectorId id, std::span<const float> vec, bool is_deleted) {
                 if (seen_tracker_.Contains(id)) return;
                 seen_tracker_.MarkSeen(id); 

                 if (is_deleted) {
                     seen_tracker_.MarkTombstone(id);
                     return;
                 }

                 float score = pomai::core::Dot(query, vec);
                 push_topk(id, score);
            });
        }

        // -------------------------
        // 1-PASS MERGE SCAN: Frozen MemTables (newest → oldest)
        // -------------------------
        // Process each ID exactly once. If ID seen before, skip.
        // If tombstone, mark and skip. Otherwise, score and push to heap.
        
        for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
            (*it)->IterateWithStatus([&](VectorId id, std::span<const float> vec, bool is_deleted) {
                // Skip if already processed (newest-wins)
                if (seen_tracker_.Contains(id)) return;
                
                // Mark as seen
                seen_tracker_.MarkSeen(id);
                
                // If tombstone, skip scoring
                if (is_deleted) {
                    seen_tracker_.MarkTombstone(id);
                    return;
                }
                
                // Score and push to top-K
                float score = pomai::core::Dot(query, vec);
                push_topk(id, score);
            });
        }

        // -------------------------
        // 1-PASS MERGE SCAN: Segments (Newest -> Oldest)
        // -------------------------
        
        uint32_t nprobe = index_params_.nprobe; 
        
        struct SegmentResult {
            std::vector<VectorId> candidates;
            std::vector<std::pair<VectorId, float>> hits;
        };

        std::vector<std::future<SegmentResult>> futures;
        std::vector<SegmentResult> inline_results;
        
        // Group segments into tasks implies we iterate them.
        // Segments are already ordered roughly newest to oldest. 
        // We can just parallelize all segments if pool is available.
        
        // Lambda to search a single segment (or batch)
        auto search_segment = [&](std::shared_ptr<table::SegmentReader> seg) -> SegmentResult {
            SegmentResult res;
            if (seg->HasIndex()) {
                // IVF Path
                std::vector<VectorId> candidates;
                Status st = seg->Search(query, nprobe, &candidates);
                if (st.ok()) {
                     res.candidates = std::move(candidates);
                }
            } else {
                // Brute force scan inside segment
                // This might return A LOT of results locally, we probably want top-K per segment?
                // Or just scan and return hits?
                // The main SearchLocalInternal merges everything.
                // For simplicity in this phase, let's just stick to IVF-only parallelization logic 
                // or just do candidates gathering.
                // BRUTE FORCE SCAN IS SLOW.
                // If we parallelize brute force, we should return top-K from that segment.
                // Let's implement full parallel support.
                
                seg->ForEach([&](VectorId id, std::span<const float> vec, bool is_deleted) {
                     if (is_deleted) return; // Wait, we need to handle tombstones globally? 
                     // No, "Delete" in segment means it's a tombstone for OLDER segments?
                     // SegmentReader::ForEach returns is_deleted flag.
                     // If it is_deleted, it IS a tombstone.
                     // So we should return it as a tombstone hit?
                     // Or just ignore it?
                     // The SeenTracker logic handles tombstones.
                     // But SeenTracker is not thread-safe.
                     // So parallel tasks CANNOT touch SeenTracker.
                     // They must return raw hits/tombstones, and main thread merges.
                     
                     float score = pomai::core::Dot(query, vec);
                     // Just collect ALL hits? Too expensive if segment is large.
                     // Collect Top-K locally? Yes.
                     // But we also need Tombstones to hide older versions.
                     // If we see a tombstone, we must report it.
                     res.hits.push_back({id, is_deleted ? -std::numeric_limits<float>::infinity() : score});
                     // Wait, how to distinguish tombstone in pair?
                     // We need a better struct.
                });
            }
            return res;
        };
        
        // NOTE: SeenTracker is NOT thread-safe.
        // Parallel Strategy:
        // 1. Scatter: Search segments in parallel. 
        //    Each task returns Candidates (IVF) or Top-K Hits (Brute Force).
        // 2. Gather: Main thread processes results in Newer->Older order using SeenTracker.
        
        size_t num_segments = snap->segments.size();
        if (num_segments > 0) {
            if (segment_pool_) {
                 futures.reserve(num_segments);
                 for(auto& seg : snap->segments) {
                      futures.push_back(segment_pool_->Enqueue([&, seg]() { 
                          return search_segment(seg); 
                      }));
                 }
            } else {
                 inline_results.reserve(num_segments);
                 for(auto& seg : snap->segments) {
                      inline_results.push_back(search_segment(seg));
                 }
            }
        }
        
        // Wait and Merge (Strict Order: Newest -> Oldest)
        // segments_ in snapshot are [0]=Newest ... [N]=Oldest.
        // So we just iterate 0..N and consume results.
        
        for (size_t i = 0; i < num_segments; ++i) {
            SegmentResult res;
            if (segment_pool_) {
                res = futures[i].get();
            } else {
                res = std::move(inline_results[i]);
            }
            
            auto& seg = snap->segments[i]; // Need to access segment for Rerank/Find
            
            // 1. Process IVF Candidates
            for (VectorId id : res.candidates) {
                 if (seen_tracker_.Contains(id)) continue;
                 seen_tracker_.MarkSeen(id); 
                 
                 std::span<const float> vec;
                 auto fr = seg->Find(id, &vec);
                 if (fr == table::SegmentReader::FindResult::kFound) {
                      float score = pomai::core::Dot(query, vec);
                      push_topk(id, score);
                 } else if (fr == table::SegmentReader::FindResult::kFoundTombstone) {
                      seen_tracker_.MarkTombstone(id);
                 }
            }
            
            // 2. Process Brute Force Hits (if any)
            // Warning: simple loop here assumes search_segment logic for brute force
            // matches this structure.
            // Currently search_segment for BF returns ALL hits in `hits` vector 
            // where score = -inf if tombstone?
            // This is inefficient for large segments (RAM).
            // But we mostly use IVF now.
            // Let's refine brute force later (Phase 4B).
            // For now, if no index, we used to do `seg->ForEach` with direct callback using seen_tracker.
            // If we parallelize, `seen_tracker` logic breaks.
            // Valid Approach: 
            //   If NO index, run serialized (old way) to allow streaming update of seen_tracker?
            //   OR returns top-K local + all tombstones.
            //   Given current scope, let's keep Brute Force SERIAL for simplicity in this PR
            //   and only parallelize IVF gather step.
            
            if (!seg->HasIndex()) {
                 // Serial Fallback for non-indexed segments
                 seg->ForEach([&](VectorId id, std::span<const float> vec, bool is_deleted) {
                    if (seen_tracker_.Contains(id)) return;
                    seen_tracker_.MarkSeen(id);
                    if (is_deleted) {
                        seen_tracker_.MarkTombstone(id);
                        return;
                    }
                    float score = pomai::core::Dot(query, vec);
                    push_topk(id, score);
                });
            }
        }
        
        // Final sort (descending by score)
        std::sort(out->begin(), out->end(),
                  [](const auto &a, const auto &b)
                  { return a.score > b.score; });

        return pomai::Status::Ok();
    }

} // namespace pomai::core


