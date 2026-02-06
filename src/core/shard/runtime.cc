#include "core/shard/runtime.h"
#include <filesystem>
#include <cassert>
#include <unordered_map>

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
#include "core/shard/layer_lookup.h"
#include <queue>
#include "core/shard/filter_evaluator.h" // Added
#include "pomai/metadata.h" // Added

namespace pomai::core
{

    namespace fs = std::filesystem; // Added

    namespace {
        struct VisibilityEntry {
            bool is_tombstone{false};
            const void* source{nullptr};
        };

        class SearchMergePolicy {
        public:
            void Reserve(std::size_t capacity) {
                visibility_.reserve(capacity);
            }

            void RecordIfUnresolved(VectorId id, bool is_deleted, const void* source) {
                if (visibility_.find(id) != visibility_.end()) {
                    return;
                }
                visibility_.emplace(id, VisibilityEntry{is_deleted, source});
            }

            const VisibilityEntry* Find(VectorId id) const {
                auto it = visibility_.find(id);
                if (it == visibility_.end()) {
                    return nullptr;
                }
                return &it->second;
            }

        private:
            std::unordered_map<VectorId, VisibilityEntry> visibility_;
        };

        struct WorseHit {
            bool operator()(const pomai::SearchHit& a, const pomai::SearchHit& b) const {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.id > b.id;
            }
        };

        bool IsBetterHit(const pomai::SearchHit& a, const pomai::SearchHit& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.id < b.id;
        }

        class LocalTopK {
        public:
            explicit LocalTopK(std::uint32_t k) : k_(k) {}

            void Push(pomai::VectorId id, float score) {
                if (k_ == 0) {
                    return;
                }
                pomai::SearchHit hit{id, score};
                if (heap_.size() < k_) {
                    heap_.push(hit);
                    return;
                }
                if (IsBetterHit(hit, heap_.top())) {
                    heap_.pop();
                    heap_.push(hit);
                }
            }

            std::vector<pomai::SearchHit> Drain() {
                std::vector<pomai::SearchHit> out;
                out.reserve(heap_.size());
                while (!heap_.empty()) {
                    out.push_back(heap_.top());
                    heap_.pop();
                }
                return out;
            }

        private:
            std::uint32_t k_;
            std::priority_queue<pomai::SearchHit, std::vector<pomai::SearchHit>, WorseHit> heap_;
        };
    } // namespace

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
        // Use atomic load to get shared_ptr
        auto m = mem_.load(std::memory_order_relaxed);
        if (m && m->GetCount() > 0) {
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
    void ShardRuntime::PublishSnapshot()
    {
        auto snap = std::make_shared<ShardSnapshot>();
        snap->version = next_snapshot_version_++;
        snap->created_at = std::chrono::steady_clock::now();
        
        // Copy atomic/shared state
        snap->segments = segments_; // Shared ownership of segments
        snap->frozen_memtables = frozen_mem_; // Shared ownership of frozen tables
        
        // INVARIANT: All frozen memtables are immutable (count fixed)
        for (const auto& fmem : snap->frozen_memtables) {
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
        // Since we are single writer, we can load relaxed.
        auto old_mem = mem_.load(std::memory_order_relaxed);
        if (old_mem->GetCount() == 0) return pomai::Status::Ok();
        
        // Push old shared_ptr to frozen
        frozen_mem_.push_back(old_mem);
        
        // Create new MemTable
        // engine.cc uses kArenaBlockBytes = 1MB. Assuming 1MB here too.
        auto new_mem = std::make_shared<table::MemTable>(dim_, 1u << 20);
        mem_.store(new_mem, std::memory_order_release);
        
        PublishSnapshot();
        return pomai::Status::Ok();
    }

    // -------------------------
    // Sync wrappers
    // -------------------------

    pomai::Status ShardRuntime::Put(pomai::VectorId id, std::span<const float> vec)
    {
        return Put(id, vec, pomai::Metadata());
    }

    pomai::Status ShardRuntime::Put(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata& meta)
    {
        if (vec.size() != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        PutCmd cmd;
        cmd.id = id;
        cmd.vec = vec.data(); 
        cmd.dim = static_cast<std::uint32_t>(vec.size());
        cmd.meta = meta; // Copy metadata
        
        auto f = cmd.done.get_future();
        auto st = Enqueue(Command{std::move(cmd)});
        if (!st.ok())
            return st;
        return f.get();
    }
// ... (BatchPut skipped) ...

    pomai::Status ShardRuntime::HandlePut(PutCmd &c)
    {
        if (c.dim != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        // 1. Write WAL
        auto st = wal_->AppendPut(c.id, {c.vec, c.dim}, c.meta);
        if (!st.ok())
            return st;

        // 2. Update MemTable
        auto m = mem_.load(std::memory_order_relaxed);
        st = m->Put(c.id, {c.vec, c.dim}, c.meta);
        if (!st.ok()) return st;

        // 3. Check Threshold for Soft Freeze (e.g. 5000 items)
        if (m->GetCount() >= 5000) {
            (void)RotateMemTable();
        }
        return pomai::Status::Ok();
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
        return Get(id, out, nullptr);
    }

    pomai::Status ShardRuntime::Get(pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta)
    {
        if (!out) return Status::InvalidArgument("out is null");

        auto active = mem_.load(std::memory_order_acquire);
        auto snap = GetSnapshot();
        if (!snap) return Status::Aborted("shard not ready");

        const auto lookup = LookupById(active, snap, id, dim_);
        if (lookup.state == LookupState::kTombstone) {
            return Status::NotFound("tombstone");
        }
        if (lookup.state == LookupState::kFound) {
            out->assign(lookup.vec.begin(), lookup.vec.end());
            if (out_meta) {
                *out_meta = lookup.meta;
            }
            return Status::Ok();
        }
        return Status::NotFound("vector not found");
    }

    // ... Exists ...

    pomai::Status ShardRuntime::GetFromSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta) {
        const auto lookup = LookupById(nullptr, snap, id, dim_);
        if (lookup.state == LookupState::kTombstone) {
            return Status::NotFound("tombstone");
        }
        if (lookup.state == LookupState::kFound) {
            out->assign(lookup.vec.begin(), lookup.vec.end());
            if (out_meta) {
                *out_meta = lookup.meta;
            }
            return Status::Ok();
        }
        return Status::NotFound("vector not found");
    }

    pomai::Status ShardRuntime::Exists(pomai::VectorId id, bool *exists)
    {
        if (!exists) return Status::InvalidArgument("exists is null");

        auto active = mem_.load(std::memory_order_acquire);
        auto snap = GetSnapshot();
        if (!snap) return Status::Aborted("shard not ready");

        const auto lookup = LookupById(active, snap, id, dim_);
        *exists = (lookup.state == LookupState::kFound);
        return Status::Ok();
    }



    std::pair<pomai::Status, bool> ShardRuntime::ExistsInSnapshot(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id) {
        const auto lookup = LookupById(nullptr, snap, id, dim_);
        return {Status::Ok(), lookup.state == LookupState::kFound};
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

    pomai::Status ShardRuntime::NewIterator(std::shared_ptr<ShardSnapshot> snap, std::unique_ptr<pomai::SnapshotIterator>* out)
    {
        *out = std::make_unique<ShardIterator>(std::move(snap));
        return pomai::Status::Ok();
    }

    // LOCK-FREE SEARCH
    pomai::Status ShardRuntime::Search(std::span<const float> query,
                                       std::uint32_t topk,
                                       std::vector<pomai::SearchHit> *out)
    {
        return Search(query, topk, SearchOptions{}, out);
    }

    pomai::Status ShardRuntime::Search(std::span<const float> query,
                                       std::uint32_t topk,
                                       const SearchOptions& opts,
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

        // P0.2: Capture active memtable for concurrency
        auto active = mem_.load(std::memory_order_acquire);

        // Local Search using Snapshot + Active (Bypass Mailbox)
        return SearchLocalInternal(active, snap, query, topk, opts, out);
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
        auto m = mem_.load(std::memory_order_relaxed);
        st = m->PutBatch(c.ids, spans);
        if (!st.ok())
            return st;
        
        // 3. Check threshold for soft freeze (same as single Put)
        if (m->GetCount() >= 5000) {
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

        st = mem_.load(std::memory_order_relaxed)->Delete(c.id);
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
        if (mem_.load(std::memory_order_relaxed)->GetCount() > 0) {
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
            fmem->IterateWithMetadata([&](VectorId id, std::span<const float> vec, bool is_deleted, const pomai::Metadata* meta) {
                pomai::Metadata meta_copy = meta ? *meta : pomai::Metadata();
                (void)builder.Add(id, vec, is_deleted, meta_copy);
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
                    pomai::Metadata meta; // Compact needs to preserve metadata!
                    if (segments_[top.seg_idx]->ReadAt(top.entry_idx, nullptr, &vec, nullptr, &meta).ok()) {
                        builder.Add(top.id, vec, false, meta);
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
        SearchReply r;
        auto snap = GetSnapshot();
        if(snap) {
             auto active = mem_.load(std::memory_order_acquire);
             r.st = SearchLocalInternal(active, snap, {c.query.data(), c.query.size()}, c.topk, SearchOptions{}, &r.hits);
        } else {
            r.st = Status::Aborted("no snapshot");
        }
        return r;
    }

// -------------------------
    // SearchLocalInternal: DB-grade 1-pass merge scan
    // -------------------------

    pomai::Status ShardRuntime::SearchLocalInternal(
            std::shared_ptr<table::MemTable> active,
            std::shared_ptr<ShardSnapshot> snap,
            std::span<const float> query,
            std::uint32_t topk,
            const SearchOptions& opts,
            std::vector<pomai::SearchHit> *out)
    {
        out->clear();
        out->reserve(topk);
        std::uint64_t visibility_scanned = 0;
        SearchMergePolicy merge_policy;

        std::size_t reserve_hint = 0;
        if (active) {
            reserve_hint += active->GetCount();
        }
        if (snap) {
            for (const auto& frozen : snap->frozen_memtables) {
                reserve_hint += frozen->GetCount();
            }
            for (const auto& seg : snap->segments) {
                reserve_hint += seg->Count();
            }
        }
        merge_policy.Reserve(reserve_hint);

        // -------------------------
        // Phase 1: Deterministic visibility map (newest -> oldest)
        // -------------------------
        if (active) {
            const void* source = active.get();
            active->IterateWithMetadata([&](VectorId id, std::span<const float>, bool is_deleted, const pomai::Metadata*) {
                ++visibility_scanned;
                merge_policy.RecordIfUnresolved(id, is_deleted, source);
            });
        }

        for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
            const void* source = it->get();
            (*it)->IterateWithMetadata([&](VectorId id, std::span<const float>, bool is_deleted, const pomai::Metadata*) {
                ++visibility_scanned;
                merge_policy.RecordIfUnresolved(id, is_deleted, source);
            });
        }

        for (const auto& seg : snap->segments) {
            const void* source = seg.get();
            seg->ForEach([&](VectorId id, std::span<const float>, bool is_deleted, const pomai::Metadata*) {
                ++visibility_scanned;
                merge_policy.RecordIfUnresolved(id, is_deleted, source);
            });
        }

        // -------------------------
        // Phase 2: Parallel scoring over authoritative sources
        // -------------------------
        std::atomic<std::uint64_t> scored_scanned{0};
        std::vector<pomai::SearchHit> candidates;

        auto score_memtable = [&](const std::shared_ptr<table::MemTable>& mem) {
            if (!mem) {
                return std::vector<pomai::SearchHit>{};
            }
            const void* source = mem.get();
            LocalTopK local(topk);
            std::uint64_t local_scanned = 0;
            mem->IterateWithMetadata([&](VectorId id, std::span<const float> vec, bool is_deleted, const pomai::Metadata* meta) {
                ++local_scanned;
                if (is_deleted) {
                    return;
                }
                const auto* entry = merge_policy.Find(id);
                if (!entry || entry->source != source || entry->is_tombstone) {
                    return;
                }
                const pomai::Metadata default_meta;
                const pomai::Metadata& m = meta ? *meta : default_meta;
                if (!core::FilterEvaluator::Matches(m, opts)) {
                    return;
                }
                float score = pomai::core::Dot(query, vec);
                local.Push(id, score);
            });
            scored_scanned.fetch_add(local_scanned, std::memory_order_relaxed);
            return local.Drain();
        };

        auto score_segment = [&](const std::shared_ptr<table::SegmentReader>& seg) {
            const void* source = seg.get();
            LocalTopK local(topk);
            std::uint64_t local_scanned = 0;
            seg->ForEach([&](VectorId id, std::span<const float> vec, bool is_deleted, const pomai::Metadata* meta) {
                ++local_scanned;
                if (is_deleted) {
                    return;
                }
                const auto* entry = merge_policy.Find(id);
                if (!entry || entry->source != source || entry->is_tombstone) {
                    return;
                }
                const pomai::Metadata default_meta;
                const pomai::Metadata& m = meta ? *meta : default_meta;
                if (!core::FilterEvaluator::Matches(m, opts)) {
                    return;
                }
                float score = pomai::core::Dot(query, vec);
                local.Push(id, score);
            });
            scored_scanned.fetch_add(local_scanned, std::memory_order_relaxed);
            return local.Drain();
        };

        {
            auto active_hits = score_memtable(active);
            candidates.insert(candidates.end(), active_hits.begin(), active_hits.end());
        }

        for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
            auto hits = score_memtable(*it);
            candidates.insert(candidates.end(), hits.begin(), hits.end());
        }

        std::vector<std::future<std::vector<pomai::SearchHit>>> futures;
        futures.reserve(snap->segments.size());
        std::vector<std::vector<pomai::SearchHit>> segment_hits(snap->segments.size());

        for (std::size_t i = 0; i < snap->segments.size(); ++i) {
            const auto& seg = snap->segments[i];
            if (segment_pool_) {
                futures.push_back(segment_pool_->Enqueue([&, seg]() { return score_segment(seg); }));
            } else {
                segment_hits[i] = score_segment(seg);
            }
        }

        for (std::size_t i = 0; i < futures.size(); ++i) {
            segment_hits[i] = futures[i].get();
        }

        for (const auto& hits : segment_hits) {
            candidates.insert(candidates.end(), hits.begin(), hits.end());
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.id < b.id;
        });

        if (candidates.size() > topk) {
            candidates.resize(topk);
        }

        out->assign(candidates.begin(), candidates.end());

        last_query_candidates_scanned_.store(
            visibility_scanned + scored_scanned.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        return pomai::Status::Ok();
    }
} // namespace pomai::core
