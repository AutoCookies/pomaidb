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
        constexpr std::chrono::milliseconds kBackgroundPoll{5};
        constexpr std::chrono::milliseconds kBackgroundBudget{2};
        constexpr std::size_t kBackgroundMaxEntriesPerTick = 2048;
        constexpr std::size_t kMaxSegmentEntries = 20000;
        constexpr std::size_t kMaxFrozenMemtables = 4;
        constexpr std::size_t kMemtableSoftLimit = 5000;

        struct BackgroundBudget {
            std::chrono::steady_clock::time_point deadline;
            std::size_t max_entries;
            std::size_t entries{0};

            bool HasBudget() const {
                return entries < max_entries && std::chrono::steady_clock::now() < deadline;
            }

            void Consume(std::size_t n = 1) {
                entries += n;
            }
        };

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

    struct ShardRuntime::BackgroundJob {
        enum class Type {
            kFreeze,
            kCompact
        };

        enum class Phase {
            kBuild,
            kFinalizeSegment,
            kCommitManifest,
            kInstall,
            kResetWal,
            kCleanup,
            kPublish,
            kDone
        };

        struct BuiltSegment {
            std::string filename;
            std::string filepath;
            std::shared_ptr<table::SegmentReader> reader;
        };

        struct FreezeState {
            Phase phase{Phase::kBuild};
            std::vector<std::shared_ptr<table::MemTable>> memtables;
            std::size_t target_frozen_count{0};
            std::size_t mem_index{0};
            std::size_t segment_part{0};
            std::optional<table::MemTable::Cursor> cursor;
            std::unique_ptr<table::SegmentBuilder> builder;
            std::string filename;
            std::string filepath;
            bool memtable_done_after_finalize{false};
            std::vector<BuiltSegment> built_segments;
            std::uint64_t wal_epoch_at_start{0};
        };

        struct CompactCursor {
            VectorId id;
            uint32_t seg_idx;
            uint32_t entry_idx;
            bool is_deleted;

            bool operator>(const CompactCursor& other) const {
                if (id != other.id) return id > other.id;
                return seg_idx > other.seg_idx;
            }
        };

        struct CompactState {
            Phase phase{Phase::kBuild};
            std::vector<std::shared_ptr<table::SegmentReader>> input_segments;
            std::priority_queue<CompactCursor, std::vector<CompactCursor>, std::greater<CompactCursor>> heap;
            VectorId last_id{std::numeric_limits<VectorId>::max()};
            bool is_first{true};
            std::unique_ptr<table::SegmentBuilder> builder;
            std::string filename;
            std::string filepath;
            std::size_t segment_part{0};
            std::vector<BuiltSegment> built_segments;
            std::vector<std::shared_ptr<table::SegmentReader>> old_segments;
            std::uint64_t total_entries_scanned{0};
            std::uint64_t tombstones_purged{0};
            std::uint64_t old_versions_dropped{0};
            std::uint64_t live_entries_kept{0};
        };

        BackgroundJob(Type t, FreezeState st) : type(t), state(std::move(st)) {}
        BackgroundJob(Type t, CompactState st) : type(t), state(std::move(st)) {}

        Type type;
        std::promise<pomai::Status> done;
        std::variant<FreezeState, CompactState> state;
    };

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
        cmd.vec = pomai::VectorView(vec);
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
        if (c.vec.dim != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        auto m = mem_.load(std::memory_order_relaxed);
        if (frozen_mem_.size() >= kMaxFrozenMemtables && m->GetCount() >= kMemtableSoftLimit) {
            return pomai::Status::ResourceExhausted("too many frozen memtables; backpressure");
        }

        // 1. Write WAL
        auto st = wal_->AppendPut(c.id, c.vec, c.meta);
        if (!st.ok())
            return st;
        ++wal_epoch_;

        // 2. Update MemTable
        st = m->Put(c.id, c.vec, c.meta);
        if (!st.ok()) return st;

        // 3. Check Threshold for Soft Freeze (e.g. 5000 items)
        if (m->GetCount() >= kMemtableSoftLimit) {
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
        
        BatchPutCmd cmd;
        cmd.ids = ids;
        cmd.vectors.reserve(vectors.size());
        for (const auto& vec : vectors) {
            cmd.vectors.emplace_back(vec);
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
            std::optional<Command> opt;
            if (background_job_) {
                opt = mailbox_.PopFor(kBackgroundPoll);
                if (!opt.has_value()) {
                    PumpBackgroundWork(kBackgroundBudget);
                    if (stop_now) {
                        break;
                    }
                    continue;
                }
            } else {
                opt = mailbox_.PopBlocking();
                if (!opt.has_value())
                    break;
            }

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
                        auto st = HandleFreeze(arg);
                        if (st.has_value()) {
                            arg.done.set_value(*st);
                        }
                    }
                    else if constexpr (std::is_same_v<T, CompactCmd>)
                    {
                        auto st = HandleCompact(arg);
                        if (st.has_value()) {
                            arg.done.set_value(*st);
                        }
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
                        CancelBackgroundJob("shard stopping");
                        arg.done.set_value();
                        stop_now = true;
                    }
                },
                cmd);

            if (background_job_) {
                PumpBackgroundWork(kBackgroundBudget);
            }

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

        auto m = mem_.load(std::memory_order_relaxed);
        if (frozen_mem_.size() >= kMaxFrozenMemtables && m->GetCount() >= kMemtableSoftLimit) {
            return pomai::Status::ResourceExhausted("too many frozen memtables; backpressure");
        }

        // 1. Batch write to WAL (KEY OPTIMIZATION: single fsync)
        auto st = wal_->AppendBatch(c.ids, c.vectors);
        if (!st.ok())
            return st;
        ++wal_epoch_;

        // 2. Batch update MemTable
        st = m->PutBatch(c.ids, c.vectors);
        if (!st.ok())
            return st;

        // 3. Check threshold for soft freeze (same as single Put)
        if (m->GetCount() >= kMemtableSoftLimit) {
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
        ++wal_epoch_;

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
    // HandleFreeze: Budgeted background freeze pipeline
    // -------------------------

    std::optional<pomai::Status> ShardRuntime::HandleFreeze(FreezeCmd &c)
    {
        if (background_job_) {
            return pomai::Status::Busy("background job already running");
        }

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

        BackgroundJob::FreezeState state;
        state.memtables = frozen_mem_;
        state.target_frozen_count = frozen_mem_.size();
        state.wal_epoch_at_start = wal_epoch_;
        auto job = std::make_unique<BackgroundJob>(BackgroundJob::Type::kFreeze, std::move(state));
        job->done = std::move(c.done);

        background_job_ = std::move(job);
        return std::nullopt;
    }

    // -------------------------
    // HandleCompact: Budgeted background compaction
    // -------------------------

    std::optional<pomai::Status> ShardRuntime::HandleCompact(CompactCmd &c)
    {
        (void)c;
        if (background_job_) {
            return pomai::Status::Busy("background job already running");
        }
        if (segments_.empty()) {
            return pomai::Status::Ok();
        }

        BackgroundJob::CompactState state;
        state.input_segments = segments_;
        state.old_segments = segments_;
        auto job = std::make_unique<BackgroundJob>(BackgroundJob::Type::kCompact, std::move(state));
        job->done = std::move(c.done);

        background_job_ = std::move(job);
        return std::nullopt;
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

    void ShardRuntime::CancelBackgroundJob(const std::string& reason)
    {
        if (!background_job_) {
            return;
        }
        background_job_->done.set_value(pomai::Status::Aborted(reason));
        background_job_.reset();
    }

    void ShardRuntime::PumpBackgroundWork(std::chrono::milliseconds budget)
    {
        if (!background_job_) {
            return;
        }

        BackgroundBudget bg_budget{
            std::chrono::steady_clock::now() + budget,
            kBackgroundMaxEntriesPerTick,
            0
        };

        auto complete_job = [&](const pomai::Status& st) {
            background_job_->done.set_value(st);
            background_job_.reset();
        };

        if (background_job_->type == BackgroundJob::Type::kFreeze) {
            auto& state = std::get<BackgroundJob::FreezeState>(background_job_->state);
            for (;;) {
                if (!bg_budget.HasBudget()) {
                    break;
                }
                if (state.phase == BackgroundJob::Phase::kBuild) {
                    if (state.mem_index >= state.memtables.size()) {
                        state.phase = BackgroundJob::Phase::kCommitManifest;
                        continue;
                    }

                    auto& mem = state.memtables[state.mem_index];
                    if (!state.cursor.has_value()) {
                        state.cursor = mem->CreateCursor();
                    }

                    table::MemTable::CursorEntry entry;
                    if (!state.cursor->Next(&entry)) {
                        state.cursor.reset();
                        if (state.builder && state.builder->Count() > 0) {
                            state.memtable_done_after_finalize = true;
                            state.phase = BackgroundJob::Phase::kFinalizeSegment;
                            continue;
                        }
                        state.mem_index++;
                        continue;
                    }

                    if (!state.builder) {
                        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
                        state.filename = "seg_" + std::to_string(now) + "_" +
                                         std::to_string(state.mem_index) + "_" +
                                         std::to_string(state.segment_part) + ".dat";
                        state.filepath = (fs::path(shard_dir_) / state.filename).string();
                        state.builder = std::make_unique<table::SegmentBuilder>(state.filepath, dim_);
                    }

                    pomai::Metadata meta_copy = entry.meta ? *entry.meta : pomai::Metadata();
                    auto st = state.builder->Add(entry.id, pomai::VectorView(entry.vec), entry.is_deleted, meta_copy);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal("Freeze: SegmentBuilder::Add failed: " + st.message()));
                        return;
                    }

                    bg_budget.Consume();

                    if (state.builder->Count() >= kMaxSegmentEntries) {
                        state.memtable_done_after_finalize = false;
                        state.phase = BackgroundJob::Phase::kFinalizeSegment;
                    }
                } else if (state.phase == BackgroundJob::Phase::kFinalizeSegment) {
                    auto st = state.builder->BuildIndex(index_params_.nlist);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal("Freeze: BuildIndex failed: " + st.message()));
                        return;
                    }
                    st = state.builder->Finish();
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal("Freeze: SegmentBuilder::Finish failed: " + st.message()));
                        return;
                    }
                    st = pomai::util::FsyncDir(shard_dir_);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal("Freeze: FsyncDir after segment failed: " + st.message()));
                        return;
                    }

                    std::unique_ptr<table::SegmentReader> reader;
                    st = table::SegmentReader::Open(state.filepath, &reader);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal("Freeze: SegmentReader::Open failed: " + st.message()));
                        return;
                    }

                    state.built_segments.push_back({state.filename, state.filepath, std::move(reader)});
                    state.builder.reset();
                    state.segment_part++;

                    if (state.memtable_done_after_finalize) {
                        state.mem_index++;
                        state.cursor.reset();
                        state.memtable_done_after_finalize = false;
                    }
                    state.phase = BackgroundJob::Phase::kBuild;
                } else if (state.phase == BackgroundJob::Phase::kCommitManifest) {
                    if (state.built_segments.empty()) {
                        state.phase = BackgroundJob::Phase::kResetWal;
                        continue;
                    }

                    std::vector<std::string> seg_names;
                    auto st = ShardManifest::Load(shard_dir_, &seg_names);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal("Freeze: ShardManifest::Load failed: " + st.message()));
                        return;
                    }

                    for (auto it = state.built_segments.rbegin(); it != state.built_segments.rend(); ++it) {
                        seg_names.insert(seg_names.begin(), it->filename);
                    }

                    st = ShardManifest::Commit(shard_dir_, seg_names);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal("Freeze: ShardManifest::Commit failed: " + st.message()));
                        return;
                    }
                    state.phase = BackgroundJob::Phase::kInstall;
                } else if (state.phase == BackgroundJob::Phase::kInstall) {
                    for (auto it = state.built_segments.rbegin(); it != state.built_segments.rend(); ++it) {
                        segments_.insert(segments_.begin(), std::move(it->reader));
                    }
                    state.phase = BackgroundJob::Phase::kResetWal;
                } else if (state.phase == BackgroundJob::Phase::kResetWal) {
                    if (state.target_frozen_count > 0 && state.target_frozen_count <= frozen_mem_.size()) {
                        frozen_mem_.erase(frozen_mem_.begin(),
                                          frozen_mem_.begin() + static_cast<std::ptrdiff_t>(state.target_frozen_count));
                    }
                    if (wal_epoch_ == state.wal_epoch_at_start) {
                        auto st = wal_->Reset();
                        if (!st.ok()) {
                            complete_job(pomai::Status::Internal("Freeze: WAL::Reset failed: " + st.message()));
                            return;
                        }
                    }
                    state.phase = BackgroundJob::Phase::kPublish;
                } else if (state.phase == BackgroundJob::Phase::kPublish) {
                    PublishSnapshot();
                    state.phase = BackgroundJob::Phase::kDone;
                } else if (state.phase == BackgroundJob::Phase::kDone) {
                    complete_job(pomai::Status::Ok());
                    return;
                } else {
                    break;
                }
            }
            return;
        }

        auto& state = std::get<BackgroundJob::CompactState>(background_job_->state);
        for (;;) {
            if (!bg_budget.HasBudget()) {
                break;
            }

            if (state.phase == BackgroundJob::Phase::kBuild) {
                if (state.heap.empty() && !state.builder) {
                    for (uint32_t i = 0; i < state.input_segments.size(); ++i) {
                        VectorId id;
                        bool del;
                        if (state.input_segments[i]->ReadAt(0, &id, nullptr, &del).ok()) {
                            state.heap.push({id, i, 0, del});
                        }
                    }
                    if (state.heap.empty()) {
                        state.phase = BackgroundJob::Phase::kCommitManifest;
                        continue;
                    }
                }

                while (bg_budget.HasBudget() && !state.heap.empty()) {
                    auto top = state.heap.top();
                    state.heap.pop();
                    state.total_entries_scanned++;

                    if (state.is_first || top.id != state.last_id) {
                        if (top.is_deleted) {
                            state.tombstones_purged++;
                        } else {
                            std::span<const float> vec;
                            pomai::Metadata meta;
                            if (state.input_segments[top.seg_idx]->ReadAt(top.entry_idx, nullptr, &vec, nullptr, &meta).ok()) {
                                if (!state.builder) {
                                    auto sys_now = std::chrono::system_clock::now().time_since_epoch().count();
                                    state.filename = "seg_" + std::to_string(sys_now) + "_compacted_" +
                                                     std::to_string(state.segment_part) + ".dat";
                                    state.filepath = (fs::path(shard_dir_) / state.filename).string();
                                    state.builder = std::make_unique<table::SegmentBuilder>(state.filepath, dim_);
                                }
                                auto st = state.builder->Add(top.id, pomai::VectorView(vec), false, meta);
                                if (!st.ok()) {
                                    complete_job(pomai::Status::Internal("Compact: SegmentBuilder::Add failed: " + st.message()));
                                    return;
                                }
                                state.live_entries_kept++;
                                if (state.builder->Count() >= kMaxSegmentEntries) {
                                    state.phase = BackgroundJob::Phase::kFinalizeSegment;
                                    break;
                                }
                            }
                        }
                        state.last_id = top.id;
                        state.is_first = false;
                    } else {
                        state.old_versions_dropped++;
                    }

                    uint32_t next_idx = top.entry_idx + 1;
                    VectorId next_id;
                    bool next_del;
                    if (state.input_segments[top.seg_idx]->ReadAt(next_idx, &next_id, nullptr, &next_del).ok()) {
                        state.heap.push({next_id, top.seg_idx, next_idx, next_del});
                    }
                    bg_budget.Consume();
                }

                if (state.heap.empty() && state.builder) {
                    state.phase = BackgroundJob::Phase::kFinalizeSegment;
                }
            } else if (state.phase == BackgroundJob::Phase::kFinalizeSegment) {
                if (!state.builder) {
                    state.phase = BackgroundJob::Phase::kCommitManifest;
                    continue;
                }
                auto st = state.builder->BuildIndex(index_params_.nlist);
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal("Compact: BuildIndex failed: " + st.message()));
                    return;
                }
                st = state.builder->Finish();
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal("Compact: SegmentBuilder::Finish failed: " + st.message()));
                    return;
                }
                st = pomai::util::FsyncDir(shard_dir_);
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal("Compact: FsyncDir after segment failed: " + st.message()));
                    return;
                }

                std::unique_ptr<table::SegmentReader> reader;
                st = table::SegmentReader::Open(state.filepath, &reader);
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal("Compact: SegmentReader::Open failed: " + st.message()));
                    return;
                }

                state.built_segments.push_back({state.filename, state.filepath, std::move(reader)});
                state.builder.reset();
                state.segment_part++;
                state.phase = state.heap.empty() ? BackgroundJob::Phase::kCommitManifest : BackgroundJob::Phase::kBuild;
            } else if (state.phase == BackgroundJob::Phase::kCommitManifest) {
                std::vector<std::string> seg_names;
                seg_names.reserve(state.built_segments.size());
                for (auto it = state.built_segments.rbegin(); it != state.built_segments.rend(); ++it) {
                    seg_names.push_back(it->filename);
                }
                auto st = ShardManifest::Commit(shard_dir_, seg_names);
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal("Compact: ShardManifest::Commit failed: " + st.message()));
                    return;
                }
                state.phase = BackgroundJob::Phase::kInstall;
            } else if (state.phase == BackgroundJob::Phase::kInstall) {
                segments_.clear();
                for (auto it = state.built_segments.rbegin(); it != state.built_segments.rend(); ++it) {
                    segments_.push_back(std::move(it->reader));
                }
                state.phase = BackgroundJob::Phase::kCleanup;
            } else if (state.phase == BackgroundJob::Phase::kCleanup) {
                for (const auto& old : state.old_segments) {
                    std::error_code ec;
                    fs::remove(old->Path(), ec);
                }
                state.phase = BackgroundJob::Phase::kPublish;
            } else if (state.phase == BackgroundJob::Phase::kPublish) {
                PublishSnapshot();
                state.phase = BackgroundJob::Phase::kDone;
            } else if (state.phase == BackgroundJob::Phase::kDone) {
                complete_job(pomai::Status::Ok());
                return;
            } else {
                break;
            }
        }
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
