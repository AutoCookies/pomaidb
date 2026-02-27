#include "core/shard/runtime.h"
#include <filesystem>
#include <cassert>
#include <unordered_map>
#ifdef __linux__
#include <sched.h>   // sched_setaffinity, cpu_set_t — CPU affinity pinning
#endif

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
#include "core/bitset_mask.h"             // Phase 3: pre-computed per-segment bitset
#include <iostream>
#include <list>
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
    } // anonymous namespace

    struct VisibilityEntry {
        bool is_tombstone{false};
        const void* source{nullptr};
    };

    class SearchMergePolicy {
    public:
        void Reserve(std::size_t capacity) {
                visibility_.reserve(capacity);
            }
            
            void Clear() {
                visibility_.clear();
            }

            bool Empty() const {
                return visibility_.empty();
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
            std::list<std::vector<float>> compact_buffers;
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
                               pomai::MembraneKind kind,
                               pomai::MetricType metric,
                               std::unique_ptr<storage::Wal> wal,
                               std::unique_ptr<table::MemTable> mem,
                               std::size_t mailbox_cap,
                               const pomai::IndexParams& index_params,
                               pomai::util::ThreadPool* thread_pool,
                               pomai::util::ThreadPool* segment_pool)
        : shard_id_(shard_id),
          shard_dir_(std::move(shard_dir)), // Added
          dim_(dim),
          kind_(kind),
          metric_(metric),
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

    // Phase 4: lock-free stats snapshot
    ShardStats ShardRuntime::GetStats() const noexcept
    {
        ShardStats s;
        s.shard_id          = shard_id_;
        s.ops_processed     = ops_processed_.load(std::memory_order_relaxed);
        s.queue_depth       = static_cast<std::uint64_t>(mailbox_.Size());
        s.candidates_scanned = last_query_candidates_scanned_.load(std::memory_order_relaxed);
        auto mem = mem_.load(std::memory_order_acquire);
        s.memtable_entries  = mem ? static_cast<std::uint64_t>(mem->GetCount()) : 0u;
        return s;
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
                               {
                                // Phase 1 (Helio shared-nothing): pin this shard's thread
                                // to a specific CPU core so its L1/L2 cache is dedicated.
                                // Guarded for Linux/Android only; silently skipped elsewhere.
#if defined(__linux__)
                                {
                                    const int nproc = static_cast<int>(
                                        std::thread::hardware_concurrency());
                                    if (nproc > 0) {
                                        cpu_set_t cs;
                                        CPU_ZERO(&cs);
                                        CPU_SET(static_cast<int>(shard_id_) % nproc, &cs);
                                        // Best-effort: ignore errors (e.g. Docker CPU restrictions)
                                        (void)sched_setaffinity(0, sizeof(cs), &cs);
                                    }
                                }
#endif
                                RunLoop();
                               });
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
        if (kind_ != pomai::MembraneKind::kVector) {
            return pomai::Status::InvalidArgument("VECTOR membrane required for Put");
        }
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
        if (kind_ != pomai::MembraneKind::kVector) {
            return pomai::Status::InvalidArgument("VECTOR membrane required for Put");
        }
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
        if (kind_ != pomai::MembraneKind::kVector) {
            return pomai::Status::InvalidArgument("VECTOR membrane required for PutBatch");
        }
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

    pomai::Status ShardRuntime::GetSemanticPointer(std::shared_ptr<ShardSnapshot> snap, pomai::VectorId id, pomai::SemanticPointer* out) {
        if (!snap) return pomai::Status::InvalidArgument("snapshot null");
        // Look in segments only since memtables are not zero-copy aligned
        for (const auto& seg : snap->segments) {
             const uint8_t* raw_payload = nullptr;
             if (seg->FindRaw(id, &raw_payload, nullptr) == table::SegmentReader::FindResult::kFound) {
                 out->raw_data_ptr = raw_payload;
                 out->dim = seg->Dim();
                 if (seg->IsQuantized()) {
                     out->quant_min = seg->GetQuantizer()->GetGlobalMin();
                     out->quant_inv_scale = seg->GetQuantizer()->GetGlobalInvScale();
                 } else {
                     out->quant_min = 0;
                     out->quant_inv_scale = 1.0f;
                 }
                 out->session_id = 0; // Filled later
                 return pomai::Status::Ok();
             }
        }
        return pomai::Status::NotFound("vector not in segments (might be in memtable or deleted)");
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

        std::vector<std::vector<pomai::SearchHit>> batch_out(1);
        auto st = SearchBatchLocal(query, {0}, topk, opts, &batch_out);
        if (st.ok()) {
            *out = std::move(batch_out[0]);
        } else {
            out->clear();
        }
        return st;
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
        if (kind_ != pomai::MembraneKind::kVector) {
            return pomai::Status::InvalidArgument("VECTOR membrane required for PutBatch");
        }
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
             std::vector<std::vector<pomai::SearchHit>> batch_out(1);
             std::vector<float> query_vec(c.query.begin(), c.query.end());
             auto st = SearchBatchLocal(query_vec, {0}, c.topk, SearchOptions{}, &batch_out);
             r.st = st;
             if (st.ok()) {
                 r.hits = std::move(batch_out[0]);
             }
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
                        state.builder = std::make_unique<table::SegmentBuilder>(state.filepath, dim_, index_params_, metric_);
                    }

                    pomai::Metadata meta_copy = entry.meta ? *entry.meta : pomai::Metadata();
                    auto st = state.builder->Add(entry.id, pomai::VectorView(entry.vec), entry.is_deleted, meta_copy);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal("Freeze: SegmentBuilder::Add failed: " + st.message()));
                        return;
                    }

                    // Feed Streaming IVF for continuous SOM updates
                    if (!entry.is_deleted) {
                         st = ivf_->Put(entry.id, std::span<const float>(entry.vec));
                         if (!st.ok()) {
                             complete_job(pomai::Status::Internal("Freeze: IVF::Put failed: " + st.message()));
                             return;
                         }
                    }

                    bg_budget.Consume();

                    if (state.builder->Count() >= kMaxSegmentEntries) {
                        state.memtable_done_after_finalize = false;
                        state.phase = BackgroundJob::Phase::kFinalizeSegment;
                    }
                } else if (state.phase == BackgroundJob::Phase::kFinalizeSegment) {
                    auto st = state.builder->BuildIndex();
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
                            std::span<const float> vec_mapped;
                            std::vector<float> vec_decoded;
                            pomai::Metadata meta;
                            auto res = state.input_segments[top.seg_idx]->FindAndDecode(top.id, &vec_mapped, &vec_decoded, &meta);
                            if (res == table::SegmentReader::FindResult::kFound) {
                                if (state.input_segments[top.seg_idx]->IsQuantized()) {
                                    state.compact_buffers.push_back(std::move(vec_decoded));
                                    vec_mapped = std::span<const float>(state.compact_buffers.back());
                                }
                                if (!state.builder) {
                                    auto sys_now = std::chrono::system_clock::now().time_since_epoch().count();
                                    state.filename = "seg_" + std::to_string(sys_now) + "_compacted_" +
                                                     std::to_string(state.segment_part) + ".dat";
                                    state.filepath = (fs::path(shard_dir_) / state.filename).string();
                                    state.builder = std::make_unique<table::SegmentBuilder>(state.filepath, dim_, index_params_, metric_);
                                }
                                std::cout << "TEST_DEBUG COMPACT PRE-ADD id: " << top.id << " vec_mapped[0]: " << vec_mapped[0] << std::endl;
                                auto st = state.builder->Add(top.id, pomai::VectorView(vec_mapped), false, meta);
                                if (!st.ok()) {
                                    complete_job(pomai::Status::Internal("Compact: SegmentBuilder::Add failed: " + st.message()));
                                    return;
                                }

                                // Feed downstream Streaming IVF
                                st = ivf_->Put(top.id, vec_mapped);
                                if (!st.ok()) {
                                     complete_job(pomai::Status::Internal("Compact: IVF::Put failed: " + st.message()));
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
                auto st = state.builder->BuildIndex();
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
                state.compact_buffers.clear();
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

    pomai::Status ShardRuntime::SearchBatchLocal(std::span<const float> queries,
                                                 const std::vector<uint32_t>& query_indices,
                                                 std::uint32_t topk,
                                                 const SearchOptions& opts,
                                                 std::vector<std::vector<pomai::SearchHit>>* out_results)
    {
        if (query_indices.empty()) return pomai::Status::Ok();
        if (queries.size() % dim_ != 0) return pomai::Status::InvalidArgument("dim mismatch");
        if (!out_results) return pomai::Status::InvalidArgument("out_results null");
        
        auto snap = GetSnapshot();
        if (!snap) return pomai::Status::Aborted("shard not ready");
        auto active = mem_.load(std::memory_order_acquire);

        // Visibility is needed if we have updates across layers or multiple segments
        bool use_visibility = (active != nullptr && active->GetCount() > 0) || 
                              (!snap->frozen_memtables.empty()) || 
                              (snap->segments.size() > 1);

        SearchMergePolicy shared_policy;
        if (use_visibility) {
            std::size_t reserve_hint = 0;
            if (active) reserve_hint += active->GetCount();
            for (const auto& frozen : snap->frozen_memtables) reserve_hint += frozen->GetCount();
            for (const auto& seg : snap->segments) reserve_hint += seg->Count();
            
            shared_policy.Reserve(reserve_hint);
            
            // Build the "Newest Wins" map ONCE for the whole batch
            if (active) {
                const void* src = active.get();
                active->IterateWithMetadata([&](VectorId id, std::span<const float>, bool is_deleted, const pomai::Metadata*) {
                    shared_policy.RecordIfUnresolved(id, is_deleted, src);
                });
            }
            for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
                const void* src = it->get();
                (*it)->IterateWithMetadata([&](VectorId id, std::span<const float>, bool is_deleted, const pomai::Metadata*) {
                    shared_policy.RecordIfUnresolved(id, is_deleted, src);
                });
            }
            for (const auto& seg : snap->segments) {
                const void* src = seg.get();
                seg->ForEach([&](VectorId id, std::span<const float>, bool is_deleted, const pomai::Metadata*) {
                    shared_policy.RecordIfUnresolved(id, is_deleted, src);
                });
            }
        }

        std::vector<float> query_sums(queries.size() / dim_, 0.0f);
        for (uint32_t q_idx : query_indices) {
            std::span<const float> q(queries.data() + q_idx * dim_, dim_);
            float s = 0.0f;
            for (float f : q) s += f;
            query_sums[q_idx] = s;
        }

        // Process queries sequentially within this shard to avoid cache thrashing and oversubscription
        // Parallelism comes from VectorEngine processing multiple shards at once.
        for (uint32_t q_idx : query_indices) {
            std::span<const float> single_query(queries.data() + (q_idx * dim_), dim_);
            float q_sum = query_sums[q_idx];
            auto st = SearchLocalInternal(active, snap, single_query, q_sum, topk, opts, shared_policy, use_visibility, &(*out_results)[q_idx], false);
            if (!st.ok()) return st;
        }

        return pomai::Status::Ok();
    }

// -------------------------
    // SearchLocalInternal: DB-grade 1-pass merge scan
    // -------------------------

    pomai::Status ShardRuntime::SearchLocalInternal(
            std::shared_ptr<table::MemTable> active,
            std::shared_ptr<ShardSnapshot> snap,
            std::span<const float> query,
            float query_sum,
            std::uint32_t topk,
            const SearchOptions& opts,
            SearchMergePolicy& merge_policy,
            bool use_visibility,
            std::vector<pomai::SearchHit> *out,
            bool use_pool)
    {
        out->clear();
        out->reserve(topk);

        if (use_visibility && merge_policy.Empty()) {
            // For single-query search, we only build the map for memtables.
            // Segment-level visibility is handled on-the-fly or via pre-built batch policy.
            merge_policy.Reserve(64); 
            if (active) {
                const void* src = active.get();
                active->IterateWithMetadata([&](VectorId id, std::span<const float>, bool is_deleted, const pomai::Metadata*) {
                    merge_policy.RecordIfUnresolved(id, is_deleted, src);
                });
            }
            for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
                const void* src = it->get();
                (*it)->IterateWithMetadata([&](VectorId id, std::span<const float>, bool is_deleted, const pomai::Metadata*) {
                    merge_policy.RecordIfUnresolved(id, is_deleted, src);
                });
            }
        }

        // -------------------------
        // Phase 2: Parallel scoring over authoritative sources
        // -------------------------
        std::atomic<std::uint64_t> scored_scanned{0};
        std::vector<pomai::SearchHit> candidates;

        bool has_filters = !opts.filters.empty();
        const std::size_t min_candidates = has_filters ? std::max<std::size_t>(static_cast<std::size_t>(topk) * 50u, 2000u) : (static_cast<std::size_t>(topk) * 10u);
        uint32_t effective_nprobe = index_params_.nprobe == 0 ? 1 : index_params_.nprobe;
        
        // If we expect to hit many candidates but nprobe is small, try to increase nprobe instead of full scan
        if (has_filters && effective_nprobe < 8) {
            effective_nprobe = std::min<uint32_t>(32u, effective_nprobe * 8); // Heuristic to avoid brute force
        }
        bool allow_fallback = true;
        if (thread_pool_) {
            const std::size_t threads = thread_pool_->Size();
            const std::size_t pending = thread_pool_->Pending();
            const bool low_end = threads <= 2;
            const bool overloaded = pending > threads;
            if (low_end || overloaded) {
                effective_nprobe = std::max(1u, effective_nprobe / 2);
                allow_fallback = false;
            }
        }

        auto score_memtable = [&](const std::shared_ptr<table::MemTable>& mem) {
            if (!mem) {
                return std::make_pair(std::vector<pomai::SearchHit>{}, static_cast<std::uint64_t>(0));
            }
            const void* source = mem.get();
            LocalTopK local(topk);
            std::uint64_t local_scanned = 0;
            mem->IterateWithMetadata([&](VectorId id, std::span<const float> vec, bool is_deleted, const pomai::Metadata* meta) {
                ++local_scanned;
                if (is_deleted) {
                    return;
                }
                if (use_visibility) {
                    const auto* entry = merge_policy.Find(id);
                    if (!entry || entry->source != source || entry->is_tombstone) {
                        return;
                    }
                }
                const pomai::Metadata default_meta;
                const pomai::Metadata& m = meta ? *meta : default_meta;
                if (!core::FilterEvaluator::Matches(m, opts)) {
                    return;
                }
                float score = 0.0f;
                if (this->metric_ == pomai::MetricType::kInnerProduct || this->metric_ == pomai::MetricType::kCosine) {
                    score = pomai::core::Dot(query, vec);
                } else {
                    score = -pomai::core::L2Sq(query, vec);
                }
                local.Push(id, score);
            });
            return std::make_pair(local.Drain(), local_scanned);
        };

        std::atomic<uint64_t> total_scanned{0}; // Declared here as per instruction
        auto score_segment = [&](const std::shared_ptr<table::SegmentReader>& seg) {
            const void* source = seg.get();
            LocalTopK local(topk);
            std::uint64_t local_scanned = 0;
            bool used_candidates = false;

            // Phase 3: Pre-compute bitset for this segment when filters active.
            // One sequential forward pass (cache-friendly mmap reads) replaces
            // per-candidate FilterEvaluator::Matches() calls in the hot loops below.
            core::BitsetMask seg_mask(seg->Count());
            if (has_filters) {
                seg_mask.BuildFromSegment(*seg, opts);
            }

            if (!use_visibility && !has_filters) { // FAST PATH
                // === ADAPTIVE DISPATCHER ===
                // Small segments: brute-force SIMD for 100% recall.
                // Large segments (>= threshold): HNSW graph traversal.
                const bool use_graph = (seg->Count() >= index_params_.adaptive_threshold) &&
                                       (seg->GetHnswIndex() != nullptr);
                if (use_graph) {
                    auto* hnsw = seg->GetHnswIndex();
                    std::vector<pomai::VectorId> out_ids;
                    std::vector<float> out_dists;
                    // Pass ef_search from index params for tuning
                    const int ef = static_cast<int>(
                        std::max(index_params_.hnsw_ef_search,
                                 static_cast<uint32_t>(topk) * 2));
                    if (hnsw->Search(query, topk, ef, &out_ids, &out_dists).ok() &&
                        !out_ids.empty()) {
                        used_candidates = true;
                        for (size_t i = 0; i < out_ids.size(); ++i) {
                            local_scanned++;
                            // id_map now stores real user VectorIds directly.
                            if (this->metric_ == pomai::MetricType::kInnerProduct ||
                                this->metric_ == pomai::MetricType::kCosine) {
                                local.Push(out_ids[i], out_dists[i]);
                            } else {
                                local.Push(out_ids[i], -out_dists[i]);
                            }
                        }
                        total_scanned.fetch_add(local_scanned, std::memory_order_relaxed);
                        return local.Drain();
                    }
                }
                if (seg->IsQuantized()) {
                    float q_min = seg->GetQuantizer()->GetGlobalMin();
                    float q_inv_scale = seg->GetQuantizer()->GetGlobalInvScale();
                    thread_local std::vector<uint32_t> cand_reuse;
                    cand_reuse.clear();
                    if (seg->Search(query, effective_nprobe, &cand_reuse).ok()) {
                        used_candidates = true; // Mark as used candidates for fast path
                        for (uint32_t idx : cand_reuse) {
                            local_scanned++;
                            const uint8_t* p = seg->GetBaseAddr() + seg->GetEntriesStartOffset() + idx * seg->GetEntrySize();
                            const uint8_t* codes_ptr = p + 12; // Assuming ID (8 bytes) + is_deleted (1 byte) + metadata_len (3 bytes) = 12 bytes offset
                            if (!(*(p+8) & 0x01)) { // not tombstone, assuming is_deleted is at offset 8
                                local.Push(*(uint64_t*)p, pomai::core::DotSq8(query, std::span<const uint8_t>(codes_ptr, dim_), q_min, q_inv_scale, query_sum));
                            }
                        }
                        total_scanned.fetch_add(local_scanned, std::memory_order_relaxed);
                        return local.Drain();
                    }
                }
            }
            
            thread_local std::vector<uint32_t> cand_idxs_reuse;
            cand_idxs_reuse.clear();
            auto cand_status = seg->Search(query, effective_nprobe, &cand_idxs_reuse);
            if (cand_status.ok() && !cand_idxs_reuse.empty()) {
                std::sort(cand_idxs_reuse.begin(), cand_idxs_reuse.end());
                cand_idxs_reuse.erase(std::unique(cand_idxs_reuse.begin(), cand_idxs_reuse.end()), cand_idxs_reuse.end());

                if (cand_idxs_reuse.size() >= min_candidates || !allow_fallback) {
                    used_candidates = true;
                    pomai::Metadata local_meta;
                    pomai::Metadata* meta_ptr = has_filters ? &local_meta : nullptr;
                    
                    if (seg->IsQuantized()) {
                        float q_min = seg->GetQuantizer()->GetGlobalMin();
                        float q_inv_scale = seg->GetQuantizer()->GetGlobalInvScale();

                        for (const uint32_t entry_idx : cand_idxs_reuse) {
                            ++local_scanned;
                            pomai::VectorId id;
                            std::span<const uint8_t> codes;
                            bool deleted = false;
                            auto st = seg->ReadAtCodes(entry_idx, &id, &codes, &deleted, meta_ptr);
                            if (!st.ok() || deleted) continue;

                            if (use_visibility) {
                                const auto* entry = merge_policy.Find(id);
                                if (!entry || entry->source != source || entry->is_tombstone) continue;
                            }
                            if (has_filters && !seg_mask.Test(entry_idx)) continue;

                            float score = pomai::core::DotSq8(query, codes, q_min, q_inv_scale, query_sum);
                            local.Push(id, score);
                        }
                    } else {
                        for (const uint32_t entry_idx : cand_idxs_reuse) {
                            ++local_scanned;
                            pomai::VectorId id;
                            std::span<const float> vec;
                            bool deleted = false;
                            auto st = seg->ReadAt(entry_idx, &id, &vec, &deleted, meta_ptr);
                            if (!st.ok() || deleted) continue;

                            if (use_visibility) {
                                const auto* entry = merge_policy.Find(id);
                                if (!entry || entry->source != source || entry->is_tombstone) continue;
                            }
                            if (has_filters && !seg_mask.Test(entry_idx)) continue;

                            float score = (this->metric_ == pomai::MetricType::kInnerProduct || this->metric_ == pomai::MetricType::kCosine)
                                              ? pomai::core::Dot(query, vec)
                                              : -pomai::core::L2Sq(query, vec);
                            local.Push(id, score);
                        }
                    }
                }
            }

            if (!used_candidates) {
                if (has_filters) {
                    // ForEach fallback with BitsetMask: use a counter to get entry_idx.
                    // ForEach doesn't expose entry_idx directly, so we use a local counter.
                    uint32_t fe_idx = 0;
                    seg->ForEach([&](VectorId id, std::span<const float> vec, bool is_deleted, const pomai::Metadata* meta) {
                        (void)meta; // suppress unused warning
                        const uint32_t my_idx = fe_idx++;
                        ++local_scanned;
                        if (is_deleted) return;
                        if (use_visibility) {
                            const auto* entry = merge_policy.Find(id);
                            if (!entry || entry->source != source || entry->is_tombstone) return;
                        }
                        // Phase 3: bit test replaces string-compare FilterEvaluator::Matches()
                        if (!seg_mask.Test(my_idx)) return;

                        float score = 0.0f;
                        if (this->metric_ == pomai::MetricType::kInnerProduct || this->metric_ == pomai::MetricType::kCosine) {
                            score = pomai::core::Dot(query, vec);
                        } else {
                            score = -pomai::core::L2Sq(query, vec);
                        }
                        local.Push(id, score);
                    });
                } else {
                    seg->ForEach([&](VectorId id, std::span<const float> vec, bool is_deleted, const pomai::Metadata*) {
                        ++local_scanned;
                        if (is_deleted) return;
                        if (use_visibility) {
                            const auto* entry = merge_policy.Find(id);
                            if (!entry || entry->source != source || entry->is_tombstone) return;
                        }
                        
                        float score = 0.0f;
                        if (this->metric_ == pomai::MetricType::kInnerProduct || this->metric_ == pomai::MetricType::kCosine) {
                            score = pomai::core::Dot(query, vec);
                        } else {
                            score = -pomai::core::L2Sq(query, vec);
                        }
                        local.Push(id, score);
                    });
                }
            }
            total_scanned.fetch_add(local_scanned, std::memory_order_relaxed);
            return local.Drain();
        };

        {
            auto [hits, scanned] = score_memtable(active);
            total_scanned.fetch_add(scanned, std::memory_order_relaxed);
            candidates.insert(candidates.end(), hits.begin(), hits.end());
        }

        for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
            auto [hits, scanned] = score_memtable(*it);
            total_scanned.fetch_add(scanned, std::memory_order_relaxed);
            candidates.insert(candidates.end(), hits.begin(), hits.end());
        }

        std::vector<std::future<std::vector<pomai::SearchHit>>> futures;
        futures.reserve(snap->segments.size());
        std::vector<std::vector<pomai::SearchHit>> segment_hits(snap->segments.size());

        for (std::size_t i = 0; i < snap->segments.size(); ++i) {
            const auto& seg = snap->segments[i];
            if (segment_pool_ && use_pool) {
                futures.push_back(segment_pool_->Enqueue([&, seg]() { return score_segment(seg); }));
            } else {
                segment_hits[i] = score_segment(seg);
            }
        }

        for (std::size_t i = 0; i < futures.size(); ++i) {
            segment_hits[i] = futures[i].get();
        }

        last_query_candidates_scanned_.fetch_add(total_scanned.load(std::memory_order_relaxed), std::memory_order_relaxed);

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
        return pomai::Status::Ok();
    }
} // namespace pomai::core
