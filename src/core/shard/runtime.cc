#include "core/shard/runtime.h"
#include <filesystem>
#include <cassert>
#include <unordered_map>

#include <algorithm>
#include <chrono>
#include <deque>

#include "core/distance.h"
#include "core/index/ivf_coarse.h"
#include "storage/wal/wal.h"
#include "table/memtable.h"
#include "table/segment.h" // Added
#include "core/shard/iterator.h"
#include "core/shard/manifest.h"
#include "core/shard/layer_lookup.h"
#include <queue>
#include "core/storage/compaction_manager.h"
#include "core/shard/filter_evaluator.h"
#include "core/bitset_mask.h"             // Phase 3: pre-computed per-segment bitset
#include <iostream>
#include <list>
#include "pomai/metadata.h" // Added
#include "util/posix_file.h" // Added for FsyncDir
#include "util/logging.h"

namespace pomai::core
{

    namespace fs = std::filesystem; // Added

    namespace {
        constexpr std::chrono::milliseconds kBackgroundPoll{5};
        // Per-tick work budget for background freeze/compact (kept small so callers stay responsive).
        constexpr std::chrono::milliseconds kBackgroundBudget{2};
        // Upper bound on how long a single synchronous Freeze()/Compact() call will spend in background work
        // before timing out and aborting the job. Prevents unbounded stalls on edge devices while leaving
        // enough room for index builds on slower hardware.
        constexpr std::chrono::seconds kBackgroundMaxSyncDuration{120};
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

    struct VectorRuntime::BackgroundJob {
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
            std::deque<std::vector<float>> compact_buffers; // Stable pointers for builder views
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
        std::optional<pomai::Status> result;  // Set when phase == kDone (single-threaded)
        std::variant<FreezeState, CompactState> state;
    };

    VectorRuntime::VectorRuntime(std::uint32_t runtime_id,
                               std::string data_dir,
                               std::uint32_t dim,
                               pomai::MembraneKind kind,
                               pomai::MetricType metric,
                               std::unique_ptr<storage::Wal> wal,
                               std::unique_ptr<table::MemTable> mem,
                               const pomai::IndexParams& index_params)
        : runtime_id_(runtime_id),
          data_dir_(std::move(data_dir)),
          dim_(dim),
          kind_(kind),
          metric_(metric),
          wal_(std::move(wal)),
          mem_(std::move(mem)),
          index_params_(index_params)
    {
        pomai::index::IvfCoarse::Options opt;
        opt.nlist = index_params_.nlist;
        opt.nprobe = index_params_.nprobe;
        opt.warmup = 256;
        ivf_ = std::make_unique<pomai::index::IvfCoarse>(dim_, opt);
        compaction_manager_ = std::make_unique<storage::CompactionManager>();
    }

    VectorRuntime::~VectorRuntime()
    {
        started_ = false;
    }

    RuntimeStats VectorRuntime::GetStats() const noexcept
    {
        RuntimeStats s;
        s.runtime_id        = runtime_id_;
        s.ops_processed     = ops_processed_;
        s.queue_depth       = 0u;
        s.candidates_scanned = last_query_candidates_scanned_;
        s.memtable_entries  = mem_ ? static_cast<std::uint64_t>(mem_->GetCount()) : 0u;

        s.mem_committed = 0;
        s.mem_used = 0;
        return s;
    }

    std::size_t VectorRuntime::MemTableBytesUsed() const noexcept {
        return mem_ ? mem_->BytesUsed() : 0u;
    }

    pomai::Status VectorRuntime::Start()
    {
        if (started_)
            return pomai::Status::Busy("shard already started");
        started_ = true;

        mem_manager_.Initialize(nullptr);

        if (mem_ && mem_->GetCount() > 0)
            (void)RotateMemTable();

        auto st = LoadSegments();
        if (!st.ok()) {
            started_ = false;
            return st;
        }
        return pomai::Status::Ok();
    }

    pomai::Status VectorRuntime::LoadSegments()
    {
        std::vector<std::string> seg_names;
        auto st = SegmentManifest::Load(data_dir_, &seg_names);
        if (!st.ok()) return st;
        
        POMAI_LOG_INFO("[runtime:{}] Loading {} segments from {}", runtime_id_, seg_names.size(), data_dir_);
        
        segments_.clear();
        for (const auto& name : seg_names) {
            std::string path = (fs::path(data_dir_) / name).string();
            table::SegmentReader::Ptr reader(nullptr, table::SegmentReader::PallocDeleter);
            st = table::SegmentReader::Open(path, &reader);
            if (!st.ok()) return st;
            segments_.push_back(std::shared_ptr<table::SegmentReader>(std::move(reader)));
        }

        PublishSnapshot();
        return pomai::Status::Ok();
    }

    // -------------------------
    // Snapshot & Rotation
    // -------------------------
    void VectorRuntime::PublishSnapshot()
    {
        auto snap = std::make_shared<VectorSnapshot>();
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

        current_snapshot_ = snap;
    }

    pomai::Status VectorRuntime::RotateMemTable()
    {
        if (mem_->GetCount() == 0) return pomai::Status::Ok();
        frozen_mem_.push_back(mem_);
        mem_ = std::make_shared<table::MemTable>(dim_, 1u << 20);
        PublishSnapshot();
        return pomai::Status::Ok();
    }

    // -------------------------
    // Sync wrappers
    // -------------------------

    pomai::Status VectorRuntime::Put(pomai::VectorId id, std::span<const float> vec)
    {
        return Put(id, vec, pomai::Metadata());
    }

    pomai::Status VectorRuntime::Put(pomai::VectorId id, std::span<const float> vec, const pomai::Metadata& meta)
    {
        if (kind_ != pomai::MembraneKind::kVector) {
            return pomai::Status::InvalidArgument("VECTOR membrane required for Put");
        }
        if (vec.size() != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");
        if (!started_)
            return pomai::Status::Aborted("shard not started");

        PutCmd cmd;
        cmd.id = id;
        cmd.vec = pomai::VectorView(vec);
        cmd.meta = meta;
        pomai::Status st = HandlePut(cmd);
        if (st.ok()) ++ops_processed_;
        return st;
    }
// ... (BatchPut skipped) ...

    pomai::Status VectorRuntime::HandlePut(PutCmd &c)
    {
        if (kind_ != pomai::MembraneKind::kVector) {
            return pomai::Status::InvalidArgument("VECTOR membrane required for Put");
        }
        if (c.vec.dim != dim_)
            return pomai::Status::InvalidArgument("dim mismatch");

        std::shared_ptr<table::MemTable> m = mem_;
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

    pomai::Status VectorRuntime::PutBatch(const std::vector<pomai::VectorId>& ids,
                                          const std::vector<std::span<const float>>& vectors)
    {
        if (kind_ != pomai::MembraneKind::kVector) {
            return pomai::Status::InvalidArgument("VECTOR membrane required for PutBatch");
        }
        if (ids.size() != vectors.size())
            return pomai::Status::InvalidArgument("ids and vectors size mismatch");
        if (ids.empty())
            return pomai::Status::Ok();
        for (const auto& vec : vectors) {
            if (vec.size() != dim_)
                return pomai::Status::InvalidArgument("dim mismatch");
        }
        if (!started_)
            return pomai::Status::Aborted("shard not started");

        BatchPutCmd cmd;
        cmd.ids = ids;
        cmd.vectors.reserve(vectors.size());
        for (const auto& vec : vectors)
            cmd.vectors.emplace_back(vec);
        pomai::Status st = HandleBatchPut(cmd);
        if (st.ok()) ++ops_processed_;
        return st;
    }

    pomai::Status VectorRuntime::Delete(pomai::VectorId id)
    {
        if (!started_)
            return pomai::Status::Aborted("shard not started");
        DelCmd c;
        c.id = id;
        pomai::Status st = HandleDel(c);
        if (st.ok()) ++ops_processed_;
        return st;
    }

    pomai::Status VectorRuntime::Get(pomai::VectorId id, std::vector<float> *out)
    {
        return Get(id, out, nullptr);
    }

    pomai::Status VectorRuntime::Get(pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta)
    {
        if (!out) return Status::InvalidArgument("out is null");

        auto active = mem_;
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

    pomai::Status VectorRuntime::GetFromSnapshot(std::shared_ptr<VectorSnapshot> snap, pomai::VectorId id, std::vector<float> *out, pomai::Metadata* out_meta) {
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

    pomai::Status VectorRuntime::Exists(pomai::VectorId id, bool *exists)
    {
        if (!exists) return Status::InvalidArgument("exists is null");

        auto active = mem_;
        auto snap = GetSnapshot();
        if (!snap) return Status::Aborted("shard not ready");

        const auto lookup = LookupById(active, snap, id, dim_);
        *exists = (lookup.state == LookupState::kFound);
        return Status::Ok();
    }



    std::pair<pomai::Status, bool> VectorRuntime::ExistsInSnapshot(std::shared_ptr<VectorSnapshot> snap, pomai::VectorId id) {
        const auto lookup = LookupById(nullptr, snap, id, dim_);
        return {Status::Ok(), lookup.state == LookupState::kFound};
    }

    pomai::Status VectorRuntime::GetSemanticPointer(std::shared_ptr<VectorSnapshot> snap, pomai::VectorId id, pomai::SemanticPointer* out) {
        if (!snap) return pomai::Status::InvalidArgument("snapshot null");
        // Look in segments only since memtables are not zero-copy aligned
        for (const auto& seg : snap->segments) {
             const uint8_t* raw_payload = nullptr;
             if (seg->FindRaw(id, &raw_payload, nullptr) == table::SegmentReader::FindResult::kFound) {
                 out->raw_data_ptr = raw_payload;
                 out->dim = seg->Dim();
                  out->quant_type = static_cast<int>(seg->GetQuantType());
                  if (seg->GetQuantType() == pomai::QuantizationType::kSq8) {
                      auto* sq8 = static_cast<const core::ScalarQuantizer8Bit*>(seg->GetQuantizer());
                      out->quant_min = sq8->GetGlobalMin();
                      out->quant_inv_scale = sq8->GetGlobalInvScale();
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

    pomai::Status VectorRuntime::Flush()
    {
        if (!started_) return pomai::Status::Aborted("shard not started");
        FlushCmd c;
        return HandleFlush(c);
    }

    pomai::Status VectorRuntime::Freeze()
    {
        if (!started_) return pomai::Status::Aborted("shard not started");
        FreezeCmd c;
        auto st = HandleFreeze(c);
        return st.has_value() ? *st : pomai::Status::Aborted("freeze not completed");
    }

    pomai::Status VectorRuntime::Compact()
    {
        if (!started_) return pomai::Status::Aborted("shard not started");
        CompactCmd c;
        auto st = HandleCompact(c);
        return st.has_value() ? *st : pomai::Status::Aborted("compact not completed");
    }

    pomai::Status VectorRuntime::NewIterator(std::unique_ptr<pomai::SnapshotIterator>* out)
    {
        if (!started_) return pomai::Status::Aborted("shard not started");
        IteratorCmd cmd;
        IteratorReply reply = HandleIterator(cmd);
        if (!reply.st.ok())
            return reply.st;
        *out = std::move(reply.iterator);
        return pomai::Status::Ok();
    }

    pomai::Status VectorRuntime::NewIterator(std::shared_ptr<VectorSnapshot> snap, std::unique_ptr<pomai::SnapshotIterator>* out)
    {
        *out = std::make_unique<VectorIterator>(std::move(snap));
        return pomai::Status::Ok();
    }

    // LOCK-FREE SEARCH
    pomai::Status VectorRuntime::Search(std::span<const float> query,
                                       std::uint32_t topk,
                                       std::vector<pomai::SearchHit> *out)
    {
        return Search(query, topk, SearchOptions{}, out);
    }

    pomai::Status VectorRuntime::Search(std::span<const float> query,
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
    // Handlers (single-threaded: invoked directly from Put/Delete/Search/etc.)
    // -------------------------



    pomai::Status VectorRuntime::HandleBatchPut(BatchPutCmd &c)
    {
        if (kind_ != pomai::MembraneKind::kVector) {
            return pomai::Status::InvalidArgument("VECTOR membrane required for PutBatch");
        }
        // Validation (already done in PutBatch, but belt-and-suspenders)
        if (c.ids.size() != c.vectors.size())
            return pomai::Status::InvalidArgument("ids and vectors size mismatch");

        std::shared_ptr<table::MemTable> m = mem_;
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

    pomai::Status VectorRuntime::HandleDel(DelCmd &c)
    {
        auto st = wal_->AppendDelete(c.id);
        if (!st.ok())
            return st;
        ++wal_epoch_;

        st = mem_->Delete(c.id);
        if (!st.ok())
            return st;

        // Tombstones live in the active memtable only; frozen/segments are immutable.
        // Search order: Active -> Frozen (newest first) -> Segments. Snapshot does not
        // include the active memtable's tail, so reads may lag by at most one freeze
        // cycle. Visibility: "newest wins" per id; deletes are applied as tombstones.
        
        (void)ivf_->Delete(c.id);
        return pomai::Status::Ok();
    }

    pomai::Status VectorRuntime::HandleFlush(FlushCmd &)
    {
        POMAI_LOG_INFO("[runtime:{}] Flushing memtables to segment", runtime_id_);
        return wal_->Flush();
    }

    // -------------------------
    // HandleFreeze: Budgeted background freeze pipeline
    // -------------------------

    std::optional<pomai::Status> VectorRuntime::HandleFreeze(FreezeCmd & /*c*/)
    {
        if (background_job_) {
            return pomai::Status::Busy("background job already running");
        }

        // Step 1: Rotate Active → Frozen (idempotent if already empty)
        if (mem_->GetCount() > 0) {
            auto st = RotateMemTable();
            if (!st.ok()) {
                return pomai::Status::Internal(std::string("Freeze: RotateMemTable failed: ") + st.message());
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
        background_job_ = std::move(job);
        last_background_result_.reset();
        const auto start = std::chrono::steady_clock::now();
        while (background_job_) {
            PumpBackgroundWork(kBackgroundBudget);
            if (std::chrono::steady_clock::now() - start > kBackgroundMaxSyncDuration) {
                CancelBackgroundJob("Freeze timed out");
                break;
            }
        }
        if (last_background_result_.has_value()) {
            return last_background_result_;
        }
        return std::optional<pomai::Status>(pomai::Status::Aborted("Freeze timed out"));
    }

    // -------------------------
    // HandleCompact: Budgeted background compaction
    // -------------------------

    std::optional<pomai::Status> VectorRuntime::HandleCompact(CompactCmd & /*c*/)
    {
        POMAI_LOG_INFO("[runtime:{}] Starting background compaction", runtime_id_);
        if (background_job_) return std::nullopt; // Keep in queue

        // 1. Calculate stats for compaction manager
        std::vector<storage::CompactionManager::LevelStats> stats;
        uint64_t total_size = 0;
        for (const auto& seg : segments_) {
            // Best effort file size estimation
            total_size += fs::file_size(seg->Path());
        }

        storage::CompactionManager::LevelStats l0_stats;
        l0_stats.level = 0;
        l0_stats.file_count = static_cast<uint32_t>(segments_.size());
        l0_stats.total_size = total_size;
        l0_stats.score = static_cast<double>(segments_.size()) / 4.0; // Assume 4 files trigger L0->L1
        stats.push_back(l0_stats);

        // 2. Pick task
        auto task = compaction_manager_->PickCompaction(stats);
        if (!task.valid && segments_.size() <= 1) {
            return pomai::Status::Ok(); // Nothing to compact
        }

        // If manual compaction and we have multiple segments, force L0->L1
        if (!task.valid) {
            task.input_level = 0;
            task.output_level = 1;
            task.valid = true;
        }

        BackgroundJob::CompactState state;
        state.input_segments = segments_;
        state.old_segments = segments_;
        auto job = std::make_unique<BackgroundJob>(BackgroundJob::Type::kCompact, std::move(state));
        background_job_ = std::move(job);
        last_background_result_.reset();
        const auto start = std::chrono::steady_clock::now();
        while (background_job_) {
            PumpBackgroundWork(kBackgroundBudget);
            if (std::chrono::steady_clock::now() - start > kBackgroundMaxSyncDuration) {
                CancelBackgroundJob("Compaction timed out");
                break;
            }
        }
        if (last_background_result_.has_value()) {
            return last_background_result_;
        }
        return std::optional<pomai::Status>(pomai::Status::Aborted("Compaction timed out"));
    }

    IteratorReply VectorRuntime::HandleIterator(IteratorCmd &c)
    {
        (void)c;  // unused parameter
        
        // Snapshot must exist (created in Start/LoadSegments or after rotate).
        auto base = current_snapshot_;
        if (!base) {
            IteratorReply reply;
            reply.st = pomai::Status::Internal("HandleIterator: snapshot is null");
            return reply;
        }

        // Include live memtable in the iterator view so unflushed data is visible.
        // Otherwise NewIterator() would see 0 vectors when all data is still in mem_.
        std::shared_ptr<VectorSnapshot> snapshot;
        if (mem_ && mem_->GetCount() > 0) {
            snapshot = std::make_shared<VectorSnapshot>();
            snapshot->version = base->version;
            snapshot->created_at = base->created_at;
            snapshot->segments = base->segments;
            snapshot->frozen_memtables = base->frozen_memtables;
            snapshot->live_memtable = mem_;
        } else {
            snapshot = base;
        }
        
        auto shard_iter = std::make_unique<VectorIterator>(snapshot);
        IteratorReply reply;
        reply.st = pomai::Status::Ok();
        reply.iterator = std::move(shard_iter);
        return reply;
    }

    SearchReply VectorRuntime::HandleSearch(SearchCmd &c)
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

    void VectorRuntime::CancelBackgroundJob(const std::string& reason)
    {
        if (!background_job_) return;
        last_background_result_ = pomai::Status::Aborted(reason);
        background_job_.reset();
    }

    void VectorRuntime::PumpBackgroundWork(std::chrono::milliseconds budget)
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
            last_background_result_ = st;
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
                        // std::cout << "[VectorRuntime] Freeze: Switching to CommitManifest" << std::endl;
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
                        state.filepath = (fs::path(data_dir_) / state.filename).string();
                        state.builder = std::make_unique<table::SegmentBuilder>(state.filepath, dim_, index_params_, metric_);
                    }

                    pomai::Metadata meta_copy = entry.meta ? *entry.meta : pomai::Metadata();
                    auto st = state.builder->Add(entry.id, pomai::VectorView(entry.vec), entry.is_deleted, meta_copy);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal(std::string("Freeze: SegmentBuilder::Add failed: ") + st.message()));
                        return;
                    }

                    // Feed Streaming IVF for continuous SOM updates
                    if (!entry.is_deleted) {
                         st = ivf_->Put(entry.id, std::span<const float>(entry.vec));
                         if (!st.ok()) {
                             complete_job(pomai::Status::Internal(std::string("Freeze: IVF::Put failed: ") + st.message()));
                             return;
                         }
                    }

                    bg_budget.Consume();

                    if (state.builder->Count() >= kMaxSegmentEntries) {
                        // std::cout << "[VectorRuntime] Freeze: Segment full, finalizing" << std::endl;
                        state.memtable_done_after_finalize = false;
                        state.phase = BackgroundJob::Phase::kFinalizeSegment;
                    }
                } else if (state.phase == BackgroundJob::Phase::kFinalizeSegment) {
                    // std::cout << "[VectorRuntime] Freeze: Finalizing segment..." << std::endl;
                    auto st = state.builder->BuildIndex();
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal(std::string("Freeze: BuildIndex failed: ") + st.message()));
                        return;
                    }
                    st = state.builder->Finish();
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal(std::string("Freeze: SegmentBuilder::Finish failed: ") + st.message()));
                        return;
                    }
                    st = pomai::util::FsyncDir(data_dir_);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal(std::string("Freeze: FsyncDir after segment failed: ") + st.message()));
                        return;
                    }

                    table::SegmentReader::Ptr reader(nullptr, table::SegmentReader::PallocDeleter);
                    st = table::SegmentReader::Open(state.filepath, &reader);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal(std::string("Freeze: SegmentReader::Open failed: ") + st.message()));
                        return;
                    }

                    state.built_segments.push_back({state.filename, state.filepath, std::shared_ptr<table::SegmentReader>(std::move(reader))});
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
                    auto st = SegmentManifest::Load(data_dir_, &seg_names);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal(std::string("Freeze: SegmentManifest::Load failed: ") + st.message()));
                        return;
                    }

                    for (auto it = state.built_segments.rbegin(); it != state.built_segments.rend(); ++it) {
                        seg_names.insert(seg_names.begin(), it->filename);
                    }

                    st = SegmentManifest::Commit(data_dir_, seg_names);
                    if (!st.ok()) {
                        complete_job(pomai::Status::Internal(std::string("Freeze: SegmentManifest::Commit failed: ") + st.message()));
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
                            complete_job(pomai::Status::Internal(std::string("Freeze: WAL::Reset failed: ") + st.message()));
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
                                if (state.input_segments[top.seg_idx]->GetQuantType() != pomai::QuantizationType::kNone) {
                                    state.compact_buffers.push_back(std::move(vec_decoded));
                                    vec_mapped = std::span<const float>(state.compact_buffers.back());
                                }
                                if (!state.builder) {
                                    auto sys_now = std::chrono::system_clock::now().time_since_epoch().count();
                                    state.filename = "seg_" + std::to_string(sys_now) + "_compacted_" +
                                                     std::to_string(state.segment_part) + ".dat";
                                    state.filepath = (fs::path(data_dir_) / state.filename).string();
                                    state.builder = std::make_unique<table::SegmentBuilder>(state.filepath, dim_, index_params_, metric_);
                                }
                                auto st = state.builder->Add(top.id, pomai::VectorView(vec_mapped), false, meta);
                                if (!st.ok()) {
                                    complete_job(pomai::Status::Internal(std::string("Compact: SegmentBuilder::Add failed: ") + st.message()));
                                    return;
                                }

                                // Feed downstream Streaming IVF
                                st = ivf_->Put(top.id, vec_mapped);
                                if (!st.ok()) {
                                     complete_job(pomai::Status::Internal(std::string("Compact: IVF::Put failed: ") + st.message()));
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
                    complete_job(pomai::Status::Internal(std::string("Compact: BuildIndex failed: ") + st.message()));
                    return;
                }
                st = state.builder->Finish();
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal(std::string("Compact: SegmentBuilder::Finish failed: ") + st.message()));
                    return;
                }
                st = pomai::util::FsyncDir(data_dir_);
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal(std::string("Compact: FsyncDir after segment failed: ") + st.message()));
                    return;
                }

                table::SegmentReader::Ptr reader(nullptr, table::SegmentReader::PallocDeleter);
                st = table::SegmentReader::Open(state.filepath, &reader);
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal(std::string("Compact: SegmentReader::Open failed: ") + st.message()));
                    return;
                }

                state.built_segments.push_back({state.filename, state.filepath, std::shared_ptr<table::SegmentReader>(std::move(reader))});
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
                auto st = SegmentManifest::Commit(data_dir_, seg_names);
                if (!st.ok()) {
                    complete_job(pomai::Status::Internal(std::string("Compact: SegmentManifest::Commit failed: ") + st.message()));
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
                    std::string p = old->Path();
                    fs::remove(p, ec);
                    if (ec) {
                        // POMAI_LOG_ERROR("Cleanup failed: {}", ec.message());
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
    }

    pomai::Status VectorRuntime::SearchBatchLocal(std::span<const float> queries,
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
        auto active = mem_;

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

        const std::size_t num_queries = queries.size() / dim_;
        search_query_sums_scratch_.resize(num_queries, 0.0f);
        for (uint32_t q_idx : query_indices) {
            std::span<const float> q(queries.data() + q_idx * dim_, dim_);
            float s = 0.0f;
            for (float f : q) s += f;
            search_query_sums_scratch_[q_idx] = s;
        }

        // Sequential path (single-threaded event loop).
        for (uint32_t q_idx : query_indices) {
            std::span<const float> single_query(queries.data() + (q_idx * dim_), dim_);
            float q_sum = search_query_sums_scratch_[q_idx];
            auto st = SearchLocalInternal(active, snap, single_query, q_sum, topk, opts, shared_policy, use_visibility, &(*out_results)[q_idx], false);
            if (!st.ok()) return st;
        }

        return pomai::Status::Ok();
    }

// -------------------------
    // SearchLocalInternal: DB-grade 1-pass merge scan
    // -------------------------

    pomai::Status VectorRuntime::SearchLocalInternal(
            std::shared_ptr<table::MemTable> active,
            std::shared_ptr<VectorSnapshot> snap,
            std::span<const float> query,
            float query_sum,
            std::uint32_t topk,
            const SearchOptions& opts,
            SearchMergePolicy& merge_policy,
            bool use_visibility,
            std::vector<pomai::SearchHit> *out,
            bool /*use_pool*/)
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
        // Phase 2: Parallel scoring over authoritative sources (reuse scratch to reduce allocations)
        // -------------------------
        search_candidates_scratch_.clear();
        search_candidates_scratch_.reserve(std::min(static_cast<std::size_t>(topk) * 4, static_cast<std::size_t>(4096)));

        bool has_filters = !opts.filters.empty();
        uint32_t effective_nprobe = index_params_.nprobe == 0 ? 1 : index_params_.nprobe;

        // If we expect to hit many candidates but nprobe is small, try to increase nprobe instead of full scan
        if (has_filters && effective_nprobe < 8) {
            effective_nprobe = std::min<uint32_t>(32u, effective_nprobe * 8); // Heuristic to avoid brute force
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

        std::uint64_t total_scanned = 0;
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
                        total_scanned += local_scanned;
                        return local.Drain();
                    }
                }
                if (seg->GetQuantType() != pomai::QuantizationType::kNone) {
                    const auto quant_type = seg->GetQuantType();
                    float q_min = 0.0f;
                    float q_inv_scale = 0.0f;
                    if (quant_type == pomai::QuantizationType::kSq8) {
                        auto* sq8 = static_cast<const core::ScalarQuantizer8Bit*>(seg->GetQuantizer());
                        q_min = sq8->GetGlobalMin();
                        q_inv_scale = sq8->GetGlobalInvScale();
                    }

                    thread_local std::vector<uint32_t> cand_reuse;
                    cand_reuse.clear();
                    if (seg->Search(query, effective_nprobe, &cand_reuse).ok()) {
                        used_candidates = true; // Mark as used candidates for fast path
                        for (uint32_t idx : cand_reuse) {
                            local_scanned++;
                            const uint8_t* p = seg->GetBaseAddr() + seg->GetEntriesStartOffset() + idx * seg->GetEntrySize();
                            const uint8_t* codes_ptr = p + 12; // Assuming ID (8 bytes) + is_deleted (1 byte) + metadata_len (3 bytes) = 12 bytes offset
                            if (!(*(p+8) & 0x01)) { // not tombstone, assuming is_deleted is at offset 8
                                float score = 0.0f;
                                const bool is_ip = (this->metric_ == pomai::MetricType::kInnerProduct || this->metric_ == pomai::MetricType::kCosine);
                                if (quant_type == pomai::QuantizationType::kSq8) {
                                    if (is_ip) {
                                        score = pomai::core::DotSq8(query, std::span<const uint8_t>(codes_ptr, dim_), q_min, q_inv_scale, query_sum);
                                    } else {
                                        const float q_max = q_min + 255.0f * q_inv_scale;
                                        score = -pomai::core::L2SqSq8(query, std::span<const uint8_t>(codes_ptr, dim_), q_min, q_max);
                                    }
                                } else if (quant_type == pomai::QuantizationType::kFp16) {
                                    if (is_ip) {
                                        score = pomai::core::DotFp16(query, {reinterpret_cast<const uint16_t*>(codes_ptr), dim_});
                                    } else {
                                        score = -pomai::core::L2SqFp16(query, {reinterpret_cast<const uint16_t*>(codes_ptr), dim_});
                                    }
                                }
                                local.Push(*(uint64_t*)p, score);
                            }
                        }
                        total_scanned += local_scanned;
                        return local.Drain();
                    }
                }
            }
            
            thread_local std::vector<uint32_t> cand_idxs_reuse;
            cand_idxs_reuse.clear();
            auto cand_status = pomai::Status::Ok();
            if (seg->Count() >= index_params_.adaptive_threshold && seg->HasIndex()) {
                cand_status = seg->Search(query, effective_nprobe, &cand_idxs_reuse);
            }
            if (cand_status.ok() && !cand_idxs_reuse.empty()) {
                std::sort(cand_idxs_reuse.begin(), cand_idxs_reuse.end());
                cand_idxs_reuse.erase(std::unique(cand_idxs_reuse.begin(), cand_idxs_reuse.end()), cand_idxs_reuse.end());

                if (!cand_idxs_reuse.empty()) {
                    used_candidates = true;
                    pomai::Metadata local_meta;
                    pomai::Metadata* meta_ptr = has_filters ? &local_meta : nullptr;
                    
                    if (seg->GetQuantType() != pomai::QuantizationType::kNone) {
                        const auto quant_type = seg->GetQuantType();
                        float q_min = 0.0f;
                        float q_inv_scale = 0.0f;
                        if (quant_type == pomai::QuantizationType::kSq8) {
                            auto* sq8 = static_cast<const core::ScalarQuantizer8Bit*>(seg->GetQuantizer());
                            q_min = sq8->GetGlobalMin();
                            q_inv_scale = sq8->GetGlobalInvScale();
                        }

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

                            float score = 0.0f;
                            const bool is_ip = (this->metric_ == pomai::MetricType::kInnerProduct || this->metric_ == pomai::MetricType::kCosine);
                            if (quant_type == pomai::QuantizationType::kSq8) {
                                if (is_ip) {
                                    score = pomai::core::DotSq8(query, codes, q_min, q_inv_scale, query_sum);
                                } else {
                                    const float q_max = q_min + 255.0f * q_inv_scale;
                                    score = -pomai::core::L2SqSq8(query, codes, q_min, q_max);
                                }
                            } else if (quant_type == pomai::QuantizationType::kFp16) {
                                if (is_ip) {
                                    score = pomai::core::DotFp16(query, {reinterpret_cast<const uint16_t*>(codes.data()), codes.size()/2});
                                } else {
                                    score = -pomai::core::L2SqFp16(query, {reinterpret_cast<const uint16_t*>(codes.data()), codes.size()/2});
                                }
                            }
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
            total_scanned += local_scanned;
            return local.Drain();
        };

        {
            auto [hits, scanned] = score_memtable(active);
            total_scanned += scanned;
            search_candidates_scratch_.insert(search_candidates_scratch_.end(), hits.begin(), hits.end());
        }

        for (auto it = snap->frozen_memtables.rbegin(); it != snap->frozen_memtables.rend(); ++it) {
            auto [hits, scanned] = score_memtable(*it);
            total_scanned += scanned;
            search_candidates_scratch_.insert(search_candidates_scratch_.end(), hits.begin(), hits.end());
        }

        search_segment_hits_scratch_.resize(snap->segments.size());
        for (std::size_t i = 0; i < snap->segments.size(); ++i) {
            search_segment_hits_scratch_[i].clear();
            search_segment_hits_scratch_[i] = score_segment(snap->segments[i]);
        }

        last_query_candidates_scanned_ += total_scanned;

        for (const auto& hits : search_segment_hits_scratch_) {
            search_candidates_scratch_.insert(search_candidates_scratch_.end(), hits.begin(), hits.end());
        }

        std::sort(search_candidates_scratch_.begin(), search_candidates_scratch_.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            return a.id < b.id;
        });

        if (search_candidates_scratch_.size() > topk) {
            search_candidates_scratch_.resize(topk);
        }

        out->assign(search_candidates_scratch_.begin(), search_candidates_scratch_.end());
        return pomai::Status::Ok();
    }
} // namespace pomai::core
