#include "shard.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <chrono>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <random>

#include "fixed_topk.h"

namespace pomai
{

    Shard::Shard(std::string name,
                 std::size_t dim,
                 std::size_t queue_cap,
                 std::string wal_dir,
                 LogFn info,
                 LogFn error)
        : name_(std::move(name)),
          wal_dir_(std::move(wal_dir)),
          wal_(name_, wal_dir_, dim),
          seed_(dim),
          ingest_q_(queue_cap),
          log_info_(std::move(info)),
          log_error_(std::move(error))
    {
        live_snap_ = seed_.MakeSnapshot();
    }

    Shard::~Shard()
    {
        Stop();
    }

    void Shard::Start()
    {
        // Replay WAL first (verify/truncate any trailing partial records) before starting
        // the WAL background fsync thread. This avoids races between replay/truncate and
        // the append/fsync thread.
        try
        {
            WalReplayStats stats = wal_.ReplayToSeed(seed_);

            // Log outcome so operator knows whether WAL recovery happened
            if (log_info_)
            {
                if (stats.records_applied == 0)
                {
                    log_info_("[" + name_ + "] WAL replay: no records recovered or wal missing");
                }
                else
                {
                    std::string msg = "[" + name_ + "] WAL replay: records=" + std::to_string(stats.records_applied) +
                                      ", vectors=" + std::to_string(stats.vectors_applied) +
                                      ", last_lsn=" + std::to_string(stats.last_lsn);
                    if (stats.truncated_bytes > 0)
                        msg += ", truncated_bytes=" + std::to_string(stats.truncated_bytes);
                    log_info_(msg);
                }
            }
            else
            {
                if (stats.records_applied == 0)
                {
                    std::cout << "[" << name_ << "] WAL replay: no records recovered or wal missing\n";
                }
                else
                {
                    std::cout << "[" << name_ << "] WAL replay: records=" << stats.records_applied
                              << ", vectors=" << stats.vectors_applied
                              << ", last_lsn=" << stats.last_lsn;
                    if (stats.truncated_bytes > 0)
                        std::cout << ", truncated_bytes=" << stats.truncated_bytes;
                    std::cout << "\n";
                }
            }
        }
        catch (const std::exception &e)
        {
            if (log_error_)
                log_error_("[" + name_ + "] WAL replay failed: " + std::string(e.what()));
            else
                std::cerr << "[" << name_ << "] WAL replay failed: " << e.what() << "\n";
            // Continue startup; seed remains as-is (empty or partially restored).
        }

        {
            std::lock_guard<std::mutex> lk(state_mu_);
            live_snap_ = seed_.MakeSnapshot();
        }

        // If replay loaded data, freeze it so it can be indexed
        if (seed_.Count() > 0)
        {
            MaybeFreezeSegment();
        }

        // Start WAL background thread after replay/truncation
        wal_.Start();

        owner_ = std::thread(&Shard::RunLoop, this);
    }

    void Shard::Stop()
    {
        ingest_q_.Close();
        if (owner_.joinable())
            owner_.join();
        wal_.Stop();
    }

    std::future<Lsn> Shard::EnqueueUpserts(std::vector<UpsertRequest> batch, bool wait_durable)
    {
        UpsertTask t;
        t.batch = std::move(batch);
        t.wait_durable = wait_durable;
        auto fut = t.done.get_future();

        if (!ingest_q_.Push(std::move(t)))
        {
            std::promise<Lsn> p;
            auto f = p.get_future();
            p.set_exception(std::make_exception_ptr(std::runtime_error("shard queue closed")));
            return f;
        }
        return fut;
    }

    std::future<bool> Shard::RequestCheckpoint()
    {
        UpsertTask t;
        t.is_checkpoint = true;
        t.checkpoint_done.emplace(); // construct optional promise

        auto fut = t.checkpoint_done->get_future();
        // Push into ingest queue like normal tasks
        if (!ingest_q_.Push(std::move(t)))
        {
            std::promise<bool> p;
            p.set_exception(std::make_exception_ptr(std::runtime_error("shard queue closed")));
            return p.get_future();
        }
        return fut;
    }

    void Shard::RequestEmergencyFreeze()
    {
        bool expected = false;
        if (!emergency_freeze_pending_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
            return;

        UpsertTask t;
        t.is_emergency_freeze = true;
        t.wait_durable = false;
        if (!ingest_q_.Push(std::move(t)))
        {
            emergency_freeze_pending_.store(false, std::memory_order_release);
        }
    }

    std::size_t Shard::ApproxCountUnsafe() const
    {
        return seed_.Count();
    }

    void Shard::MergeTopK(SearchResponse &out, const SearchResponse &in, std::size_t k)
    {
        if (k == 0)
        {
            out.items.clear();
            return;
        }
        if (in.items.empty())
            return;

        FixedTopK topk(k);

        auto feed = [&](const std::vector<SearchResultItem> &items)
        {
            for (const auto &it : items)
                topk.Push(it.score, it.id);
        };

        if (!out.items.empty())
            feed(out.items);
        feed(in.items);

        topk.FillSorted(out.items);
    }

    SearchResponse Shard::Search(const SearchRequest &req, const pomai::ai::Budget &budget) const
    {
        std::vector<IndexedSegment> segs;
        Seed::Snapshot live;

        {
            std::lock_guard<std::mutex> lk(state_mu_);
            segs = segments_;
            live = live_snap_;
        }

        SearchResponse out;

        // 1) frozen segments
        for (const auto &s : segs)
        {
            if (s.index)
            {
                auto r = s.index->Search(req.query, budget);
                MergeTopK(out, r, req.topk);
            }
            else if (s.snap)
            {
                auto r = Seed::SearchSnapshot(s.snap, req);
                MergeTopK(out, r, req.topk);
            }
        }

        // 2) live memtable snapshot
        if (live)
        {
            auto r = Seed::SearchSnapshot(live, req);
            MergeTopK(out, r, req.topk);
        }

        return out;
    }

    void Shard::RunLoop()
    {
        while (true)
        {
            auto opt = ingest_q_.Pop();
            if (!opt)
                break;

            UpsertTask task = std::move(*opt);

            // Handle checkpoint control task
            if (task.is_checkpoint)
            {
                bool ok = false;
                try
                {
                    // Local helper to throw with errno message
                    auto ThrowLocal = [](const std::string &what)
                    {
                        throw std::runtime_error(what + ": " + std::string(std::strerror(errno)));
                    };

                    // 1) Ensure WAL up to currently written LSN is durable
                    Lsn last_written = wal_.WrittenLsn();
                    if (last_written != 0)
                        wal_.WaitDurable(last_written);

                    // 2) Create snapshot of seed (atomic)
                    auto snap = seed_.MakeSnapshot();

                    // 3) Serialize snapshot to checkpoint file
                    std::string cp_dir = wal_dir_;
                    std::filesystem::create_directories(cp_dir);
                    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
                    std::string path = cp_dir + "/checkpoint-" + std::to_string(ts) + ".bin";

                    int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
                    if (fd < 0)
                        ThrowLocal("open checkpoint file failed");

                    // Header: dim (u16), count (u64)
                    uint16_t dim16 = static_cast<uint16_t>(snap->dim);
                    uint64_t count64 = static_cast<uint64_t>(snap->ids.size());

#if __BYTE_ORDER == __BIG_ENDIAN
                    uint16_t dim_be = __builtin_bswap16(dim16);
                    uint64_t count_be = __builtin_bswap64(count64);
                    if (::write(fd, &dim_be, sizeof(dim_be)) != sizeof(dim_be))
                        ThrowLocal("write cp header");
                    if (::write(fd, &count_be, sizeof(count_be)) != sizeof(count_be))
                        ThrowLocal("write cp header");
#else
                    if (::write(fd, &dim16, sizeof(dim16)) != sizeof(dim16))
                        ThrowLocal("write cp header");
                    if (::write(fd, &count64, sizeof(count64)) != sizeof(count64))
                        ThrowLocal("write cp header");
#endif

                    // write ids
                    if (!snap->ids.empty())
                    {
                        ssize_t w = ::write(fd, snap->ids.data(), snap->ids.size() * sizeof(uint64_t));
                        if (w < 0 || static_cast<size_t>(w) != snap->ids.size() * sizeof(uint64_t))
                            ThrowLocal("write checkpoint ids failed");
                    }

                    // write data (floats)
                    if (!snap->data.empty())
                    {
                        ssize_t w2 = ::write(fd, snap->data.data(), snap->data.size() * sizeof(float));
                        if (w2 < 0 || static_cast<size_t>(w2) != snap->data.size() * sizeof(float))
                            ThrowLocal("write checkpoint data failed");
                    }

                    if (::fdatasync(fd) != 0)
                        ThrowLocal("fdatasync checkpoint file failed");
                    ::close(fd);

                    // fsync directory
                    int dfd = ::open(cp_dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
                    if (dfd >= 0)
                    {
                        if (::fsync(dfd) != 0)
                        {
                            ::close(dfd);
                            ThrowLocal("fsync checkpoint dir failed");
                        }
                        ::close(dfd);
                    }

                    // 4) Truncate WAL to zero safely
                    wal_.TruncateToZero();

                    ok = true;
                }
                catch (...)
                {
                    ok = false;
                }

                // Fulfill checkpoint promise so caller knows outcome
                try
                {
                    if (task.checkpoint_done)
                        task.checkpoint_done->set_value(ok);
                }
                catch (...)
                {
                }
                continue; // proceed to next loop iteration
            }

            if (task.is_emergency_freeze)
            {
                MaybeFreezeSegment();
                emergency_freeze_pending_.store(false, std::memory_order_release);
                continue;
            }

            // --- Normal upsert path ---
            try
            {
                // 1) WAL: pass through wait_durable flag so AppendUpserts can optionally fsync immediately.
                Lsn lsn = wal_.AppendUpserts(task.batch, task.wait_durable);

                // 2) Seed apply
                seed_.ApplyUpserts(task.batch);

                // 3) publish live snapshot sometimes
                since_live_publish_ += task.batch.size();
                if (since_live_publish_ >= kPublishLiveEveryVectors)
                {
                    since_live_publish_ = 0;
                    auto snap = seed_.MakeSnapshot();
                    std::lock_guard<std::mutex> lk(state_mu_);
                    live_snap_ = std::move(snap);
                }

                // 4) freeze to segment sometimes
                since_freeze_ += task.batch.size();
                if (since_freeze_ >= kFreezeEveryVectors)
                {
                    since_freeze_ = 0;
                    MaybeFreezeSegment();
                }

                // 5) durable wait if requested (usually fast because AppendUpserts already fdatasync'd)
                if (task.wait_durable)
                    wal_.WaitDurable(lsn);

                task.done.set_value(lsn);
            }
            catch (...)
            {
                try
                {
                    task.done.set_exception(std::current_exception());
                }
                catch (...)
                {
                }
            }
        }

        // final freeze
        MaybeFreezeSegment();
    }

    void Shard::MaybeFreezeSegment()
    {
        auto snap = seed_.MakeSnapshot();
        if (!snap || snap->ids.empty())
            return;

        std::size_t seg_pos = 0;
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            live_snap_ = snap;
            segments_.push_back(IndexedSegment{snap, nullptr});
            seg_pos = segments_.size() - 1;
        }

        // enqueue to global pool (if configured)
        if (build_pool_)
        {
            IndexBuildPool::Job job;
            job.segment_pos = seg_pos;
            job.snap = snap;
            job.M = 48;
            job.ef_construction = 200;

            job.attach = [this](std::size_t pos,
                                Seed::Snapshot s,
                                std::shared_ptr<pomai::core::OrbitIndex> idx)
            {
                this->AttachIndex(pos, std::move(s), std::move(idx));
            };

            (void)build_pool_->Enqueue(std::move(job));
        }

        // reset memtable
        seed_ = Seed(seed_.Dim());
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            live_snap_ = seed_.MakeSnapshot();
        }
    }

    void Shard::AttachIndex(std::size_t segment_pos,
                            Seed::Snapshot snap,
                            std::shared_ptr<pomai::core::OrbitIndex> idx)
    {
        if (!snap || !idx)
            return;

        std::lock_guard<std::mutex> lk(state_mu_);

        // fast path by position
        if (segment_pos < segments_.size() && segments_[segment_pos].snap == snap)
        {
            const std::size_t indexed = snap->ids.size();
            segments_[segment_pos].index = std::move(idx);
            // Release the snapshot to free memory — index now holds the searchable data.
            segments_[segment_pos].snap.reset();

            if (log_info_)
                log_info_("[" + name_ + "] Snapshot released. Vectors indexed: " + std::to_string(indexed));
            return;
        }

        // fallback by pointer
        for (auto &s : segments_)
        {
            if (s.snap == snap)
            {
                const std::size_t indexed = snap->ids.size();
                s.index = std::move(idx);
                s.snap.reset();

                if (log_info_)
                    log_info_("[" + name_ + "] Snapshot released. Vectors indexed: " + std::to_string(indexed));
                return;
            }
        }
    }

    // -------------------- Sampling implementation --------------------
    // Return up to max_samples vectors sampled uniformly from this shard's snapshots.
    std::vector<Vector> Shard::SampleVectors(std::size_t max_samples) const
    {
        if (max_samples == 0)
            return {};

        // Acquire short lock and copy references to segment snapshots and live snapshot.
        std::vector<Seed::Snapshot> snaps;
        Seed::Snapshot live;
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            snaps.reserve(segments_.size());
            for (const auto &s : segments_)
            {
                if (s.snap)
                    snaps.push_back(s.snap);
            }
            live = live_snap_;
        }

        std::vector<Vector> reservoir;
        reservoir.reserve(std::min<std::size_t>(max_samples, 1024));

        std::mt19937_64 rng(std::random_device{}());
        std::size_t seen = 0;

        auto process_snapshot = [&](const Seed::Snapshot &snap)
        {
            if (!snap || snap->ids.empty())
                return;
            const std::size_t n = snap->ids.size();
            const std::size_t dim = snap->dim;
            for (std::size_t i = 0; i < n; ++i)
            {
                Vector v;
                v.data.resize(dim);
                const float *src = snap->data.data() + i * dim;
                std::copy(src, src + dim, v.data.begin());

                ++seen;
                if (reservoir.size() < max_samples)
                {
                    reservoir.push_back(std::move(v));
                }
                else
                {
                    std::uniform_int_distribution<std::size_t> dist(0, seen - 1);
                    std::size_t j = dist(rng);
                    if (j < max_samples)
                        reservoir[j] = std::move(v);
                }
            }
        };

        for (const auto &s : snaps)
            process_snapshot(s);
        if (live)
            process_snapshot(live);

        return reservoir;
    }

} // namespace pomai
