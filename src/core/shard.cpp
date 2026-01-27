#include "shard.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <utility>

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

        struct Node
        {
            float score;
            Id id;
        };

        // min-heap where top() is worst kept
        auto cmp = [](const Node &a, const Node &b)
        { return a.score > b.score; };
        std::priority_queue<Node, std::vector<Node>, decltype(cmp)> heap(cmp);

        auto feed = [&](const std::vector<SearchResultItem> &items)
        {
            for (const auto &it : items)
            {
                if (heap.size() < k)
                {
                    heap.push(Node{it.score, it.id});
                }
                else if (it.score > heap.top().score)
                {
                    heap.pop();
                    heap.push(Node{it.score, it.id});
                }
            }
        };

        if (!out.items.empty())
            feed(out.items);
        feed(in.items);

        std::vector<SearchResultItem> merged;
        merged.reserve(heap.size());
        while (!heap.empty())
        {
            auto n = heap.top();
            heap.pop();
            merged.push_back(SearchResultItem{n.id, n.score});
        }
        std::reverse(merged.begin(), merged.end());
        out.items = std::move(merged);
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
            segments_[segment_pos].index = std::move(idx);
            if (log_info_)
                log_info_("[" + name_ + "] Indexed segment: " + std::to_string(snap->ids.size()) + " vectors");
            else
                std::cout << "[" << name_ << "] Indexed segment: " << snap->ids.size() << " vectors\n";
            return;
        }

        // fallback by pointer
        for (auto &s : segments_)
        {
            if (s.snap == snap)
            {
                s.index = std::move(idx);
                if (log_info_)
                    log_info_("[" + name_ + "] Indexed segment: " + std::to_string(snap->ids.size()) + " vectors");
                else
                    std::cout << "[" << name_ << "] Indexed segment: " << snap->ids.size() << " vectors\n";
                return;
            }
        }
    }

} // namespace pomai