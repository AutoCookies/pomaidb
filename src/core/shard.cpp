#include "shard.h"
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <future>

namespace pomai
{

    Shard::Shard(std::string name, std::size_t dim, std::size_t queue_cap, std::string wal_dir)
        : name_(std::move(name)),
          wal_dir_(std::move(wal_dir)),
          wal_(name_, wal_dir_, dim),
          seed_(dim),
          ingest_q_(queue_cap)
    {
        // Init empty state
        auto empty_snap = std::make_shared<SeedSnapshot>();
        empty_snap->dim = dim;
        published_ = empty_snap;

        // Init Vortex Index (Empty)
        // M=48, ef=200 (High Quality)
        index_ = std::make_shared<pomai::core::OrbitIndex>(dim, 48, 200);
    }

    Shard::~Shard()
    {
        Stop();
    }

    void Shard::Start()
    {
        if (owner_.joinable())
            return;

        std::cout << "[" << name_ << "] Vortex Engine Online." << std::endl;

        // 1. Replay WAL
        wal_.Start();
        wal_.ReplayToSeed(seed_);

        // 2. Load data vào Index ngay khi khởi động
        if (seed_.Count() > 0)
        {
            std::cout << "[" << name_ << "] Loading " << seed_.Count() << " vectors into Vortex..." << std::endl;
            // Build batch vào Index
            index_->Build(seed_.GetFlatData(), seed_.GetFlatIds());
        }

        PublishSnapshot();
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
        UpsertTask task;
        task.batch = std::move(batch);
        task.wait_durable = wait_durable;
        auto fut = task.done.get_future();

        if (!ingest_q_.Push(std::move(task)))
        {
            std::promise<Lsn> p;
            auto f = p.get_future();
            try
            {
                throw std::runtime_error("shard queue closed");
            }
            catch (...)
            {
                p.set_exception(std::current_exception());
            }
            return f;
        }
        return fut;
    }

    void Shard::PublishSnapshot()
    {
        auto snap = seed_.MakeSnapshot();
        std::lock_guard<std::mutex> lk(state_mu_);
        published_ = std::move(snap);
    }

    SearchResponse Shard::Search(const SearchRequest &req, const pomai::ai::Budget &base_budget) const
    {
        // Lấy pointer an toàn
        std::shared_ptr<pomai::core::OrbitIndex> idx_ptr;
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            idx_ptr = index_;
        }

        if (idx_ptr)
        {
            return idx_ptr->Search(req.query, base_budget);
        }
        return {};
    }

    void Shard::RunLoop()
    {
        while (true)
        {
            auto opt = ingest_q_.Pop();
            if (!opt.has_value())
                break;
            UpsertTask task = std::move(*opt);

            // 1. Ghi WAL (Bền vững)
            Lsn lsn = wal_.AppendUpserts(task.batch);

            // 2. Cập nhật Seed (Memory Storage)
            seed_.ApplyUpserts(task.batch);

            // 3. INSERT TRỰC TIẾP VÀO INDEX (Real-time)
            // Lấy pointer hiện tại (không cần lock lâu vì index_ instance là thread-safe internal)
            std::shared_ptr<pomai::core::OrbitIndex> idx_ptr;
            {
                std::lock_guard<std::mutex> lk(state_mu_);
                idx_ptr = index_;
            }

            // Chuẩn bị data phẳng
            std::vector<float> batch_data;
            std::vector<Id> batch_ids;
            size_t dim = seed_.Dim();
            batch_data.reserve(task.batch.size() * dim);
            batch_ids.reserve(task.batch.size());

            for (const auto &req : task.batch)
            {
                batch_ids.push_back(req.id);
                batch_data.insert(batch_data.end(), req.vec.data.begin(), req.vec.data.end());
            }

            // Insert vào Vortex
            idx_ptr->InsertBatch(batch_data, batch_ids);

            // 4. Wait Durable nếu cần
            if (task.wait_durable)
                wal_.WaitDurable(lsn);

            if (++batch_since_publish_ >= kPublishEvery)
            {
                batch_since_publish_ = 0;
                PublishSnapshot();
            }
            task.done.set_value(lsn);
        }
        PublishSnapshot();
    }

    std::size_t Shard::ApproxCountUnsafe() const
    {
        return seed_.Count();
    }

    void Shard::TryBuildIndex(bool) {} // Deprecated logic
}