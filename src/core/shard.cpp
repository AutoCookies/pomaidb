#include "shard.h"
#include <stdexcept>
#include <utility>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace pomai
{

    Shard::Shard(std::string name, std::size_t dim, std::size_t queue_cap, std::string wal_dir)
        : name_(std::move(name)),
          wal_dir_(std::move(wal_dir)),
          wal_(name_, wal_dir_, dim),
          seed_(dim),
          ingest_q_(queue_cap)
    {
        // NSW Vortex: M=48 (kết nối dày), ef=100 (build kỹ)
        auto new_index = std::make_shared<pomai::core::OrbitIndex>(dim, 48, 100);
        std::atomic_store(&index_, new_index);

        auto empty_snap = std::make_shared<SeedSnapshot>();
        empty_snap->dim = dim;
        std::atomic_store(&published_, empty_snap);
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

        wal_.ReplayToSeed(seed_);

        if (seed_.Count() > 0)
        {
            std::cout << "[" << name_ << "] Loading " << seed_.Count() << " vectors into Vortex..." << std::endl;
            auto idx = std::atomic_load(&index_);
            idx->Build(seed_.GetFlatData(), seed_.GetFlatIds());
        }

        wal_.Start();
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
        std::atomic_store_explicit(&published_, std::move(snap), std::memory_order_release);
    }

    SearchResponse Shard::Search(const SearchRequest &req, const pomai::ai::Budget &base_budget) const
    {
        auto idx = std::atomic_load_explicit(&index_, std::memory_order_acquire);
        if (idx)
            return idx->Search(req.query, base_budget);
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

            Lsn lsn = wal_.AppendUpserts(task.batch);
            seed_.ApplyUpserts(task.batch);

            // REAL-TIME VORTEX INSERT
            auto idx = std::atomic_load(&index_);
            std::vector<float> batch_data;
            std::vector<Id> batch_ids;
            batch_data.reserve(task.batch.size() * seed_.Dim());
            batch_ids.reserve(task.batch.size());

            for (const auto &req : task.batch)
            {
                batch_ids.push_back(req.id);
                batch_data.insert(batch_data.end(), req.vec.data.begin(), req.vec.data.end());
            }

            idx->InsertBatch(batch_data, batch_ids);

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
        auto idx = std::atomic_load_explicit(&index_, std::memory_order_acquire);
        return idx ? idx->TotalVectors() : 0;
    }
}