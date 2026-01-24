#include "src/facade/server.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <cstddef>

using namespace pomai::server;

PomaiServer::PomaiServer(pomai::core::PomaiMap *kv_map, pomai::core::PomaiDB *pomai_db, const pomai::config::PomaiConfig &config)
    : kv_map_(kv_map),
      pomai_db_(pomai_db),
      config_(config),
      port_(static_cast<int>(config.net.port)),
      whisper_(config.whisper)
{
    init_shard_workers();
    start_workers();
}

PomaiServer::~PomaiServer()
{
    stop();
}

void PomaiServer::stop()
{
    stop_workers();
}

void PomaiServer::init_shard_workers()
{
    size_t n = config_.orchestrator.shard_count > 0 ? static_cast<size_t>(config_.orchestrator.shard_count) : std::max<size_t>(1, std::thread::hardware_concurrency());
    worker_queues_.clear();
    worker_threads_.clear();
    worker_queues_.reserve(n);
    for (size_t i = 0; i < n; ++i)
        worker_queues_.emplace_back(std::make_unique<pomai::core::WorkerQueue>());
}

void PomaiServer::start_workers()
{
    bool expected = false;
    if (!workers_running_.compare_exchange_strong(expected, true))
        return;
    for (size_t i = 0; i < worker_queues_.size(); ++i)
    {
        worker_threads_.emplace_back([this, i]()
                                     { this->worker_thread_loop(i); });
    }
}

void PomaiServer::stop_workers()
{
    bool expected = true;
    if (!workers_running_.compare_exchange_strong(expected, false))
        return;
    for (size_t i = 0; i < worker_queues_.size(); ++i)
    {
        worker_queues_[i]->push(pomai::core::Work::make_stop());
    }
    for (auto &t : worker_threads_)
        if (t.joinable())
            t.join();
    worker_threads_.clear();
    worker_queues_.clear();
}

std::future<bool> PomaiServer::dispatch_insert(const std::string &membr, uint64_t label, std::vector<float> vec)
{
    size_t n = worker_queues_.size();
    if (n == 0)
    {
        std::promise<bool> p;
        p.set_value(false);
        return p.get_future();
    }
    size_t idx = pomai::core::shard_for_label(label, n);
    pomai::core::Work w = pomai::core::Work::make_insert(membr, label, std::move(vec), next_req_id_.fetch_add(1, std::memory_order_relaxed));
    std::future<bool> fut = w.prom_insert.get_future();
    worker_queues_[idx]->push(std::move(w));
    return fut;
}

std::future<pomai::core::ShardSearchResult> PomaiServer::dispatch_search(const std::string &membr, std::vector<float> query, size_t k, uint64_t label_hint)
{
    size_t n = worker_queues_.size();
    if (n == 0)
    {
        std::promise<pomai::core::ShardSearchResult> p;
        p.set_value(pomai::core::ShardSearchResult{});
        return p.get_future();
    }
    size_t idx = pomai::core::shard_for_label(label_hint, n);
    pomai::core::Work w = pomai::core::Work::make_search(membr, std::move(query), k, next_req_id_.fetch_add(1, std::memory_order_relaxed));
    std::future<pomai::core::ShardSearchResult> fut = w.prom_search.get_future();
    worker_queues_[idx]->push(std::move(w));
    return fut;
}

void PomaiServer::worker_thread_loop(size_t idx)
{
    pomai::core::WorkerQueue *q = worker_queues_[idx].get();
    while (workers_running_.load(std::memory_order_acquire))
    {
        pomai::core::Work w = q->pop_wait();
        if (w.kind == pomai::core::Work::Kind::STOP)
            break;
        if (w.kind == pomai::core::Work::Kind::INSERT)
        {
            bool ok = false;
            try
            {
                ok = pomai_db_->insert(w.membrance, w.vec.data(), w.label);
            }
            catch (...)
            {
                ok = false;
            }
            try
            {
                w.prom_insert.set_value(ok);
            }
            catch (...)
            {
            }
        }
        else if (w.kind == pomai::core::Work::Kind::SEARCH)
        {
            pomai::core::ShardSearchResult res;
            try
            {
                res.hits = pomai_db_->search(w.membrance, w.vec.data(), w.k);
            }
            catch (...)
            {
                res.hits.clear();
            }
            try
            {
                w.prom_search.set_value(std::move(res));
            }
            catch (...)
            {
            }
        }
    }
    while (true)
    {
        auto opt = q->try_pop();
        if (!opt.has_value())
            break;
        pomai::core::Work w = std::move(*opt);
        if (w.kind == pomai::core::Work::Kind::INSERT)
        {
            bool ok = false;
            try
            {
                ok = pomai_db_->insert(w.membrance, w.vec.data(), w.label);
            }
            catch (...)
            {
                ok = false;
            }
            try
            {
                w.prom_insert.set_value(ok);
            }
            catch (...)
            {
            }
        }
        else if (w.kind == pomai::core::Work::Kind::SEARCH)
        {
            pomai::core::ShardSearchResult res;
            try
            {
                res.hits = pomai_db_->search(w.membrance, w.vec.data(), w.k);
            }
            catch (...)
            {
                res.hits.clear();
            }
            try
            {
                w.prom_search.set_value(std::move(res));
            }
            catch (...)
            {
            }
        }
    }
}