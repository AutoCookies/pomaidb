#include "src/facade/server.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <cstddef>
#include <chrono>

using namespace pomai::server;

PomaiServer::PomaiServer(pomai::core::PomaiMap *kv_map, pomai::core::PomaiDB *pomai_db, const pomai::config::PomaiConfig &config)
    : kv_map_(kv_map),
      pomai_db_(pomai_db),
      config_(config),
      port_(static_cast<int>(config.net.port)),
      whisper_(config.whisper),
      sql_exec_()
{
    init_shard_workers();
    start_workers();

    // Start TCP listener to serve clients (required by Architect)
    start_tcp_listener();
}

PomaiServer::~PomaiServer()
{
    stop();
    stop_tcp_listener();
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

    // Acquire request id explicitly so we can log it consistently
    uint64_t req_id = next_req_id_.fetch_add(1, std::memory_order_relaxed);
    pomai::core::Work w = pomai::core::Work::make_insert(membr, label, std::move(vec), req_id);
    std::future<bool> fut = w.prom_insert.get_future();

    // Debug log: enqueue
    std::clog << "[Dispatch] INSERT req=" << req_id << " membr=" << membr << " label=" << label << " -> shard=" << idx << "\n";

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

    // Acquire request id explicitly so we can log it consistently
    uint64_t req_id = next_req_id_.fetch_add(1, std::memory_order_relaxed);
    pomai::core::Work w = pomai::core::Work::make_search(membr, std::move(query), k, req_id);
    std::future<pomai::core::ShardSearchResult> fut = w.prom_search.get_future();

    std::clog << "[Dispatch] SEARCH req=" << req_id << " membr=" << membr << " k=" << k << " -> shard=" << idx << "\n";

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

        auto work_start = std::chrono::steady_clock::now();
        if (w.kind == pomai::core::Work::Kind::INSERT)
        {
            std::clog << "[Worker " << idx << "] START INSERT req=" << w.req_id << " membr=" << w.membrance << " label=" << w.label << "\n";
            bool ok = false;
            try
            {
                ok = pomai_db_->insert(w.membrance, w.vec.data(), w.label);
            }
            catch (...)
            {
                ok = false;
            }
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - work_start).count();
            std::clog << "[Worker " << idx << "] DONE  INSERT req=" << w.req_id << " ok=" << (ok ? "1" : "0") << " dur_ms=" << dur << "\n";
            if (dur > 50)
                std::clog << "[Worker " << idx << "] WARNING: slow insert req=" << w.req_id << " dur_ms=" << dur << "\n";
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
            std::clog << "[Worker " << idx << "] START SEARCH req=" << w.req_id << " membr=" << w.membrance << " k=" << w.k << "\n";
            pomai::core::ShardSearchResult res;
            try
            {
                res.hits = pomai_db_->search(w.membrance, w.vec.data(), w.k);
            }
            catch (...)
            {
                res.hits.clear();
            }
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - work_start).count();
            std::clog << "[Worker " << idx << "] DONE  SEARCH req=" << w.req_id << " hits=" << res.hits.size() << " dur_ms=" << dur << "\n";
            if (dur > 50)
                std::clog << "[Worker " << idx << "] WARNING: slow search req=" << w.req_id << " dur_ms=" << dur << "\n";
            try
            {
                w.prom_search.set_value(std::move(res));
            }
            catch (...)
            {
            }
        }
    }

    // Drain remaining
    while (true)
    {
        auto opt = q->try_pop();
        if (!opt.has_value())
            break;
        pomai::core::Work w = std::move(*opt);
        auto work_start = std::chrono::steady_clock::now();

        if (w.kind == pomai::core::Work::Kind::INSERT)
        {
            std::clog << "[Worker " << idx << "] DRAIN INSERT req=" << w.req_id << " membr=" << w.membrance << " label=" << w.label << "\n";
            bool ok = false;
            try
            {
                ok = pomai_db_->insert(w.membrance, w.vec.data(), w.label);
            }
            catch (...)
            {
                ok = false;
            }
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - work_start).count();
            std::clog << "[Worker " << idx << "] DRAIN DONE INSERT req=" << w.req_id << " ok=" << (ok ? "1" : "0") << " dur_ms=" << dur << "\n";
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
            std::clog << "[Worker " << idx << "] DRAIN SEARCH req=" << w.req_id << " membr=" << w.membrance << " k=" << w.k << "\n";
            pomai::core::ShardSearchResult res;
            try
            {
                res.hits = pomai_db_->search(w.membrance, w.vec.data(), w.k);
            }
            catch (...)
            {
                res.hits.clear();
            }
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - work_start).count();
            std::clog << "[Worker " << idx << "] DRAIN DONE SEARCH req=" << w.req_id << " hits=" << res.hits.size() << " dur_ms=" << dur << "\n";
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

bool PomaiServer::start_tcp_listener()
{
    if (tcp_listener_ && tcp_listener_->running())
        return true;

    // Handler: map incoming text commands to SqlExecutor
    auto handler = [this](const std::string &cmd) -> std::string
    {
        ClientState state;
        // route to SqlExecutor which returns an ASCII response
        try
        {
            return sql_exec_.execute(pomai_db_, whisper_, state, cmd);
        }
        catch (const std::exception &e)
        {
            return std::string("ERR: ") + e.what() + "\n";
        }
        catch (...)
        {
            return std::string("ERR: unknown\n");
        }
    };

    tcp_listener_ = std::make_unique<pomai::server::net::TcpListener>(static_cast<uint16_t>(port_), std::move(handler));
    if (!tcp_listener_->start())
    {
        tcp_listener_.reset();
        std::cerr << "[PomaiServer] TCP listener failed to start on port " << port_ << "\n";
        return false;
    }
    std::clog << "[PomaiServer] TCP listener started on port " << port_ << "\n";
    return true;
}

void PomaiServer::stop_tcp_listener()
{
    if (tcp_listener_)
    {
        tcp_listener_->stop();
        tcp_listener_.reset();
        std::clog << "[PomaiServer] TCP listener stopped\n";
    }
}