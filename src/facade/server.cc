#include "src/facade/server.h"
#include "src/core/cpu_kernels.h"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <cstring>
#include <iostream>

namespace pomai::server
{
    enum class OpCode : uint8_t
    {
        INSERT = 0x01,
        SEARCH = 0x02,
        INFO = 0x03,
        INSERT_BATCH = 0x04,
        PING = 0xFF
    };

    PomaiServer::PomaiServer(pomai::core::PomaiMap *kv_map,
                             pomai::core::PomaiDB *pomai_db,
                             const pomai::config::PomaiConfig &config)
        : kv_map_(kv_map),
          pomai_db_(pomai_db),
          config_(config),
          port_(static_cast<int>(config.net.port)),
          whisper_(config.whisper)
    {
        init_shard_workers();
        start_workers();

        // [CRITICAL FIX] Nếu TCP không bind được port, ném Exception sập server ngay.
        // Không để tình trạng "xác sống" (Zombie) chỉ chạy UDP.
        if (!start_tcp_listener())
        {
            // Ném exception để dừng server ngay lập tức và in ra lỗi
            throw std::runtime_error("PomaiServer FATAL: Could not bind TCP port " + std::to_string(port_));
        }
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
        size_t n = config_.orchestrator.shard_count;
        if (n == 0)
            n = std::max<size_t>(1, std::thread::hardware_concurrency());

        worker_queues_.clear();
        worker_threads_.clear();
        worker_queues_.reserve(n);

        for (size_t i = 0; i < n; ++i)
        {
            worker_queues_.emplace_back(std::make_unique<pomai::core::WorkerQueue>());
        }
    }

    void PomaiServer::start_workers()
    {
        size_t n = worker_queues_.size();
        worker_threads_.reserve(n);
        for (size_t i = 0; i < n; ++i)
        {
            worker_threads_.emplace_back(&PomaiServer::worker_thread_loop, this, i);
        }
    }

    void PomaiServer::stop_workers()
    {
        for (auto &q : worker_queues_)
        {
            if (q)
                q->push(pomai::core::Work::make_stop());
        }

        for (auto &t : worker_threads_)
        {
            if (t.joinable())
                t.join();
        }
        worker_threads_.clear();
    }

    std::future<bool> PomaiServer::dispatch_insert(const std::string &membr, uint64_t label, std::vector<float> vec)
    {
        std::vector<std::pair<uint64_t, std::vector<float>>> batch;
        batch.emplace_back(label, std::move(vec));

        bool posted = pomai_db_->insert_batch(membr, batch);

        std::promise<bool> p;
        p.set_value(posted);
        return p.get_future();
    }

    std::future<pomai::core::ShardSearchResult> PomaiServer::dispatch_search(const std::string &membr, std::vector<float> query, size_t k)
    {
        auto work = pomai::core::Work::make_search(membr, std::move(query), k);
        std::future<pomai::core::ShardSearchResult> fut = work.prom_search.get_future();

        static std::atomic<size_t> rr{0};
        size_t shard_idx = rr.fetch_add(1, std::memory_order_relaxed) % worker_queues_.size();

        worker_queues_[shard_idx]->push(std::move(work));
        return fut;
    }

    void PomaiServer::worker_thread_loop(size_t idx)
    {
        auto &queue = worker_queues_[idx];
        while (true)
        {
            auto w = queue->pop_wait();
            if (w.kind == pomai::core::Work::Kind::STOP)
                break;

            try
            {
                if (w.kind == pomai::core::Work::Kind::SEARCH)
                {
                    auto *m = pomai_db_->get_membrance(w.membrance);
                    pomai::core::ShardSearchResult res;
                    if (m && m->orbit)
                    {
                        res.hits = m->orbit->search(w.vec.data(), w.k, 32);
                    }
                    w.prom_search.set_value(std::move(res));
                }
            }
            catch (...)
            {
                if (w.kind == pomai::core::Work::Kind::SEARCH)
                {
                    try
                    {
                        w.prom_search.set_value({});
                    }
                    catch (...)
                    {
                    }
                }
            }
        }
    }

    std::pair<size_t, std::string> PomaiServer::handle_binary_command(std::string_view data)
    {
        if (data.size() < 1)
            return {0, ""};

        OpCode op = static_cast<OpCode>(data[0]);

        switch (op)
        {
        case OpCode::INSERT:
        {
            if (data.size() < 13)
                return {0, ""};

            uint64_t label;
            uint32_t dim;
            std::memcpy(&label, data.data() + 1, 8);
            std::memcpy(&dim, data.data() + 9, 4);

            size_t vec_bytes = dim * sizeof(float);
            size_t total_len = 13 + vec_bytes;
            if (data.size() < total_len)
                return {0, ""};

            std::vector<float> vec(dim);
            std::memcpy(vec.data(), data.data() + 13, vec_bytes);

            bool ok = pomai_db_->insert_batch("default", {{label, std::move(vec)}});
            return {total_len, ok ? "\x01" : "\x00"};
        }

        case OpCode::INSERT_BATCH:
        {
            if (data.size() < 9)
                return {0, ""};

            uint32_t count, dim;
            std::memcpy(&count, data.data() + 1, 4);
            std::memcpy(&dim, data.data() + 5, 4);

            size_t vec_bytes = dim * sizeof(float);
            size_t item_size = 8 + vec_bytes;
            size_t total_len = 9 + (size_t)count * item_size;

            if (data.size() < total_len)
                return {0, ""};

            const char *ptr = data.data() + 9;
            std::vector<std::pair<uint64_t, std::vector<float>>> batch;
            try
            {
                batch.reserve(count);
            }
            catch (const std::bad_alloc &)
            {
                return {total_len, "\x00"};
            }

            for (uint32_t i = 0; i < count; ++i)
            {
                uint64_t label;
                std::memcpy(&label, ptr, 8);
                std::vector<float> vec(dim);
                std::memcpy(vec.data(), ptr + 8, vec_bytes);
                batch.emplace_back(label, std::move(vec));
                ptr += item_size;
            }

            bool ok = pomai_db_->insert_batch("default", batch);
            return {total_len, ok ? "\x01" : "\x00"};
        }

        case OpCode::SEARCH:
        {
            if (data.size() < 9)
                return {0, ""};

            uint32_t k, dim;
            std::memcpy(&k, data.data() + 1, 4);
            std::memcpy(&dim, data.data() + 5, 4);

            size_t vec_bytes = dim * sizeof(float);
            size_t total_len = 9 + vec_bytes;

            if (data.size() < total_len)
                return {0, ""};

            std::vector<float> query(dim);
            std::memcpy(query.data(), data.data() + 9, vec_bytes);

            auto fut = dispatch_search("default", std::move(query), k);
            auto result = fut.get();

            std::string resp;
            try
            {
                uint32_t r_count = static_cast<uint32_t>(result.hits.size());
                resp.resize(4 + r_count * 12);
                char *out = resp.data();
                std::memcpy(out, &r_count, 4);
                out += 4;
                for (const auto &hit : result.hits)
                {
                    std::memcpy(out, &hit.first, 8);
                    float d = hit.second;
                    std::memcpy(out + 8, &d, 4);
                    out += 12;
                }
            }
            catch (...)
            {
                return {total_len, "\x00"};
            }
            return {total_len, resp};
        }

        case OpCode::PING:
            return {1, "PONG"};

        default:
            return {1, "\x00"};
        }
    }

    bool PomaiServer::start_tcp_listener()
    {
        if (tcp_listener_ && tcp_listener_->running())
            return true;

        auto handler = [this](std::string_view data) -> std::pair<size_t, std::string>
        {
            try
            {
                return handle_binary_command(data);
            }
            catch (...)
            {
                return {data.size(), "\x00"};
            }
        };

        tcp_listener_ = std::make_unique<pomai::server::net::TcpListener>(static_cast<uint16_t>(port_), std::move(handler));

        if (!tcp_listener_->start())
        {
            tcp_listener_.reset();
            std::cerr << "[PomaiServer] ERROR: Failed to bind TCP port " << port_ << "\n";
            return false;
        }
        std::clog << "[PomaiServer] Binary Listener active on port " << port_ << "\n";
        return true;
    }

    void PomaiServer::stop_tcp_listener()
    {
        if (tcp_listener_)
        {
            tcp_listener_->stop();
            tcp_listener_.reset();
        }
    }
}