#pragma once

// src/facade/server.h
// Clean, minimal server facade header for Pomai.
//
// NOTE: This file previously contained an accidental "contents =" token
// at the top which broke compilation by injecting invalid tokens into the
// preprocessor stream. That stray text has been removed.

#include <vector>
#include <thread>
#include <memory>
#include <atomic>
#include <future>
#include <string>
#include <cstdint>

#include "src/core/worker_queue.h"
#include "src/core/worker.h"
#include "src/core/pomai_db.h"
#include "src/core/map.h"
#include "src/core/config.h"
#include "src/ai/whispergrain.h"
#include "src/facade/sql_executor.h"
#include "src/facade/tcp_listener.h" // <- new

namespace pomai::server
{

    class PomaiServer
    {
    public:
        PomaiServer(pomai::core::PomaiMap *kv_map, pomai::core::PomaiDB *pomai_db, const pomai::config::PomaiConfig &config);
        ~PomaiServer();

        void stop();

        std::future<bool> dispatch_insert(const std::string &membr, uint64_t label, std::vector<float> vec);
        std::future<pomai::core::ShardSearchResult> dispatch_search(const std::string &membr, std::vector<float> query, size_t k, uint64_t label_hint);

        // Start/stop TCP listener (binds to config.net.port). Started automatically in constructor.
        bool start_tcp_listener();
        void stop_tcp_listener();

    private:
        void init_shard_workers();
        void start_workers();
        void stop_workers();
        void worker_thread_loop(size_t idx);

        pomai::core::PomaiMap *kv_map_;
        pomai::core::PomaiDB *pomai_db_;
        pomai::config::PomaiConfig config_;
        int port_;
        pomai::ai::WhisperGrain whisper_;

        // SQL executor used by TCP connections
        pomai::server::SqlExecutor sql_exec_;

        // TCP listener (owns acceptor thread + client threads)
        std::unique_ptr<pomai::server::net::TcpListener> tcp_listener_;

        std::vector<std::unique_ptr<pomai::core::WorkerQueue>> worker_queues_;
        std::vector<std::thread> worker_threads_;
        std::atomic<bool> workers_running_{false};
        std::atomic<uint64_t> next_req_id_{1};
    };

} // namespace pomai::server