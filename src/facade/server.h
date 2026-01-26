#pragma once

#include <vector>
#include <thread>
#include <memory>
#include <atomic>
#include <future>
#include <string>
#include <string_view>
#include <cstdint>
#include <utility>

#include "src/core/worker_queue.h"
#include "src/core/worker.h"
#include "src/core/pomai_db.h"
#include "src/core/map.h"
#include "src/core/config.h"
#include "src/ai/whispergrain.h"
#include "src/facade/tcp_listener.h"

namespace pomai::server
{
    class PomaiServer
    {
    public:
        PomaiServer(pomai::core::PomaiMap *kv_map,
                    pomai::core::PomaiDB *pomai_db,
                    const pomai::config::PomaiConfig &config);
        ~PomaiServer();

        void stop();

        std::future<bool> dispatch_insert(const std::string &membr, uint64_t label, std::vector<float> vec);
        std::future<pomai::core::ShardSearchResult> dispatch_search(const std::string &membr, std::vector<float> query, size_t k);

        bool start_tcp_listener();
        void stop_tcp_listener();

    private:
        void init_shard_workers();
        void start_workers();
        void stop_workers();
        void worker_thread_loop(size_t idx);

        std::pair<size_t, std::string> handle_binary_command(std::string_view payload);

        pomai::core::PomaiMap *kv_map_;
        pomai::core::PomaiDB *pomai_db_;
        pomai::config::PomaiConfig config_;
        int port_;
        pomai::ai::WhisperGrain whisper_;

        std::unique_ptr<pomai::server::net::TcpListener> tcp_listener_;
        std::vector<std::unique_ptr<pomai::core::WorkerQueue>> worker_queues_;
        std::vector<std::thread> worker_threads_;
    };
}