#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <vector>

#include <pomai/api/pomai_db.h>
#include <pomai/server/config.h>
#include <pomai/util/logger.h>

namespace pomai::server
{

    class PomaiServer
    {
    public:
        PomaiServer(ServerConfig cfg, pomai::Logger *log);
        ~PomaiServer();

        bool Start();
        void Stop();

    private:
        // Vòng lặp chấp nhận kết nối TCP
        void AcceptLoop();

        // Vòng lặp chấp nhận kết nối Unix Domain Socket (Localhost High Perf)
        void AcceptLoopIPC();

        // Xử lý phiên làm việc của Client (dùng chung cho cả TCP và UDS)
        void ClientSession(int fd);
        void JoinClientThreads();

        static bool ReadFrame(int fd, std::vector<std::uint8_t> &payload);
        static bool WriteFrame(int fd, const std::vector<std::uint8_t> &payload);

        bool HandlePayload(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload);

        void RespErr(std::vector<std::uint8_t> &out_payload, const std::string &msg);
        void RespOk(std::vector<std::uint8_t> &out_payload);

        bool OpPing(std::vector<std::uint8_t> &out_payload);
        bool OpCreateCollection(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload);
        bool OpUpsertBatch(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload);
        bool OpSearch(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload);

        static bool EnsureDir(const std::string &path);

        // Catalog Management
        void LoadCatalog();
        void SaveCollectionMeta(const std::string &name, const DbOptions &opt);

        ServerConfig cfg_;
        pomai::Logger *log_{nullptr};
        std::atomic<bool> running_{false};

        // TCP Socket
        int listen_fd_{-1};

        // Unix Domain Socket (IPC)
        int ipc_fd_{-1};
        std::thread ipc_thread_;

        // Connection Management
        std::atomic<int32_t> active_connections_{0};
        const int32_t MAX_CONNECTIONS = 100;
        std::mutex client_mu_;
        std::vector<std::thread> client_threads_;

        // Collection Management
        std::mutex cols_mu_;
        std::uint32_t next_col_id_{1};
        struct Col
        {
            std::string name;
            std::uint16_t dim{0};
            pomai::Metric metric{pomai::Metric::Cosine};
            std::shared_ptr<pomai::PomaiDB> db;
        };
        std::unordered_map<std::uint32_t, Col> cols_;
        std::unordered_map<std::string, std::uint32_t> name_to_id_;
    };

} // namespace pomai::server
