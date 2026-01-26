#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "pomai/pomai_db.h"
#include "pomai/server/config.h"
#include "pomai/server/logger.h"

namespace pomai::server
{

    class PomaiServer
    {
    public:
        PomaiServer(ServerConfig cfg, Logger *log);
        ~PomaiServer();

        bool Start(); // blocks in single-user mode
        void Stop();

    private:
        void AcceptOnce();
        void ClientLoop(int fd);

        bool ReadFrame(int fd, std::vector<std::uint8_t> &payload);
        bool WriteFrame(int fd, const std::vector<std::uint8_t> &payload);

        bool HandlePayload(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload);

        // ops
        void RespErr(std::vector<std::uint8_t> &out_payload, const std::string &msg);
        void RespOk(std::vector<std::uint8_t> &out_payload);

        bool OpPing(std::vector<std::uint8_t> &out_payload);
        bool OpCreateCollection(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload);
        bool OpUpsertBatch(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload);
        bool OpSearch(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload);

        static bool EnsureDir(const std::string &path);

        ServerConfig cfg_;
        Logger *log_{nullptr};
        std::atomic<bool> running_{false};

        int listen_fd_{-1};

        std::uint32_t next_col_id_{1};
        struct Col
        {
            std::uint16_t dim{0};
            pomai::Metric metric{pomai::Metric::Cosine};
            std::shared_ptr<pomai::PomaiDB> db;
        };
        std::unordered_map<std::uint32_t, Col> cols_;
    };

} // namespace pomai::server
