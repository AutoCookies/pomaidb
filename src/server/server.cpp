#include "pomai/server/server.h"
#include "pomai/server/protocol.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h> // TCP_NODELAY
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

namespace pomai::server
{

    static bool ReadExact(int fd, void *buf, std::size_t n)
    {
        auto *p = reinterpret_cast<std::uint8_t *>(buf);
        std::size_t got = 0;
        while (got < n)
        {
            ssize_t r = ::recv(fd, p + got, n - got, 0);
            if (r == 0)
                return false;
            if (r < 0)
            {
                if (errno == EINTR)
                    continue;
                return false;
            }
            got += (std::size_t)r;
        }
        return true;
    }

    static bool WriteExact(int fd, const void *buf, std::size_t n)
    {
        const auto *p = reinterpret_cast<const std::uint8_t *>(buf);
        std::size_t sent = 0;
        while (sent < n)
        {
            ssize_t w = ::send(fd, p + sent, n - sent, 0);
            if (w <= 0)
            {
                if (w < 0 && errno == EINTR)
                    continue;
                return false;
            }
            sent += (std::size_t)w;
        }
        return true;
    }

    bool PomaiServer::EnsureDir(const std::string &path)
    {
        struct stat st;
        if (stat(path.c_str(), &st) == 0)
            return S_ISDIR(st.st_mode);
        return mkdir(path.c_str(), 0755) == 0;
    }

    PomaiServer::PomaiServer(ServerConfig cfg, Logger *log) : cfg_(std::move(cfg)), log_(log) {}
    PomaiServer::~PomaiServer() { Stop(); }

    bool PomaiServer::Start()
    {
        if (running_.exchange(true))
            return true;

        if (!EnsureDir(cfg_.data_dir))
        {
            if (log_)
                log_->Error("Failed to create/open data_dir: " + cfg_.data_dir);
            running_ = false;
            return false;
        }

        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
        {
            if (log_)
                log_->Error("socket failed: " + std::string(std::strerror(errno)));
            running_ = false;
            return false;
        }

        int yes = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(cfg_.listen_port);
        if (inet_pton(AF_INET, cfg_.listen_host.c_str(), &addr.sin_addr) != 1)
        {
            if (log_)
                log_->Error("inet_pton failed for host: " + cfg_.listen_host);
            Stop();
            return false;
        }

        if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
        {
            if (log_)
                log_->Error("bind failed: " + std::string(std::strerror(errno)));
            Stop();
            return false;
        }

        if (listen(listen_fd_, 1) != 0)
        {
            if (log_)
                log_->Error("listen failed: " + std::string(std::strerror(errno)));
            Stop();
            return false;
        }

        if (log_)
        {
            log_->Info("PomaiServer listening on " + cfg_.listen_host + ":" + std::to_string(cfg_.listen_port));
        }

        AcceptOnce();
        return true;
    }

    void PomaiServer::Stop()
    {
        if (!running_.exchange(false))
            return;

        if (listen_fd_ >= 0)
        {
            ::close(listen_fd_);
            listen_fd_ = -1;
        }

        for (auto &kv : cols_)
        {
            if (kv.second.db)
                kv.second.db->Stop();
        }
        cols_.clear();

        if (log_)
            log_->Info("PomaiServer stopped");
    }

    void PomaiServer::AcceptOnce()
    {
        int c = ::accept(listen_fd_, nullptr, nullptr);
        if (c < 0)
        {
            if (log_)
                log_->Error("accept failed: " + std::string(std::strerror(errno)));
            return;
        }

        // Critical fix: disable Nagle on the accepted socket.
        int yes = 1;
        ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

        if (log_)
            log_->Info("Client connected (single-user)");

        ClientLoop(c);

        ::close(c);
        if (log_)
            log_->Info("Client disconnected");
    }

    bool PomaiServer::ReadFrame(int fd, std::vector<std::uint8_t> &payload)
    {
        std::uint8_t len_le[4];
        if (!ReadExact(fd, len_le, 4))
            return false;

        std::uint32_t len = LoadU32LE(len_le);
        if (len == 0 || len > 256u * 1024u * 1024u)
            return false;

        payload.resize(len);
        return ReadExact(fd, payload.data(), payload.size());
    }

    bool PomaiServer::WriteFrame(int fd, const std::vector<std::uint8_t> &payload)
    {
        std::uint8_t len_le[4];
        StoreU32LE(len_le, (std::uint32_t)payload.size());
        if (!WriteExact(fd, len_le, 4))
            return false;
        return WriteExact(fd, payload.data(), payload.size());
    }

    void PomaiServer::RespOk(std::vector<std::uint8_t> &out_payload)
    {
        out_payload.assign(1, (std::uint8_t)1);
    }

    void PomaiServer::RespErr(std::vector<std::uint8_t> &out_payload, const std::string &msg)
    {
        Buf b;
        PutU8(b, 0);
        PutString(b, msg);
        out_payload = std::move(b.data);
    }

    bool PomaiServer::HandlePayload(const std::vector<std::uint8_t> &payload, std::vector<std::uint8_t> &out_payload)
    {
        if (payload.empty())
        {
            RespErr(out_payload, "empty payload");
            return true;
        }

        Op op = (Op)payload[0];
        switch (op)
        {
        case Op::PING:
            return OpPing(out_payload);
        case Op::CREATE_COLLECTION:
            return OpCreateCollection(payload, out_payload);
        case Op::UPSERT_BATCH:
            return OpUpsertBatch(payload, out_payload);
        case Op::SEARCH:
            return OpSearch(payload, out_payload);
        default:
            RespErr(out_payload, "unknown op");
            return true;
        }
    }

    bool PomaiServer::OpPing(std::vector<std::uint8_t> &out_payload)
    {
        RespOk(out_payload);
        return true;
    }

    bool PomaiServer::OpCreateCollection(const std::vector<std::uint8_t> &payload,
                                         std::vector<std::uint8_t> &out_payload)
    {
        Reader rd{payload.data() + 1, payload.data() + payload.size()};

        std::string name;
        std::uint16_t dim = 0;
        std::uint8_t metric_u8 = 1;

        std::uint32_t shards = (std::uint32_t)cfg_.shards;
        std::uint32_t cap = (std::uint32_t)cfg_.shard_queue_capacity;

        if (!rd.ReadString(name) || !rd.ReadU16(dim) || !rd.ReadU8(metric_u8))
        {
            RespErr(out_payload, "bad create payload");
            return true;
        }

        // Optional overrides if present
        rd.ReadU32(shards);
        rd.ReadU32(cap);

        if (dim == 0)
            dim = (std::uint16_t)cfg_.default_dim;

        pomai::DbOptions opt;
        opt.dim = dim;
        opt.metric = (metric_u8 == 0) ? pomai::Metric::L2 : pomai::Metric::Cosine;
        opt.shards = shards ? (std::size_t)shards : (std::size_t)cfg_.shards;
        opt.shard_queue_capacity = cap ? (std::size_t)cap : (std::size_t)cfg_.shard_queue_capacity;

        // Important: put WAL under data_dir/<collection_name>
        // Keep it simple and deterministic.
        opt.wal_dir = cfg_.data_dir + "/" + name;
        (void)EnsureDir(opt.wal_dir);

        auto db = std::make_shared<pomai::PomaiDB>(opt);
        db->Start();

        std::uint32_t id = next_col_id_++;
        cols_[id] = Col{dim, opt.metric, db};

        Buf b;
        PutU8(b, 1);
        PutU32(b, id);
        out_payload = std::move(b.data);

        if (log_)
        {
            log_->Info("Created collection '" + name + "' id=" + std::to_string(id) +
                       " dim=" + std::to_string(dim) +
                       " shards=" + std::to_string(opt.shards) +
                       " wal_dir=" + opt.wal_dir);
        }
        return true;
    }

    bool PomaiServer::OpUpsertBatch(const std::vector<std::uint8_t> &payload,
                                    std::vector<std::uint8_t> &out_payload)
    {
        Reader rd{payload.data() + 1, payload.data() + payload.size()};

        std::uint32_t col_id = 0, n = 0;
        std::uint16_t dim = 0;
        if (!rd.ReadU32(col_id) || !rd.ReadU32(n) || !rd.ReadU16(dim))
        {
            RespErr(out_payload, "bad upsert_batch header");
            return true;
        }

        auto it = cols_.find(col_id);
        if (it == cols_.end())
        {
            RespErr(out_payload, "collection not found");
            return true;
        }
        if (it->second.dim != dim)
        {
            RespErr(out_payload, "dim mismatch");
            return true;
        }
        if (n == 0)
        {
            RespErr(out_payload, "empty batch");
            return true;
        }

        std::vector<std::uint64_t> ids(n);
        if (!rd.ReadBytes(ids.data(), (std::size_t)n * sizeof(std::uint64_t)))
        {
            RespErr(out_payload, "bad ids");
            return true;
        }

        const std::size_t total_f = (std::size_t)n * (std::size_t)dim;
        std::vector<float> all(total_f);
        if (!rd.ReadBytes(all.data(), total_f * sizeof(float)))
        {
            RespErr(out_payload, "bad vectors");
            return true;
        }

        // Still allocates per item. Next step is UpsertBatchRaw to remove this.
        std::vector<pomai::UpsertRequest> batch;
        batch.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i)
        {
            pomai::UpsertRequest r;
            r.id = (pomai::Id)ids[i];
            r.vec.data.assign(all.begin() + (std::size_t)i * dim,
                              all.begin() + (std::size_t)(i + 1) * dim);
            batch.push_back(std::move(r));
        }

        try
        {
            auto fut = it->second.db->UpsertBatch(std::move(batch), true);
            pomai::Lsn lsn = fut.get();

            Buf b;
            PutU8(b, 1);
            PutU64(b, (std::uint64_t)lsn);
            out_payload = std::move(b.data);
        }
        catch (const std::exception &e)
        {
            RespErr(out_payload, std::string("upsert_batch failed: ") + e.what());
        }
        return true;
    }

    bool PomaiServer::OpSearch(const std::vector<std::uint8_t> &payload,
                               std::vector<std::uint8_t> &out_payload)
    {
        Reader rd{payload.data() + 1, payload.data() + payload.size()};

        std::uint32_t col_id = 0, topk = 10;
        std::uint16_t dim = 0;

        if (!rd.ReadU32(col_id) || !rd.ReadU32(topk) || !rd.ReadU16(dim))
        {
            RespErr(out_payload, "bad search header");
            return true;
        }

        auto it = cols_.find(col_id);
        if (it == cols_.end())
        {
            RespErr(out_payload, "collection not found");
            return true;
        }
        if (it->second.dim != dim)
        {
            RespErr(out_payload, "dim mismatch");
            return true;
        }

        std::vector<float> q(dim);
        if (!rd.ReadBytes(q.data(), (std::size_t)dim * sizeof(float)))
        {
            RespErr(out_payload, "bad query");
            return true;
        }

        pomai::SearchRequest req;
        req.topk = topk;
        req.query.data = std::move(q);

        pomai::SearchResponse resp = it->second.db->Search(req);

        Buf b;
        PutU8(b, 1);
        PutU32(b, (std::uint32_t)resp.items.size());
        for (auto &item : resp.items)
            PutU64(b, (std::uint64_t)item.id);
        for (auto &item : resp.items)
            PutF32(b, item.score);

        out_payload = std::move(b.data);
        return true;
    }

    void PomaiServer::ClientLoop(int fd)
    {
        std::vector<std::uint8_t> payload, out_payload;
        while (running_)
        {
            if (!ReadFrame(fd, payload))
                break;
            out_payload.clear();
            HandlePayload(payload, out_payload);
            if (!WriteFrame(fd, out_payload))
                break;
        }
    }

} // namespace pomai::server
