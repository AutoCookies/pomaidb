// Full server.cpp with DbOptions.allow_sync_on_append passed into created DbOptions
#include <pomai/server/server.h>
#include <pomai/server/protocol.h>
#include <pomai/core/types.h>
#include <pomai/core/posix_compat.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>

#include <cerrno>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace pomai::server
{

    // --- Static Helpers ---
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
            // MSG_NOSIGNAL để tránh crash process khi client ngắt kết nối
            ssize_t w = ::send(fd, p + sent, n - sent, MSG_NOSIGNAL);
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

    // Read exactly n bytes from a file descriptor (regular file). Returns false on EOF/error.
    static bool ReadExactFd(int fd, void *buf, std::size_t n)
    {
        auto *p = reinterpret_cast<std::uint8_t *>(buf);
        std::size_t got = 0;
        while (got < n)
        {
            ssize_t r = ::read(fd, p + got, n - got);
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

    // Try to infer vector dimension by reading the first WAL record header.
    // WAL payload layout begins with: LSN (u64) | count (u32) | dim (u16)
    // Returns 0 on failure or inferred dim (>0) on success.
    static uint16_t InferDimFromWal(const std::string &wal_path)
    {
        int fd = ::open(wal_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            return 0;

        uint8_t header[sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t)];
        bool ok = ReadExactFd(fd, header, sizeof(header));
        ::close(fd);
        if (!ok)
            return 0;

        const uint8_t *p = header;
        uint64_t lsn_le;
        uint32_t count_le;
        uint16_t dim_le;
        std::memcpy(&lsn_le, p, sizeof(lsn_le));
        p += sizeof(lsn_le);
        std::memcpy(&count_le, p, sizeof(count_le));
        p += sizeof(count_le);
        std::memcpy(&dim_le, p, sizeof(dim_le));

#if __BYTE_ORDER == __BIG_ENDIAN
        dim_le = __builtin_bswap16(dim_le);
#endif

        // Basic sanity check
        if (dim_le == 0 || dim_le > 10'000)
            return 0;

        return dim_le;
    }

    // --- Server Implementation ---

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
                log_->Error("server.start", "Failed to create/open data_dir: " + cfg_.data_dir);
            running_ = false;
            return false;
        }

        // 1. Recover Data from Disk
        LoadCatalog();

        // 2. Setup Unix Domain Socket (IPC) - High Performance Local
        if (!cfg_.unix_socket.empty())
        {
            // Xóa socket file cũ nếu còn sót lại
            ::unlink(cfg_.unix_socket.c_str());

            ipc_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
            if (ipc_fd_ < 0)
            {
                if (log_)
                    log_->Error("server.ipc", "IPC socket creation failed: " + std::string(std::strerror(errno)));
            }
            else
            {
                sockaddr_un addr_un{};
                addr_un.sun_family = AF_UNIX;
                // Copy path an toàn
                std::strncpy(addr_un.sun_path, cfg_.unix_socket.c_str(), sizeof(addr_un.sun_path) - 1);

                if (::bind(ipc_fd_, (sockaddr *)&addr_un, sizeof(addr_un)) != 0)
                {
                    if (log_)
                        log_->Error("server.ipc", "IPC bind failed: " + cfg_.unix_socket + " (" + std::strerror(errno) + ")");
                    ::close(ipc_fd_);
                    ipc_fd_ = -1;
                }
                else
                {
                    // Set permission rộng rãi để dễ dev (777)
                    // Trong production nên dùng 660 và group ownership
                    ::chmod(cfg_.unix_socket.c_str(), 0777);

                    if (::listen(ipc_fd_, 128) != 0)
                    {
                        if (log_)
                            log_->Error("server.ipc", "IPC listen failed");
                        ::close(ipc_fd_);
                        ipc_fd_ = -1;
                    }
                    else
                    {
                        if (log_)
                            log_->Info("server.ipc", "IPC Listening on " + cfg_.unix_socket);
                        // Chạy AcceptLoopIPC trên thread riêng
                        ipc_thread_ = std::thread([this]()
                                                  { this->AcceptLoopIPC(); });
                    }
                }
            }
        }

        // 3. Setup TCP Socket - Remote Access
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
        {
            if (log_)
                log_->Error("server.tcp", "TCP socket failed: " + std::string(std::strerror(errno)));
            // Nếu TCP lỗi nhưng IPC chạy thì vẫn OK? Thôi cứ coi như lỗi nghiêm trọng.
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
                log_->Error("server.tcp", "inet_pton failed for host: " + cfg_.listen_host);
            Stop();
            return false;
        }

        if (bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
        {
            if (log_)
                log_->Error("server.tcp", "bind failed: " + std::string(std::strerror(errno)));
            Stop();
            return false;
        }

        if (listen(listen_fd_, 128) != 0)
        {
            if (log_)
                log_->Error("server.tcp", "listen failed: " + std::string(std::strerror(errno)));
            Stop();
            return false;
        }

        if (log_)
            log_->Info("server.tcp", "TCP Listening on " + cfg_.listen_host + ":" + std::to_string(cfg_.listen_port));

        // Chạy AcceptLoop TCP trên thread hiện tại (blocking main thread)
        AcceptLoop();
        return true;
    }

    void PomaiServer::Stop()
    {
        if (!running_.exchange(false))
            return;

        // Đóng TCP Socket
        if (listen_fd_ >= 0)
        {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }

        // Đóng IPC Socket
        if (ipc_fd_ >= 0)
        {
            ::shutdown(ipc_fd_, SHUT_RDWR);
            ::close(ipc_fd_);
            ipc_fd_ = -1;
            // Xóa file socket khi tắt server
            if (!cfg_.unix_socket.empty())
            {
                ::unlink(cfg_.unix_socket.c_str());
            }
        }

        if (ipc_thread_.joinable())
            ipc_thread_.join();
        JoinClientThreads();

        // Chờ Client thoát
        int retries = 50;
        while (active_connections_ > 0 && retries-- > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        // Dọn dẹp DB Resources
        {
            std::lock_guard<std::mutex> lk(cols_mu_);
            for (auto &kv : cols_)
            {
                if (kv.second.db)
                { // Thêm ngoặc nhọn ở đây
                    auto st = kv.second.db->Stop();
                    if (!st.ok() && log_)
                        log_->Warn("server.stop", "Stop failed for collection '" + kv.second.name + "': " + st.msg);
                } // Đóng ngoặc nhọn ở đây
            }
            cols_.clear();
            name_to_id_.clear();
        }

        if (log_)
            log_->Info("server.stop", "PomaiServer stopped");
    }

    void PomaiServer::LoadCatalog()
    {
        if (log_)
            log_->Info("server.catalog", "Scanning catalog in: " + cfg_.data_dir);

        for (const auto &entry : fs::directory_iterator(cfg_.data_dir))
        {
            if (!entry.is_directory())
                continue;

            std::string name = entry.path().filename().string();
            std::string meta_path = entry.path().string() + "/meta.bin";

            // If meta.bin exists, we parse it. If not, try to auto-detect WAL files and synthesize meta.
            if (!fs::exists(meta_path))
            {
                // Look for shard-*.wal files in this directory
                bool found_any_wal = false;
                uint16_t inferred_dim = 0;
                std::string wal_dir = entry.path().string() + "/wal";
                for (std::size_t i = 0; i < cfg_.shards; ++i)
                {
                    std::string wp = wal_dir + "/shard-" + std::to_string(i) + ".wal";
                    if (fs::exists(wp))
                    {
                        found_any_wal = true;
                        inferred_dim = InferDimFromWal(wp);
                        if (inferred_dim != 0)
                            break; // good enough
                    }
                }

                if (!found_any_wal)
                {
                    // No meta and no wal files -> skip
                    if (log_)
                        log_->Debug("server.catalog", "Skipping directory (no meta.bin, no wal files): " + entry.path().string());
                    continue;
                }

                // Found WAL files. If we couldn't infer dim, fall back to default server config.
                if (inferred_dim == 0)
                    inferred_dim = static_cast<uint16_t>(cfg_.default_dim);

                // Synthesize DbOptions and persist meta.bin for future runs.
                pomai::DbOptions opt;
                opt.dim = inferred_dim;
                opt.metric = pomai::Metric::L2; // conservative default; operator can change later
                opt.shards = cfg_.shards;
                opt.shard_queue_capacity = cfg_.shard_queue_capacity;
                opt.wal_dir = entry.path().string();

                // IMPORTANT: carry over global server-level allow_sync_on_append into collection options
                opt.allow_sync_on_append = cfg_.allow_sync_on_append;

                // Persist meta so we don't have to re-infer next time
                SaveCollectionMeta(name, opt);
                if (log_)
                    log_->Info("server.catalog", "Synthesized meta.bin for collection '" + name +
                                                     "' (inferred dim=" + std::to_string(opt.dim) +
                                                     ", shards=" + std::to_string(opt.shards) + ")");
            }

            // Re-check presence of meta.bin after possible synthesis
            if (!fs::exists(meta_path))
            {
                // still not present -> skip
                continue;
            }

            std::ifstream in(meta_path, std::ios::binary);
            if (!in)
                continue;

            uint16_t dim;
            uint8_t metric_u8;
            uint32_t shards;
            uint32_t cap;

            in.read((char *)&dim, 2);
            in.read((char *)&metric_u8, 1);
            in.read((char *)&shards, 4);
            in.read((char *)&cap, 4);

            if (!in)
            {
                if (log_)
                    log_->Error("server.catalog", "Corrupt meta for collection: " + name);
                continue;
            }

            pomai::DbOptions opt;
            opt.dim = dim;
            opt.metric = (metric_u8 == 0) ? pomai::Metric::L2 : pomai::Metric::Cosine;
            opt.shards = shards;
            opt.shard_queue_capacity = cap;
            opt.wal_dir = entry.path().string();

            // Carry global server-level policy into the DbOptions as a default unless the meta specifies otherwise.
            opt.allow_sync_on_append = cfg_.allow_sync_on_append;

            if (log_)
                log_->Info("server.catalog", "Recovering collection: " + name + " (dim=" + std::to_string(dim) + ")");

            auto db = std::make_shared<pomai::PomaiDB>(opt, log_);
            auto st = db->Start();
            if (!st.ok() && log_)
                log_->Error("server.catalog", "Failed to start collection '" + name + "': " + st.msg);

            std::lock_guard<std::mutex> lk(cols_mu_);
            std::uint32_t id = next_col_id_++;
            cols_[id] = Col{name, dim, opt.metric, db};
            name_to_id_[name] = id;
        }
    }

    void PomaiServer::SaveCollectionMeta(const std::string &name, const DbOptions &opt)
    {
        std::string path = opt.wal_dir + "/meta.bin";
        std::ofstream out(path, std::ios::binary);
        if (!out)
            return;

        uint16_t dim = opt.dim;
        uint8_t metric_u8 = (opt.metric == pomai::Metric::L2) ? 0 : 1;
        uint32_t shards = (uint32_t)opt.shards;
        uint32_t cap = (uint32_t)opt.shard_queue_capacity;

        out.write((const char *)&dim, 2);
        out.write((const char *)&metric_u8, 1);
        out.write((const char *)&shards, 4);
        out.write((const char *)&cap, 4);
        out.flush();
    }

    // --- TCP Accept Loop ---
    void PomaiServer::AcceptLoop()
    {
        while (running_)
        {
            sockaddr_in client_addr{};
            socklen_t len = sizeof(client_addr);
            int client_fd = ::accept(listen_fd_, (sockaddr *)&client_addr, &len);

            if (client_fd < 0)
            {
                if (errno == EINTR)
                    continue;
                if (!running_)
                    break;

                // Đưa ra ngoài hoặc dùng ngoặc nhọn nếu muốn nó chạy cùng điều kiện
                if (log_)
                    log_->Error("server.tcp", "TCP accept failed: " + std::string(std::strerror(errno)));

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (active_connections_ >= MAX_CONNECTIONS)
            {
                if (log_)
                    log_->Warn("server.tcp", "Max connections reached. Rejecting TCP client.");
                ::close(client_fd);
                continue;
            }

            // TCP Optimize
            int yes = 1;
            ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

            active_connections_++;
            std::thread t([this, client_fd]()
                          { this->ClientSession(client_fd); });
            {
                std::lock_guard<std::mutex> lk(client_mu_);
                client_threads_.push_back(std::move(t));
            }
        }
    }

    // --- IPC (Unix Domain Socket) Accept Loop ---
    void PomaiServer::AcceptLoopIPC()
    {
        while (running_)
        {
            int client_fd = ::accept(ipc_fd_, nullptr, nullptr);
            if (client_fd < 0)
            {
                if (errno == EINTR)
                    continue;
                if (!running_)
                    break;
                // Không log lỗi quá nhiều ở loop phụ
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (active_connections_ >= MAX_CONNECTIONS)
            {
                // Backpressure
                ::close(client_fd);
                continue;
            }

            active_connections_++;
            // Spawn thread xử lý y hệt như TCP, chỉ khác là không cần set TCP_NODELAY
            std::thread t([this, client_fd]()
                          { this->ClientSession(client_fd); });
            {
                std::lock_guard<std::mutex> lk(client_mu_);
                client_threads_.push_back(std::move(t));
            }
        }
    }

    void PomaiServer::JoinClientThreads()
    {
        std::vector<std::thread> threads;
        {
            std::lock_guard<std::mutex> lk(client_mu_);
            threads.swap(client_threads_);
        }
        for (auto &t : threads)
        {
            if (t.joinable())
                t.join();
        }
    }

    void PomaiServer::ClientSession(int fd)
    {
        struct Guard
        {
            PomaiServer *s;
            int fd;
            ~Guard()
            {
                ::close(fd);
                s->active_connections_--;
            }
        } guard{this, fd};

        if (log_)
            log_->Debug("server.client", "Client connected. Active: " + std::to_string(active_connections_));

        std::vector<std::uint8_t> payload, out_payload;
        while (running_)
        {
            if (!ReadFrame(fd, payload))
                break;

            out_payload.clear();
            try
            {
                HandlePayload(payload, out_payload);
            }
            catch (const std::exception &e)
            {
                if (log_)
                    log_->Error("server.client", std::string("Exception in handler: ") + e.what());
                break;
            }

            if (!WriteFrame(fd, out_payload))
                break;
        }
    }

    // --- Protocol Implementations ---

    bool PomaiServer::ReadFrame(int fd, std::vector<std::uint8_t> &payload)
    {
        std::uint8_t len_le[4];
        if (!ReadExact(fd, len_le, 4))
            return false;

        std::uint32_t len = LoadU32LE(len_le);
        if (len == 0 || len > 64 * 1024 * 1024)
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
        rd.ReadU32(shards);
        rd.ReadU32(cap);

        if (dim == 0)
            dim = (std::uint16_t)cfg_.default_dim;

        // Check exists
        std::lock_guard<std::mutex> lk(cols_mu_);
        if (name_to_id_.count(name))
        {
            uint32_t existing_id = name_to_id_[name];
            Buf b;
            PutU8(b, 1);
            PutU32(b, existing_id);
            out_payload = std::move(b.data);
            return true;
        }

        pomai::DbOptions opt;
        opt.dim = dim;
        opt.metric = (metric_u8 == 0) ? pomai::Metric::L2 : pomai::Metric::Cosine;
        opt.shards = shards ? (std::size_t)shards : (std::size_t)cfg_.shards;
        opt.shard_queue_capacity = cap ? (std::size_t)cap : (std::size_t)cfg_.shard_queue_capacity;
        opt.wal_dir = cfg_.data_dir + "/" + name;

        // carry server-level policy into the new collection's options
        opt.allow_sync_on_append = cfg_.allow_sync_on_append;

        (void)EnsureDir(opt.wal_dir);

        // SAVE META
        SaveCollectionMeta(name, opt);

        auto db = std::make_shared<pomai::PomaiDB>(opt, log_);
        auto start_status = db->Start();
        if (!start_status.ok() && log_)
            log_->Error("server.collection", "Failed to start collection '" + name + "': " + start_status.msg);

        std::uint32_t id = next_col_id_++;
        cols_[id] = Col{name, dim, opt.metric, db};
        name_to_id_[name] = id;

        Buf b;
        PutU8(b, 1);
        PutU32(b, id);
        out_payload = std::move(b.data);

        if (log_)
            log_->Info("server.collection", "Created collection '" + name + "' id=" + std::to_string(id));
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

        // Optional per-request durability flag (u8). Default preserved as 'true' for backward compatibility.
        std::uint8_t wait_u8 = 1; // default true (existing behavior)
        // Try to read an extra byte; if absent, keep default.
        (void)rd.ReadU8(wait_u8);
        bool wait_durable = (wait_u8 != 0);

        std::shared_ptr<pomai::PomaiDB> db;
        {
            std::lock_guard<std::mutex> lk(cols_mu_);
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
            db = it->second.db;
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

        // Forward the per-request wait_durable flag into the DB call.
        // PomaiDB is expected to consult its DbOptions.allow_sync_on_append
        // and decide whether to honor per-request synchronous durability.
        auto fut = db->UpsertBatch(std::move(batch), wait_durable);
        auto res = fut.get();
        if (!res.ok())
        {
            RespErr(out_payload, std::string("upsert_batch failed: ") + res.status().msg);
            return true;
        }

        Buf b;
        PutU8(b, 1);
        PutU64(b, (std::uint64_t)res.value());
        out_payload = std::move(b.data);
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

        std::shared_ptr<pomai::PomaiDB> db;
        {
            std::lock_guard<std::mutex> lk(cols_mu_);
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
            db = it->second.db;
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

        auto resp_res = db->Search(req);
        if (!resp_res.ok())
        {
            RespErr(out_payload, std::string("search failed: ") + resp_res.status().msg);
            return true;
        }
        pomai::SearchResponse resp = resp_res.move_value();

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

} // namespace pomai::server
