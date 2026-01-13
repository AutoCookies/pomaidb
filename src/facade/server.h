/*
 * src/facade/server.h
 * PomaiServer (V2) - SQL-like text protocol only (no RESP).
 *
 * Clean, self-contained header implementation (inline methods) so it compiles
 * when included by src/main.cc. This file provides:
 *  - epoll-based non-blocking TCP server
 *  - per-connection buffering and a simple SQL-like parser (commands end with ';')
 *  - supported commands: CREATE MEMBRANCE, DROP MEMBRANCE, SHOW MEMBRANCES,
 *    USE, INSERT INTO, SEARCH, GET, DELETE (soft delete)
 *
 * Performance notes:
 *  - Epoll + non-blocking sockets used for scale.
 *  - Parsing is intentionally simple and allocation-light.
 *
 * Usage:
 *  - Construct PomaiServer(kv_map, pomai_db, port) and it will start a worker thread.
 *  - Stop with server.stop() which will shut down worker and close sockets.
 */

#pragma once

#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/eventfd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <sys/resource.h>
#include <errno.h>

#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstring>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <thread>
#include <functional>
#include <cstdint>
#include <mutex>
#include <cctype>

#include "src/core/map.h"
#include "src/core/metrics.h"
#include "src/core/config.h"
#include "src/core/pomai_db.h"

class PomaiServer
{
public:
    PomaiServer(PomaiMap *kv_map, pomai::core::PomaiDB *pomai_db, int port)
        : kv_map_(kv_map), pomai_db_(pomai_db), port_(port)
    {
        start();
    }

    ~PomaiServer()
    {
        stop();
    }

    void stop()
    {
        running_ = false;
        uint64_t u = 1;
        if (event_fd_ != -1)
        {
            ssize_t w = write(event_fd_, &u, sizeof(u));
            (void)w;
        }
        if (worker_.joinable())
            worker_.join();
        if (listen_fd_ != -1)
        {
            close(listen_fd_);
            listen_fd_ = -1;
        }
        if (epoll_fd_ != -1)
        {
            close(epoll_fd_);
            epoll_fd_ = -1;
        }
        if (event_fd_ != -1)
        {
            close(event_fd_);
            event_fd_ = -1;
        }

        std::lock_guard<std::mutex> lk(conns_mu_);
        for (auto &kv : conns_)
            close(kv.first);
        conns_.clear();
    }

private:
    PomaiMap *kv_map_;
    pomai::core::PomaiDB *pomai_db_;
    int port_;
    int listen_fd_ = -1;
    int epoll_fd_ = -1;
    int event_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::mutex conns_mu_;

    struct Conn
    {
        int fd;
        std::vector<char> rbuf;
        std::vector<char> wbuf;
        std::string sql_current_membr;
    };

    std::unordered_map<int, Conn> conns_; // protected by conns_mu_

    // ---------- Utilities ----------
    static inline std::string trim(const std::string &s)
    {
        const char *ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws);
        if (b == std::string::npos)
            return "";
        size_t e = s.find_last_not_of(ws);
        return s.substr(b, e - b + 1);
    }

    static inline std::vector<std::string> split_ws(const std::string &s)
    {
        std::istringstream iss(s);
        std::vector<std::string> out;
        std::string t;
        while (iss >> t)
            out.push_back(t);
        return out;
    }

    static inline std::string to_upper(const std::string &s)
    {
        std::string r = s;
        for (char &c : r)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return r;
    }

    static uint64_t fnv1a_hash(const void *data, size_t len)
    {
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < len; ++i)
        {
            h ^= bytes[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    static inline uint64_t hash_key(const std::string &k) { return fnv1a_hash(k.data(), k.size()); }

    static inline void append_end_marker(std::string &s)
    {
        if (s.empty() || s.back() != '\n')
            s.push_back('\n');
        s += "<END>\n";
    }

    // Parse CSV list of floats (no nested support)
    static std::vector<float> parse_float_list(const std::string &csv)
    {
        std::vector<float> out;
        std::istringstream iss(csv);
        std::string tok;
        while (std::getline(iss, tok, ','))
        {
            try
            {
                out.push_back(std::stof(trim(tok)));
            }
            catch (...)
            {
            }
        }
        return out;
    }

    // ---------- socket/epoll helpers (must be before start()) ----------
    void set_nonblocking(int fd)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0)
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    void add_event(int fd, uint32_t events)
    {
        struct epoll_event ev{};
        ev.events = events;
        ev.data.fd = fd;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) != 0)
        {
            throw std::runtime_error(std::string("epoll_ctl add failed: ") + std::strerror(errno));
        }
    }

    void mod_event(int fd, uint32_t events)
    {
        struct epoll_event ev{};
        ev.events = events;
        ev.data.fd = fd;
        if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, fd, &ev) != 0)
        {
            std::cerr << "epoll_ctl mod failed: " << std::strerror(errno) << "\n";
        }
    }

    void del_event(int fd)
    {
        epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
    }

    // ---------- SQL command execution ----------
    std::string exec_sql_command(Conn &c, const std::string &raw_cmd)
    {
        std::string cmd = trim(raw_cmd);
        if (cmd.empty())
            return std::string("ERR empty command\n");
        std::string up = to_upper(cmd);

        // SHOW MEMBRANCES;
        if (up == "SHOW MEMBRANCES;" || up == "SHOW MEMBRANCES")
        {
            auto list = pomai_db_->list_membrances();
            std::ostringstream ss;
            ss << "MEMBRANCES: " << list.size() << "\n";
            for (auto &m : list)
                ss << " - " << m << "\n";
            return ss.str();
        }

        // USE <name>;
        if (up.rfind("USE ", 0) == 0)
        {
            auto parts = split_ws(cmd);
            if (parts.size() >= 2)
            {
                std::string name = parts[1];
                if (!name.empty() && name.back() == ';')
                    name.pop_back();
                c.sql_current_membr = name;
                return std::string("OK: switched to membrance ") + name + "\n";
            }
            return "ERR: USE <name>;\n";
        }

        // CREATE MEMBRANCE <name> DIM <n> [RAM <mb>];
        if (up.rfind("CREATE MEMBRANCE", 0) == 0)
        {
            size_t pos_dim = up.find(" DIM ");
            if (pos_dim == std::string::npos)
                return "ERR: CREATE MEMBRANCE missing DIM\n";
            std::string name = trim(cmd.substr(std::string("CREATE MEMBRANCE").size(), pos_dim - std::string("CREATE MEMBRANCE").size()));
            std::string tail = trim(cmd.substr(pos_dim + 5)); // after "DIM "
            size_t ram_mb = 256;
            size_t dim = 0;
            {
                std::istringstream iss(tail);
                iss >> dim;
                std::string w;
                if (iss >> w)
                {
                    if (to_upper(w) == "RAM")
                    {
                        iss >> ram_mb;
                    }
                }
            }
            if (dim == 0)
                return "ERR: invalid DIM\n";
            pomai::core::MembranceConfig cfg;
            cfg.dim = dim;
            cfg.ram_mb = ram_mb;
            bool ok = pomai_db_->create_membrance(name, cfg);
            if (ok)
            {
                std::ostringstream ss;
                ss << "OK: created membrance " << name << " dim=" << dim << " ram=" << ram_mb << "MB\n";
                return ss.str();
            }
            return "ERR: create failed (exists or invalid)\n";
        }

        // DROP MEMBRANCE <name>;
        if (up.rfind("DROP MEMBRANCE", 0) == 0)
        {
            auto parts = split_ws(cmd);
            if (parts.size() < 3)
                return "ERR: DROP MEMBRANCE <name>;\n";
            std::string name = parts[2];
            if (!name.empty() && name.back() == ';')
                name.pop_back();
            bool ok = pomai_db_->drop_membrance(name);
            return ok ? std::string("OK: dropped ") + name + "\n" : std::string("ERR: drop failed\n");
        }

        // INSERT INTO <name> VALUES (<label>, [f1,f2,...]);
        {
            std::string upstart = to_upper(cmd.substr(0, std::min<size_t>(cmd.size(), 12)));
            if (upstart.rfind("INSERT INTO", 0) == 0 || to_upper(cmd).rfind("INSERT VALUES", 0) == 0)
            {
                std::string body = cmd;
                std::string name;
                if (to_upper(cmd).rfind("INSERT VALUES", 0) == 0)
                {
                    if (c.sql_current_membr.empty())
                        return "ERR: no current membrance (USE <name>)\n";
                    name = c.sql_current_membr;
                    body = std::string("INSERT INTO ") + name + " " + cmd.substr(std::string("INSERT VALUES").size());
                }
                size_t pos_values = to_upper(body).find("VALUES");
                if (pos_values == std::string::npos)
                    return "ERR: INSERT missing VALUES\n";
                size_t pos_into = to_upper(body).find("INTO");
                if (pos_into == std::string::npos)
                    return "ERR: INSERT syntax\n";
                size_t name_start = pos_into + 4;
                size_t name_end = pos_values;
                name = trim(body.substr(name_start, name_end - name_start));
                size_t lpar = body.find('(', pos_values);
                size_t rpar = body.find(')', lpar);
                if (lpar == std::string::npos || rpar == std::string::npos)
                    return "ERR: INSERT VALUES syntax\n";
                std::string inside = trim(body.substr(lpar + 1, rpar - lpar - 1));
                size_t comma = inside.find(',');
                if (comma == std::string::npos)
                    return "ERR: INSERT syntax\n";
                std::string label = trim(inside.substr(0, comma));
                std::string vecpart = trim(inside.substr(comma + 1));
                size_t lb = vecpart.find('[');
                size_t rb = vecpart.rfind(']');
                if (lb == std::string::npos || rb == std::string::npos || rb <= lb)
                    return "ERR: vector syntax\n";
                std::string veccsv = vecpart.substr(lb + 1, rb - lb - 1);
                auto vec = parse_float_list(veccsv);
                auto *m = pomai_db_->get_membrance(name);
                if (!m)
                    return "ERR: membrance not found\n";
                if (vec.size() != m->dim)
                {
                    std::ostringstream ss;
                    ss << "ERR: dim mismatch expected=" << m->dim << " got=" << vec.size() << "\n";
                    return ss.str();
                }
                uint64_t label_hash = hash_key(label);
                bool ok = m->orbit->insert(vec.data(), label_hash);
                return ok ? std::string("OK\n") : std::string("ERR: insert failed\n");
            }
        }

        // SEARCH <name> QUERY ([...]) TOP k;
        if (to_upper(cmd).rfind("SEARCH ", 0) == 0)
        {
            size_t pos_q = to_upper(cmd).find(" QUERY ");
            std::string name;
            size_t vec_lb = std::string::npos;
            size_t vec_rb = std::string::npos;
            if (pos_q != std::string::npos)
            {
                name = trim(cmd.substr(7, pos_q - 7));
                vec_lb = cmd.find('[', pos_q);
                vec_rb = cmd.find(']', vec_lb);
            }
            else
            {
                size_t pos_q2 = to_upper(cmd).find("SEARCH QUERY");
                if (pos_q2 == std::string::npos)
                    return "ERR: SEARCH syntax\n";
                if (c.sql_current_membr.empty())
                    return "ERR: no current membrance (USE <name>)\n";
                name = c.sql_current_membr;
                vec_lb = cmd.find('[', pos_q2);
                vec_rb = cmd.find(']', vec_lb);
            }
            if (vec_lb == std::string::npos || vec_rb == std::string::npos)
                return "ERR: SEARCH missing vector\n";
            std::string veccsv = cmd.substr(vec_lb + 1, vec_rb - vec_lb - 1);
            auto vec = parse_float_list(veccsv);
            size_t pos_top = to_upper(cmd).find(" TOP ", vec_rb);
            int topk = 10;
            if (pos_top != std::string::npos)
            {
                size_t start = pos_top + 5;
                size_t end = cmd.find(';', start);
                std::string kn = (end == std::string::npos) ? trim(cmd.substr(start)) : trim(cmd.substr(start, end - start));
                try
                {
                    topk = std::stoi(kn);
                }
                catch (...)
                {
                    topk = 10;
                }
            }
            auto *m = pomai_db_->get_membrance(name);
            if (!m)
                return "ERR: membrance not found\n";
            if (vec.size() != m->dim)
            {
                std::ostringstream ss;
                ss << "ERR: dim mismatch expected=" << m->dim << " got=" << vec.size() << "\n";
                return ss.str();
            }
            auto res = m->orbit->search(vec.data(), topk);
            std::ostringstream ss;
            ss << "RESULTS " << res.size() << "\n";
            for (auto &p : res)
                ss << p.first << " " << p.second << "\n";
            return ss.str();
        }

        // GET <name> LABEL <label>;
        if (to_upper(cmd).rfind("GET ", 0) == 0)
        {
            auto parts = split_ws(cmd);
            if (parts.size() < 4)
                return "ERR: GET <name> LABEL <label>\n";
            std::string name = parts[1];
            std::string label = parts[3];
            if (!label.empty() && label.back() == ';')
                label.pop_back();
            auto *m = pomai_db_->get_membrance(name);
            if (!m)
                return "ERR: membrance not found\n";
            std::vector<float> out;
            bool ok = m->orbit->get(hash_key(label), out);
            if (!ok)
                return "ERR: not found\n";
            std::ostringstream ss;
            ss << "VECTOR " << out.size() << " ";
            for (float v : out)
                ss << v << " ";
            ss << "\n";
            return ss.str();
        }

        // DELETE <name> LABEL <label>;
        if (to_upper(cmd).rfind("DELETE ", 0) == 0 || to_upper(cmd).rfind("VDEL ", 0) == 0)
        {
            auto parts = split_ws(cmd);
            if (parts.size() < 4)
                return "ERR: DELETE <name> LABEL <label>\n";
            std::string name = parts[1];
            std::string label = parts[3];
            if (!label.empty() && label.back() == ';')
                label.pop_back();
            auto *m = pomai_db_->get_membrance(name);
            if (!m)
                return "ERR: membrance not found\n";
            bool ok = m->orbit->remove(hash_key(label));
            return ok ? std::string("OK\n") : std::string("ERR: remove failed\n");
        }

        return "ERR: unknown command\n";
    }

    // send response + "<END>\n" marker
    void send_response(Conn &c, const std::string &resp)
    {
        std::string out = resp;
        append_end_marker(out);
        size_t written = 0;
        const char *ptr = out.data();
        size_t left = out.size();
        while (left > 0)
        {
            ssize_t n = write(c.fd, ptr + written, left);
            if (n < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                break;
            }
            written += static_cast<size_t>(n);
            left -= static_cast<size_t>(n);
        }
    }

    // Process any complete SQL commands (terminated by ';') in conn buffer.
    void process_sql_buffer(Conn &c)
    {
        size_t consumed = 0;
        for (size_t i = 0; i < c.rbuf.size(); ++i)
        {
            if (c.rbuf[i] == ';')
            {
                std::string cmd(c.rbuf.begin() + consumed, c.rbuf.begin() + i + 1);
                std::string resp = exec_sql_command(c, cmd);
                send_response(c, resp);
                consumed = i + 1;
            }
        }
        if (consumed > 0)
        {
            c.rbuf.erase(c.rbuf.begin(), c.rbuf.begin() + static_cast<ptrdiff_t>(consumed));
        }
    }

    void start()
    {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0)
            throw std::runtime_error("socket failed");

        int opt = 1;
        (void)setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, static_cast<socklen_t>(sizeof(opt)));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);

        if (bind(listen_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0)
            throw std::runtime_error("bind failed");

        if (listen(listen_fd_, 1024) < 0)
            throw std::runtime_error("listen failed");

        set_nonblocking(listen_fd_);

        epoll_fd_ = epoll_create1(0);
        if (epoll_fd_ < 0)
            throw std::runtime_error("epoll_create1 failed");

        event_fd_ = static_cast<int>(eventfd(0, EFD_NONBLOCK));
        if (event_fd_ < 0)
            throw std::runtime_error("eventfd failed");

        add_event(listen_fd_, EPOLLIN);
        add_event(event_fd_, EPOLLIN);

        running_ = true;
        worker_ = std::thread(&PomaiServer::loop, this);
    }

    void loop()
    {
        std::vector<struct epoll_event> events(1024);

        while (running_)
        {
            int n = epoll_wait(epoll_fd_, events.data(), static_cast<int>(events.size()), 100);
            if (n < 0)
            {
                if (errno == EINTR)
                    continue;
                break;
            }
            for (int i = 0; i < n; ++i)
            {
                int fd = events[i].data.fd;
                if (fd == event_fd_)
                {
                    uint64_t u;
                    if (read(fd, &u, sizeof(u)) < 0)
                    { /* ignore */
                    }
                }
                else if (fd == listen_fd_)
                {
                    int conn_fd = accept(listen_fd_, nullptr, nullptr);
                    if (conn_fd >= 0)
                    {
                        set_nonblocking(conn_fd);
                        add_event(conn_fd, EPOLLIN);
                        std::lock_guard<std::mutex> lk(conns_mu_);
                        conns_.emplace(conn_fd, Conn{conn_fd, {}, {}, ""});
                    }
                }
                else
                {
                    if (events[i].events & EPOLLIN)
                    {
                        char buf[4096];
                        ssize_t rb = read(fd, buf, sizeof(buf));
                        if (rb > 0)
                        {
                            std::lock_guard<std::mutex> lk(conns_mu_);
                            auto it = conns_.find(fd);
                            if (it == conns_.end())
                            {
                                close(fd);
                                continue;
                            }
                            Conn &c = it->second;
                            c.rbuf.insert(c.rbuf.end(), buf, buf + rb);
                            // Always treat as SQL-like text
                            process_sql_buffer(c);
                        }
                        else if (rb == 0 || (rb < 0 && errno != EAGAIN))
                        {
                            std::lock_guard<std::mutex> lk(conns_mu_);
                            close(fd);
                            conns_.erase(fd);
                        }
                    }
                }
            }
        }

        // cleanup
        std::lock_guard<std::mutex> lk(conns_mu_);
        for (auto &p : conns_)
            close(p.first);
        conns_.clear();
    }
};