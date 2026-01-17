/*
 * src/facade/server.h
 * PomaiServer (V3) - Turbo Parser + Batch INSERT support + WhisperGrain integration
 *
 * Upgrades:
 *  - Batch INSERT: support "INSERT INTO <name> VALUES (<label>, [v...]),(...)" and fast parsing
 *  - WhisperGrain controller integrated: server computes per-query budget and reports latencies/CPU
 *  - CPU sampler thread updates WhisperGrain periodically using /proc/stat
 *
 * Notes:
 *  - This file is a self-contained server header. It depends on pomai_db and orbit APIs
 *    including budget-aware search methods (search_with_budget / search_filtered_with_budget).
 *  - Added support for: GET MEMBRANCE INFO <name>;  (returns dim, estimated num_vectors, disk size)
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
#include <exception>
#include <string_view>
#include <cstdlib> // strtof
#include <string>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <iomanip>

#include "src/core/map.h"
#include "src/core/metrics.h"
#include "src/core/config.h"
#include "src/core/pomai_db.h"
#include "src/core/metadata_index.h" // for Tag
#include "src/ai/whispergrain.h"
#include "src/core/config.h"
#include "src/core/metrics.h"

class PomaiServer
{
public:
    PomaiServer(pomai::core::PomaiMap *kv_map, pomai::core::PomaiDB *pomai_db, const pomai::config::PomaiConfig &config)
        : kv_map_(kv_map),
          pomai_db_(pomai_db),
          config_(config),
          port_(config.net.port),
          whisper_(config.whisper),
          cpu_sampler_running_(false)
    {
        start();
    }

    ~PomaiServer()
    {
        stop();
    }

    void stop()
    {
        // stop CPU sampler first
        cpu_sampler_running_.store(false);
        if (cpu_sampler_thread_.joinable())
            cpu_sampler_thread_.join();

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

        // Best-effort snapshot on server stop (safe to call twice; main also does it).
        try
        {
            if (pomai_db_)
            {
                bool ok_schemas = pomai_db_->save_all_membrances();
                bool ok_manifest = pomai_db_->save_manifest();
                if (!ok_schemas || !ok_manifest)
                {
                    std::cerr << "[PomaiServer] Warning: snapshot on stop completed with warnings. Schemas: "
                              << (ok_schemas ? "ok" : "fail") << ", manifest: " << (ok_manifest ? "ok" : "fail") << "\n";
                }
                else
                {
                    std::clog << "[PomaiServer] Snapshot on stop completed.\n";
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "[PomaiServer] Exception while snapshotting DB on stop: " << e.what() << "\n";
        }
        catch (...)
        {
            std::cerr << "[PomaiServer] Unknown exception while snapshotting DB on stop\n";
        }
    }

private:
    pomai::core::PomaiMap *kv_map_;
    pomai::core::PomaiDB *pomai_db_;
    pomai::config::PomaiConfig config_;
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

    // WhisperGrain controller (single instance for server)
    pomai::ai::WhisperGrain whisper_;
    // Simple frequency map for hot-query detection (string -> count)
    std::unordered_map<std::string, uint32_t> query_freq_;
    mutable std::mutex query_freq_mu_;

    // CPU sampler thread
    std::atomic<bool> cpu_sampler_running_;
    std::thread cpu_sampler_thread_;
    // previous cpu snapshot for /proc/stat sampling
    uint64_t prev_total_{0}, prev_idle_{0};

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

    static inline std::string_view trim_sv(std::string_view v)
    {
        const char *ws = " \t\r\n";
        size_t b = 0;
        while (b < v.size() && std::strchr(ws, static_cast<unsigned char>(v[b])))
            ++b;
        size_t e = v.size();
        while (e > b && std::strchr(ws, static_cast<unsigned char>(v[e - 1])))
            --e;
        return v.substr(b, e - b);
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

    static inline uint64_t hash_label(const std::string &s)
    {
        uint64_t h = 14695981039346656037ULL;
        for (char c : s)
        {
            h ^= static_cast<uint64_t>(c);
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

    // Fast parse helpers: use string_view and strtof into a small stack buffer.
    // Token is a substring view representing the number text (no surrounding whitespace).
    static inline float parse_float_token_sv(std::string_view tok)
    {
        // small stack buffer optimization
        constexpr size_t STACK_BUF_SZ = 128;
        if (tok.empty())
            return 0.0f;
        if (tok.size() < STACK_BUF_SZ)
        {
            char tmp[STACK_BUF_SZ];
            std::memcpy(tmp, tok.data(), tok.size());
            tmp[tok.size()] = '\0';
            char *endp = nullptr;
            float v = std::strtof(tmp, &endp);
            (void)endp;
            return v;
        }
        else
        {
            // fallback allocate
            std::string tmp(tok);
            char *endp = nullptr;
            float v = std::strtof(tmp.c_str(), &endp);
            (void)endp;
            return v;
        }
    }

    // Parse CSV list of floats using string_view tokens and parse_float_token_sv.
    static std::vector<float> parse_float_list_sv(std::string_view csv)
    {
        std::vector<float> out;
        size_t i = 0;
        while (i < csv.size())
        {
            // find comma
            size_t j = csv.find(',', i);
            std::string_view tok = (j == std::string::npos) ? csv.substr(i) : csv.substr(i, j - i);
            tok = trim_sv(tok);
            if (!tok.empty())
            {
                try
                {
                    float v = parse_float_token_sv(tok);
                    out.push_back(v);
                }
                catch (...)
                {
                    // ignore parse error for this token
                }
            }
            if (j == std::string::npos)
                break;
            i = j + 1;
        }
        return out;
    }

    // Parse tags in form: k1:v1,k2:v2  -> returns vector<Tag>
    static std::vector<pomai::core::Tag> parse_tags_list(const std::string &s)
    {
        std::vector<pomai::core::Tag> out;
        size_t i = 0;
        while (i < s.size())
        {
            // find comma-separated token
            size_t j = s.find(',', i);
            std::string tok = (j == std::string::npos) ? s.substr(i) : s.substr(i, j - i);
            // split by ':' first occurrence
            size_t c = tok.find(':');
            if (c != std::string::npos)
            {
                std::string k = trim(tok.substr(0, c));
                std::string v = trim(tok.substr(c + 1));
                // strip optional quotes
                if (v.size() >= 2 && ((v.front() == '\'' && v.back() == '\'') || (v.front() == '\"' && v.back() == '\"')))
                    v = v.substr(1, v.size() - 2);
                if (!k.empty() && !v.empty())
                    out.push_back(pomai::core::Tag{k, v});
            }
            if (j == std::string::npos)
                break;
            i = j + 1;
        }
        return out;
    }

    // Helper: format bytes -> GB string with 2 decimals
    static std::string bytes_human(uint64_t bytes)
    {
        std::ostringstream ss;
        double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        ss << bytes << " (" << std::fixed << std::setprecision(2) << gb << " GB)";
        return ss.str();
    }

    // ---------- /proc/stat CPU sampling helpers ----------
    // Read first line of /proc/stat and parse cpu times
    static bool read_proc_stat(uint64_t &idle, uint64_t &total)
    {
        std::ifstream f("/proc/stat");
        if (!f.good())
            return false;
        std::string line;
        std::getline(f, line);
        f.close();
        // line format: cpu  user nice system idle iowait irq softirq steal guest guest_nice
        std::istringstream iss(line);
        std::string cpu_label;
        iss >> cpu_label;
        uint64_t value;
        uint64_t sum = 0;
        idle = 0;
        for (int i = 0; iss >> value; ++i)
        {
            sum += value;
            // idle is value at position 3 (0-based: user(0),nice(1),system(2),idle(3))
            if (i == 3)
                idle = value;
        }
        total = sum;
        return true;
    }

    void cpu_sampler_loop()
    {
        // initialize prev snapshot
        uint64_t idle, total;
        if (!read_proc_stat(prev_idle_, prev_total_))
        {
            // cannot read procstat: set cpu to 0 and return
            whisper_.set_cpu_load(0.0f);
            return;
        }

        const std::chrono::milliseconds sample_interval(config_.server.cpu_sample_interval_ms);
        while (cpu_sampler_running_.load())
        {
            std::this_thread::sleep_for(sample_interval);
            if (!read_proc_stat(idle, total))
                continue;
            uint64_t delta_idle = idle - prev_idle_;
            uint64_t delta_total = total - prev_total_;
            prev_idle_ = idle;
            prev_total_ = total;
            float cpu_percent = 0.0f;
            if (delta_total > 0)
            {
                cpu_percent = 100.0f * (1.0f - (static_cast<double>(delta_idle) / static_cast<double>(delta_total)));
                if (cpu_percent < 0.0f)
                    cpu_percent = 0.0f;
                if (cpu_percent > 100.0f)
                    cpu_percent = 100.0f;
            }
            whisper_.set_cpu_load(cpu_percent);
        }
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

        if (up.rfind("EXEC SPLIT", 0) == 0)
        {
            auto parts = split_ws(cmd);
            if (parts.size() < 6)
                return "ERR: Usage: EXEC SPLIT <name> <train> <val> <test>\n";

            std::string name = parts[2];
            float train = 0.0f, val = 0.0f, test = 0.0f;

            try
            {
                train = std::stof(parts[3]);
                val = std::stof(parts[4]);
                test = std::stof(parts[5]);
            }
            catch (...)
            {
                return "ERR: Invalid ratio format (must be float)\n";
            }

            // Validate tổng <= 1.0 (cho phép sai số nhỏ do float)
            if (train + val + test > 1.001f)
                return "ERR: Ratios sum must be <= 1.0\n";

            auto *m = pomai_db_->get_membrance(name);
            if (!m)
                return "ERR: Membrance not found\n";

            if (!m->split_mgr)
                return "ERR: SplitManager not initialized for this membrance\n";

            // Lấy tổng số vector hiện có để chia
            size_t total_vectors = 0;
            try
            {
                // Orbit là nơi giữ thông tin chính xác nhất về số lượng vector đã insert
                auto info = m->orbit->get_info();
                total_vectors = info.num_vectors;
            }
            catch (...)
            {
                return "ERR: Could not retrieve vector count from engine\n";
            }

            if (total_vectors == 0)
                return "ERR: Membrance is empty, nothing to split\n";

            // Thực hiện chia
            std::clog << "[Server] Splitting " << name << " (" << total_vectors << " vecs) "
                      << " T:" << train << " V:" << val << " T:" << test << "\n";

            m->split_mgr->execute_random_split(total_vectors, train, val, test);

            // Lưu ngay xuống đĩa
            if (m->split_mgr->save(m->data_path))
            {
                std::ostringstream ss;
                ss << "OK: Split " << total_vectors << " vectors into "
                   << m->split_mgr->train_indices.size() << " train, "
                   << m->split_mgr->val_indices.size() << " val, "
                   << m->split_mgr->test_indices.size() << " test\n";
                return ss.str();
            }
            else
            {
                return "ERR: Failed to save split file to disk\n";
            }
        }

        if (up.rfind("EXEC SPLIT", 0) == 0)
        {
            auto parts = split_ws(cmd);
            if (parts.size() < 6)
                return "ERR: Usage: EXEC SPLIT <name> <tr%> <val%> <te%> [STRATIFIED <key> | CLUSTER]\n";

            std::string name = parts[2];
            float tr = 0, val = 0, te = 0;
            try
            {
                tr = std::stof(parts[3]);
                val = std::stof(parts[4]);
                te = std::stof(parts[5]);
            }
            catch (...)
            {
                return "ERR: Invalid split percentages\n";
            }

            if (tr + val + te > 1.001f)
                return "ERR: Ratios sum must be <= 1.0\n";

            auto *m = pomai_db_->get_membrance(name);
            if (!m)
                return "ERR: Membrance not found\n";
            if (!m->split_mgr)
                return "ERR: Split manager not initialized\n";

            // --- Detect Strategy ---
            std::string strategy = "RANDOM";
            std::string strat_key = "";

            if (parts.size() >= 7)
            {
                std::string type = to_upper(parts[6]);
                if (type == "STRATIFIED")
                {
                    strategy = "STRATIFIED";
                    if (parts.size() >= 8)
                        strat_key = parts[7];
                    else
                        return "ERR: STRATIFIED requires a key (e.g. STRATIFIED class)\n";
                }
                else if (type == "CLUSTER")
                {
                    strategy = "CLUSTER";
                }
            }

            // Check Empty (fallback to 0 if info fails)
            size_t total_vectors = 0;
            try
            {
                total_vectors = m->orbit->get_info().num_vectors;
            }
            catch (...)
            {
            }

            // --- Strategy Dispatch ---

            // 1. STRATIFIED SPLIT (Chia đều theo nhãn)
            if (strategy == "STRATIFIED")
            {
                if (!m->meta_index)
                    return "ERR: Metadata Index not enabled for this membrance\n";

                auto groups = m->meta_index->get_groups(strat_key);
                if (groups.empty())
                    return "ERR: No metadata found for key '" + strat_key + "'\n";

                // Flatten map -> vector for manager
                std::vector<uint64_t> items;
                std::vector<uint64_t> labels;
                for (const auto &kv : groups)
                {
                    uint64_t label_hash = hash_label(kv.first);
                    for (uint64_t id : kv.second)
                    {
                        items.push_back(id);
                        labels.push_back(label_hash);
                    }
                }

                m->split_mgr->execute_stratified_split(items, labels, tr, val, te);
                total_vectors = items.size();
            }
            // 2. CLUSTER SPLIT (Chia theo cụm không gian)
            else if (strategy == "CLUSTER")
            {
                if (!m->orbit)
                    return "ERR: Orbit engine required for Cluster split\n";

                size_t num_c = m->orbit->num_centroids();
                if (num_c == 0)
                    return "ERR: No centroids found (Train model first)\n";

                // Danh sách ID của các cụm [0...N-1]
                std::vector<uint32_t> cids(num_c);
                std::iota(cids.begin(), cids.end(), 0);

                // Trộn ngẫu nhiên thứ tự các cụm
                std::mt19937 g(std::random_device{}());
                std::shuffle(cids.begin(), cids.end(), g);

                // Tính điểm cắt trên danh sách cụm
                size_t n_train_c = static_cast<size_t>(num_c * tr);
                size_t n_val_c = static_cast<size_t>(num_c * val);

                std::vector<uint64_t> train_items, val_items, test_items;
                size_t idx = 0;

                // Gom ID từ từng cụm vào các tập
                for (; idx < n_train_c; ++idx)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[idx]);
                    train_items.insert(train_items.end(), vec_ids.begin(), vec_ids.end());
                }
                for (; idx < n_train_c + n_val_c; ++idx)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[idx]);
                    val_items.insert(val_items.end(), vec_ids.begin(), vec_ids.end());
                }
                for (; idx < num_c; ++idx)
                {
                    auto vec_ids = m->orbit->get_centroid_ids(cids[idx]);
                    test_items.insert(test_items.end(), vec_ids.begin(), vec_ids.end());
                }

                // Gán trực tiếp vào SplitManager
                m->split_mgr->reset();
                m->split_mgr->train_indices = std::move(train_items);
                m->split_mgr->val_indices = std::move(val_items);
                m->split_mgr->test_indices = std::move(test_items);

                total_vectors = m->split_mgr->train_indices.size() +
                                m->split_mgr->val_indices.size() +
                                m->split_mgr->test_indices.size();
            } else if (strategy == "TEMPORAL")
            {
                if (!m->meta_index)
                    return "ERR: Metadata Index not enabled\n";
                if (strat_key.empty())
                    return "ERR: TEMPORAL requires a key (e.g. TEMPORAL date)\n";

                // std::map tự động sắp xếp theo Key (thời gian) tăng dần
                auto groups = m->meta_index->get_groups(strat_key);
                if (groups.empty())
                    return "ERR: No metadata found for key\n";

                // 1. Flatten dữ liệu theo đúng thứ tự thời gian
                std::vector<uint64_t> all_ordered_items;
                for (const auto &kv : groups)
                {
                    // kv.first là mốc thời gian (đã sort)
                    // kv.second là danh sách ID trong mốc đó
                    all_ordered_items.insert(all_ordered_items.end(), kv.second.begin(), kv.second.end());
                }

                if (all_ordered_items.empty())
                    return "ERR: No items found\n";

                // 2. Cắt tuyến tính (Linear Cut)
                size_t n = all_ordered_items.size();
                size_t n_train = static_cast<size_t>(n * tr);
                size_t n_val = static_cast<size_t>(n * val);

                std::vector<uint64_t> train_items, val_items, test_items;

                auto it = all_ordered_items.begin();

                // Train: Lấy đoạn đầu (Quá khứ)
                if (n_train > 0)
                {
                    train_items.assign(it, it + n_train);
                    it += n_train;
                }

                // Val: Lấy đoạn giữa
                if (n_val > 0)
                {
                    val_items.assign(it, it + n_val);
                    it += n_val;
                }

                // Test: Lấy đoạn cuối (Tương lai)
                if (it != all_ordered_items.end())
                {
                    test_items.assign(it, all_ordered_items.end());
                }

                // 3. Gán vào SplitManager
                m->split_mgr->reset();
                m->split_mgr->train_indices = std::move(train_items);
                m->split_mgr->val_indices = std::move(val_items);
                m->split_mgr->test_indices = std::move(test_items);

                total_vectors = n;
            }
            // 3. RANDOM SPLIT (Mặc định)
            else
            {
                if (total_vectors == 0)
                    return "ERR: Membrance is empty (0 vectors)\n";

                // Lưu ý: Random split mặc định dùng chỉ số ảo 0..N.
                // Nếu muốn chính xác ID, cần dùng item-based split nhưng chậm hơn.
                // Hiện tại giữ nguyên random index-based cho tốc độ.
                m->split_mgr->execute_random_split(total_vectors, tr, val, te);
            }

            // Save result to disk
            if (m->split_mgr->save(m->data_path))
            {
                std::stringstream ss;
                ss << "OK: Split " << total_vectors << " vectors into "
                   << m->split_mgr->train_indices.size() << " train, "
                   << m->split_mgr->val_indices.size() << " val, "
                   << m->split_mgr->test_indices.size() << " test";
                return ss.str() + "\n";
            }
            else
            {
                return "ERR: Failed to save split file to disk\n";
            }
        }

        // ---------------------------------------------------------
        // 2. NEW: ITERATE COMMAND (Binary Stream)
        // Syntax: ITERATE <name> <TRAIN|VAL|TEST> <offset> <limit>
        // ---------------------------------------------------------
        if (up.rfind("ITERATE", 0) == 0)
        {
            auto parts = split_ws(cmd);
            if (parts.size() < 5)
                return "ERR: Usage: ITERATE <name> <split> <off> <lim>\n";

            std::string name = parts[1];
            std::string type = to_upper(parts[2]);
            size_t off = 0, lim = 0;
            try
            {
                off = std::stoul(parts[3]);
                lim = std::stoul(parts[4]);
            }
            catch (...)
            {
            }

            auto *m = pomai_db_->get_membrance(name);
            if (!m || !m->split_mgr)
                return "ERR: Invalid membrance or no split\n";

            const std::vector<uint64_t> *indices = nullptr;
            if (type == "TRAIN")
                indices = &m->split_mgr->train_indices;
            else if (type == "VAL")
                indices = &m->split_mgr->val_indices;
            else if (type == "TEST")
                indices = &m->split_mgr->test_indices;
            else
                return "ERR: Invalid split type\n";

            if (off >= indices->size())
                return "ERR: Offset out of range\n";
            size_t end = std::min(off + lim, indices->size());
            size_t count = end - off;

            // Protocol: OK BINARY <count> <dim> <bytes>\n[RAW BYTES]
            size_t dim = m->dim;
            size_t bytes_per_vec = dim * sizeof(float);
            size_t total_bytes = count * bytes_per_vec;

            std::string header = "OK BINARY " + std::to_string(count) + " " +
                                 std::to_string(dim) + " " + std::to_string(total_bytes) + "\n";

            std::string response;
            response.reserve(header.size() + total_bytes + 16);
            response += header;

            // Bulk Copy Data
            for (size_t i = 0; i < count; ++i)
            {
                uint64_t offset = (*indices)[off + i];
                const char *ptr = m->arena->blob_ptr_from_offset_for_map(offset);

                if (ptr)
                {
                    // Skip 4 bytes length header, copy float array
                    response.append(ptr + sizeof(uint32_t), bytes_per_vec);
                }
                else
                {
                    // Padding nếu lỗi (Zero vector)
                    response.append(bytes_per_vec, 0);
                }
            }
            // Không thêm \n hay <END> ở cuối phần binary để giữ tốc độ và sự đơn giản
            return response;
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

        // GET MEMBRANCE INFO [<name>];
        // Supported forms:
        //   GET MEMBRANCE INFO;            -> uses current membrance in conn (USE ...)
        //   GET MEMBRANCE INFO <name>;
        //   GET MEMBRANCE <name> INFO;
        if (up.rfind("GET MEMBRANCE", 0) == 0)
        {
            auto parts = split_ws(cmd);
            if (!parts.empty() && !parts.back().empty() && parts.back().back() == ';')
            {
                parts.back() = parts.back().substr(0, parts.back().size() - 1);
            }

            std::string name;
            bool ok_parse = false;

            // Parsing logic (giữ nguyên logic ưu tiên của bạn)
            if (parts.size() >= 4 && to_upper(parts[2]) == "INFO") {
                name = parts[3]; ok_parse = true;
            } else if (parts.size() >= 4 && to_upper(parts[3]) == "INFO") {
                name = parts[2]; ok_parse = true;
            } else if (parts.size() >= 3 && to_upper(parts[2]) == "INFO") {
                if (c.sql_current_membr.empty()) return "ERR: no current membrance (USE <name>)\n";
                name = c.sql_current_membr; ok_parse = true;
            }

            if (!ok_parse) return "ERR: expected 'GET MEMBRANCE INFO ...'\n";

            auto *m = pomai_db_->get_membrance(name);
            if (!m) return std::string("ERR: membrance not found: ") + name + "\n";

            // 1. Gather Storage Info (Physical)
            pomai::ai::orbit::MembranceInfo info;
            try {
                info = m->orbit->get_info();
            } catch (...) {
                // fallback: use available membrance fields
                info.dim = m->dim;
                info.num_vectors = 0;
                info.disk_bytes = 0;
            }

            // Disk calc fallback (if orbit returned 0)
            if (info.disk_bytes == 0) {
                try {
                    std::filesystem::path dp(m->data_path);
                    if (std::filesystem::exists(dp)) {
                        for (auto const &entry : std::filesystem::recursive_directory_iterator(dp)) {
                            if (!entry.is_regular_file()) continue;
                            std::error_code ec;
                            uint64_t fsz = static_cast<uint64_t>(entry.file_size(ec));
                            if (!ec) info.disk_bytes += fsz;
                        }
                    }
                } catch (...) {}
            }

            // 2. Gather AI Contract Info (Logical Split) -> ĐÂY LÀ PHẦN MỚI
            size_t n_train = 0, n_val = 0, n_test = 0;
            if (m->split_mgr) {
                n_train = m->split_mgr->train_indices.size();
                n_val = m->split_mgr->val_indices.size();
                n_test = m->split_mgr->test_indices.size();
            }

            // Determine feature dim (prefer explicit membrance dim)
            size_t feature_dim = (m->dim > 0) ? m->dim : info.dim;

            // Determine total vectors: prefer Orbit info, fallback to splits sum
            size_t total_vectors = info.num_vectors;
            if (total_vectors == 0) {
                total_vectors = n_train + n_val + n_test;
            }

            // 3. Construct Unified Report (match your requested format)
            std::ostringstream ss;
            ss << "MEMBRANCE: " << name << "\n";
            ss << "--- AI Contract ---\n";
            ss << " feature_dim: " << feature_dim << "\n";
            ss << " metric: L2\n";
            ss << " data_type: float32\n";
            ss << " total_vectors: " << total_vectors << "\n";
            ss << " split_train: " << n_train << "\n";
            ss << " split_val: " << n_val << "\n";
            ss << " split_test: " << n_test << "\n";
            ss << "--- Storage Stats ---\n";
            ss << " disk_bytes: " << bytes_human(info.disk_bytes) << "\n";
            ss << " ram_mb_configured: " << m->ram_mb << "\n";

            return ss.str();
        }

        // INSERT INTO <name> VALUES (<label>, [f1,f2,...]) (,(<label2>,[...])... ) [TAGS (...)]
        {
            std::string upstart = to_upper(cmd.substr(0, std::min<size_t>(cmd.size(), 16)));
            if (upstart.rfind("INSERT INTO", 0) == 0 || to_upper(cmd).rfind("INSERT VALUES", 0) == 0)
            {
                std::string body = cmd;
                std::string name;
                bool has_explicit_into = (to_upper(cmd).find("INTO") != std::string::npos);
                if (!has_explicit_into && to_upper(cmd).rfind("INSERT VALUES", 0) == 0)
                {
                    if (c.sql_current_membr.empty())
                        return "ERR: no current membrance (USE <name>)\n";
                    name = c.sql_current_membr;
                    body = std::string("INSERT INTO ") + name + " " + cmd.substr(std::string("INSERT VALUES").size());
                }
                // find INTO and VALUES
                size_t pos_into = to_upper(body).find("INTO");
                size_t pos_values = to_upper(body).find("VALUES");
                if (pos_values == std::string::npos)
                    return "ERR: INSERT missing VALUES\n";
                if (pos_into == std::string::npos)
                    return "ERR: INSERT missing INTO\n";
                // extract name
                size_t name_start = pos_into + 4;
                name = trim(body.substr(name_start, pos_values - name_start));
                if (name.empty())
                    return "ERR: missing membrance name\n";

                auto *m = pomai_db_->get_membrance(name);
                if (!m)
                    return "ERR: membrance not found\n";

                // After VALUES we expect a sequence of tuples: ( ... ),( ... ),...
                size_t pos_after_values = pos_values + std::string("VALUES").size();
                size_t cur = body.find_first_not_of(" \t\r\n", pos_after_values);
                if (cur == std::string::npos || body[cur] != '(')
                    return "ERR: VALUES syntax\n";

                // collect tuples
                std::vector<std::pair<std::string, std::vector<float>>> tuples;
                tuples.reserve(8);

                while (true)
                {
                    // find matching parenthesis pair for tuple starting at cur (which should be '(')
                    if (cur >= body.size() || body[cur] != '(')
                        break;
                    size_t depth = 0;
                    size_t start = cur;
                    size_t end = std::string::npos;
                    for (size_t p = cur; p < body.size(); ++p)
                    {
                        if (body[p] == '(')
                            ++depth;
                        else if (body[p] == ')')
                        {
                            --depth;
                            if (depth == 0)
                            {
                                end = p;
                                cur = p + 1;
                                break;
                            }
                        }
                    }
                    if (end == std::string::npos)
                        return "ERR: unmatched parentheses in VALUES\n";

                    std::string tuple_text = body.substr(start + 1, end - start - 1); // contents inside ( ... )
                    // parse tuple: expected <label>, [f1,f2,...]
                    // find first comma that separates label and vector (not comma inside vector)
                    // We'll find first '[' to locate vector start
                    size_t vec_lb = tuple_text.find('[');
                    size_t vec_rb = tuple_text.rfind(']');
                    if (vec_lb == std::string::npos || vec_rb == std::string::npos || vec_rb <= vec_lb)
                        return "ERR: tuple vector syntax\n";
                    // label text is before the comma that precedes the '['
                    size_t comma_before_vec = tuple_text.rfind(',', vec_lb);
                    if (comma_before_vec == std::string::npos)
                        return "ERR: tuple syntax (label, [vec])\n";
                    std::string label_tok = trim(tuple_text.substr(0, comma_before_vec));
                    std::string_view vec_view(tuple_text.data() + vec_lb + 1, vec_rb - vec_lb - 1);

                    // parse floats fast
                    auto vec_vals = parse_float_list_sv(trim_sv(vec_view));
                    if (vec_vals.size() != m->dim)
                    {
                        std::ostringstream ss;
                        ss << "ERR: dim mismatch expected=" << m->dim << " got=" << vec_vals.size() << "\n";
                        return ss.str();
                    }

                    tuples.emplace_back(label_tok, std::move(vec_vals));

                    // skip whitespace, if next char is ',' then continue to next tuple group, else break
                    cur = body.find_first_not_of(" \t\r\n", cur);
                    if (cur == std::string::npos)
                        break;
                    if (body[cur] == ',')
                    {
                        ++cur;
                        cur = body.find_first_not_of(" \t\r\n", cur);
                        continue;
                    }
                    break;
                }

                // After tuple list, optional TAGS(...)
                std::vector<pomai::core::Tag> global_tags;
                size_t pos_tags = to_upper(body).find(" TAGS ", pos_after_values);
                if (pos_tags != std::string::npos)
                {
                    size_t t_l = body.find('(', pos_tags);
                    size_t t_r = body.find(')', t_l);
                    if (t_l != std::string::npos && t_r != std::string::npos && t_r > t_l)
                    {
                        std::string tags_inside = body.substr(t_l + 1, t_r - t_l - 1);
                        global_tags = parse_tags_list(tags_inside);
                    }
                }

                // --- Fast Batch Insert Path ---
                // Prepare batch payload for PomaiDB
                std::vector<std::pair<uint64_t, std::vector<float>>> batch_data;
                batch_data.reserve(tuples.size());
                std::vector<uint64_t> inserted_hashes;
                if (!global_tags.empty())
                    inserted_hashes.reserve(tuples.size());

                for (auto &tp : tuples)
                {
                    uint64_t label_hash = hash_key(tp.first);
                    batch_data.emplace_back(label_hash, std::move(tp.second));
                    if (!global_tags.empty())
                        inserted_hashes.push_back(label_hash);
                }

                bool ok_batch = false;
                try
                {
                    ok_batch = pomai_db_->insert_batch(name, batch_data);
                }
                catch (...)
                {
                    ok_batch = false;
                }

                // If batch insert succeeded, attach global tags (if any) via membrance meta index
                if (ok_batch && !global_tags.empty() && m->meta_index)
                {
                    for (uint64_t h : inserted_hashes)
                    {
                        try
                        {
                            m->meta_index->add_tags(h, global_tags);
                        }
                        catch (...)
                        {
                            std::clog << "[PomaiServer] Warning: failed to add tags for label " << h << "\n";
                        }
                    }
                }

                std::ostringstream ss;
                ss << "OK: inserted " << (ok_batch ? batch_data.size() : 0) << " / " << tuples.size() << (ok_batch ? " (batch)" : " (failed)") << "\n";
                return ss.str();
            }
        }

        // SEARCH <name> QUERY ([...]) [WHERE key='val'] TOP k;
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
            auto vec = parse_float_list_sv(trim_sv(std::string_view(veccsv)));
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

            // Check for WHERE clause (simple form: WHERE key='value' or WHERE key="value" or WHERE key=value)
            std::string upcmd = to_upper(cmd);
            size_t pos_where = upcmd.find(" WHERE ", vec_rb);
            std::vector<std::pair<uint64_t, float>> res;

            // Determine hot-ness key
            std::string hot_key_for_freq;

            if (pos_where != std::string::npos && m->meta_index)
            {
                size_t cond_start = pos_where + 7;
                size_t cond_end = std::string::npos;
                // try to find TOP after WHERE
                size_t pos_top_after = upcmd.find(" TOP ", cond_start);
                if (pos_top_after != std::string::npos)
                    cond_end = pos_top_after;
                else
                {
                    size_t semi = cmd.find(';', cond_start);
                    cond_end = (semi == std::string::npos) ? cmd.size() : semi;
                }
                std::string cond = trim(cmd.substr(cond_start, cond_end - cond_start));
                // parse key=value (value may be quoted)
                size_t eq = cond.find('=');
                if (eq != std::string::npos)
                {
                    std::string key = trim(cond.substr(0, eq));
                    std::string val = trim(cond.substr(eq + 1));
                    if (val.size() >= 2 && ((val.front() == '\'' && val.back() == '\'') || (val.front() == '\"' && val.back() == '\"')))
                        val = val.substr(1, val.size() - 2);

                    // Get candidates from metadata index
                    std::vector<uint64_t> candidates = m->meta_index->filter(key, val);

                    // hot-key for frequency tracking:
                    hot_key_for_freq = key + "=" + val;

                    // Use filtered (budgeted) search
                    bool is_hot = false;
                    {
                        std::lock_guard<std::mutex> qlk(query_freq_mu_);
                        uint32_t &cnt = query_freq_[hot_key_for_freq];
                        cnt++;
                        if (cnt >= 5)
                            is_hot = true; // simple threshold
                    }

                    // compute budget and time search
                    auto budget = whisper_.compute_budget(is_hot);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    try
                    {
                        res = m->orbit->search_filtered_with_budget(vec.data(), static_cast<size_t>(topk), candidates, budget);
                    }
                    catch (...)
                    {
                        res = m->orbit->search(vec.data(), static_cast<size_t>(topk));
                    }
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    whisper_.observe_latency(static_cast<float>(ms));
                }
                else
                {
                    // malformed where -> fallback to regular search
                    auto budget = whisper_.compute_budget(false);
                    auto t0 = std::chrono::high_resolution_clock::now();
                    res = m->orbit->search_with_budget(vec.data(), static_cast<size_t>(topk), budget);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    whisper_.observe_latency(static_cast<float>(ms));
                }
            }
            else
            {
                // No WHERE or no meta index -> regular budgeted search
                // hot-key by membrance name
                hot_key_for_freq = name;
                bool is_hot = false;
                {
                    std::lock_guard<std::mutex> qlk(query_freq_mu_);
                    uint32_t &cnt = query_freq_[hot_key_for_freq];
                    cnt++;
                    if (cnt >= 5)
                        is_hot = true;
                }
                auto budget = whisper_.compute_budget(is_hot);
                auto t0 = std::chrono::high_resolution_clock::now();
                try
                {
                    res = m->orbit->search_with_budget(vec.data(), static_cast<size_t>(topk), budget);
                }
                catch (...)
                {
                    res = m->orbit->search(vec.data(), static_cast<size_t>(topk));
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                whisper_.observe_latency(static_cast<float>(ms));
            }

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

        if (listen(listen_fd_, config_.server.backlog) < 0)
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

        // Start CPU sampler thread
        cpu_sampler_running_.store(true);
        cpu_sampler_thread_ = std::thread([this]()
                                          { cpu_sampler_loop(); });

        running_ = true;
        worker_ = std::thread(&PomaiServer::loop, this);
    }

    void loop()
    {
        std::vector<struct epoll_event> events(config_.server.max_events);

        while (running_)
        {
            int n = epoll_wait(epoll_fd_, events.data(), static_cast<int>(events.size()), config_.server.epoll_timeout_ms);
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
                        char buf[pomai::config::SERVER_READ_BUFFER];
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