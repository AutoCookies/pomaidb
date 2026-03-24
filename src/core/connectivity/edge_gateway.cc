#include "core/connectivity/edge_gateway.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>
#include <string_view>
#include <vector>
#include <set>

#include "core/membrane/manager.h"

namespace pomai::core {

namespace {
int OpenListenSocket(uint16_t port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int on = 1;
    (void)::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return -1;
    }
    if (::listen(fd, 16) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

bool ParseHttpUrl(const std::string& url, std::string* host, uint16_t* port, std::string* path) {
    if (!host || !port || !path) return false;
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) != 0) return false;
    const std::string rem = url.substr(prefix.size());
    const auto slash = rem.find('/');
    const std::string hostport = (slash == std::string::npos) ? rem : rem.substr(0, slash);
    *path = (slash == std::string::npos) ? "/" : rem.substr(slash);
    const auto colon = hostport.find(':');
    if (colon == std::string::npos) {
        *host = hostport;
        *port = 80;
    } else {
        *host = hostport.substr(0, colon);
        *port = static_cast<uint16_t>(std::strtoul(hostport.substr(colon + 1).c_str(), nullptr, 10));
    }
    return !host->empty() && !path->empty();
}

bool HttpPostJson(const std::string& host, uint16_t port, const std::string& path, const std::string& body) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) return false;
    int fd = -1;
    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) return false;
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n";
    req << "Host: " << host << "\r\n";
    req << "Content-Type: application/json\r\n";
    req << "Connection: close\r\n";
    req << "Content-Length: " << body.size() << "\r\n\r\n";
    req << body;
    const std::string s = req.str();
    const auto sent = ::send(fd, s.data(), s.size(), 0);
    char buf[256];
    const auto n = ::recv(fd, buf, sizeof(buf) - 1, 0);
    ::close(fd);
    if (sent < 0 || n <= 0) return false;
    buf[n] = '\0';
    const std::string resp(buf, static_cast<size_t>(n));
    return resp.find(" 200 ") != std::string::npos || resp.find(" 201 ") != std::string::npos || resp.find(" 202 ") != std::string::npos;
}
} // namespace

EdgeGateway::EdgeGateway(MembraneManager* manager) : manager_(manager) {}
EdgeGateway::~EdgeGateway() { (void)Stop(); }

Status EdgeGateway::Start(uint16_t http_port, uint16_t ingest_port, std::string auth_token) {
    if (!manager_) return Status::InvalidArgument("manager is null");
    if (running_.exchange(true)) return Status::AlreadyExists("edge gateway already running");
    auth_token_ = std::move(auth_token);
    const auto& opt = manager_->GetOptions();
    token_file_path_ = opt.gateway_token_file;
    if (opt.gateway_rate_limit_per_sec > 0) {
        rate_limit_per_sec_ = opt.gateway_rate_limit_per_sec;
    }
    if (opt.gateway_idempotency_ttl_sec > 0) {
        idempotency_ttl_sec_ = opt.gateway_idempotency_ttl_sec;
    }
    idempotency_log_path_ = manager_->GetOptions().path + "/gateway/idempotency.log";
    audit_log_path_ = manager_->GetOptions().path + "/gateway/audit.log";
    seq_state_path_ = manager_->GetOptions().path + "/gateway/ingest.seq";
    sync_checkpoint_path_ = manager_->GetOptions().path + "/gateway/sync.checkpoint";
    const auto load_st = LoadIdempotencyLog();
    if (!load_st.ok()) {
        running_.store(false);
        return load_st;
    }
    const auto seq_st = LoadSeqState();
    if (!seq_st.ok()) {
        running_.store(false);
        return seq_st;
    }
    ReloadTokensIfNeeded();
    http_thread_ = std::thread([this, http_port]() { HttpLoop(http_port); });
    ingest_thread_ = std::thread([this, ingest_port]() { IngestLoop(ingest_port); });
    if (opt.gateway_upstream_sync_enabled && !opt.gateway_upstream_sync_url.empty()) {
        sync_thread_ = std::thread([this]() { SyncLoop(); });
    }
    return Status::Ok();
}

Status EdgeGateway::Stop() {
    if (!running_.exchange(false)) return Status::Ok();
    const int hfd = http_fd_.exchange(-1);
    if (hfd >= 0) {
        ::shutdown(hfd, SHUT_RDWR);
        ::close(hfd);
    }
    const int ifd = ingest_fd_.exchange(-1);
    if (ifd >= 0) {
        ::shutdown(ifd, SHUT_RDWR);
        ::close(ifd);
    }
    if (http_thread_.joinable()) http_thread_.join();
    if (ingest_thread_.joinable()) ingest_thread_.join();
    sync_cv_.notify_all();
    if (sync_thread_.joinable()) sync_thread_.join();
    return Status::Ok();
}

bool EdgeGateway::IsAuthorized(const std::string& request) const {
    if (auth_token_.empty()) return true;
    const std::string expected = "Authorization: Bearer " + auth_token_;
    return request.find(expected) != std::string::npos;
}

void EdgeGateway::ReloadTokensIfNeeded() {
    if (token_file_path_.empty()) return;
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    if (now < token_last_reload_sec_ + token_reload_every_sec_) return;
    token_last_reload_sec_ = now;
    struct stat st{};
    if (::stat(token_file_path_.c_str(), &st) != 0) return;
    const uint64_t mtime = static_cast<uint64_t>(st.st_mtime);
    if (mtime == token_file_mtime_sec_) return;
    token_file_mtime_sec_ = mtime;

    std::ifstream in(token_file_path_);
    if (!in.good()) return;
    std::unordered_map<std::string, TokenRecord> tmp;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto p1 = line.find('|');
        const auto p2 = (p1 == std::string::npos) ? std::string::npos : line.find('|', p1 + 1);
        if (p1 == std::string::npos || p2 == std::string::npos) continue;
        const std::string token = line.substr(0, p1);
        const uint64_t exp = static_cast<uint64_t>(std::strtoull(line.substr(p1 + 1, p2 - p1 - 1).c_str(), nullptr, 10));
        const std::string scopes_raw = line.substr(p2 + 1);
        TokenRecord rec;
        rec.exp_unix = exp;
        std::istringstream s(scopes_raw);
        std::string sc;
        while (std::getline(s, sc, ',')) {
            if (!sc.empty()) rec.scopes.insert(sc);
        }
        if (!token.empty()) tmp[token] = std::move(rec);
    }
    std::lock_guard<std::mutex> lock(token_mu_);
    token_store_ = std::move(tmp);
}

bool EdgeGateway::TokenAllowsScope(const std::string& token, std::string_view scope) const {
    if (!auth_token_.empty() && token == auth_token_) return true;
    std::lock_guard<std::mutex> lock(token_mu_);
    const auto it = token_store_.find(token);
    if (it == token_store_.end()) return false;
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    if (it->second.exp_unix != 0 && now > it->second.exp_unix) return false;
    return it->second.scopes.count(std::string(scope)) > 0 || it->second.scopes.count("*") > 0;
}

bool EdgeGateway::IsAuthorizedForScope(const std::string& request, std::string_view scope) {
    if (auth_token_.empty() && token_file_path_.empty()) return true;
    ReloadTokensIfNeeded();
    const auto pos = request.find("Authorization: Bearer ");
    if (pos == std::string::npos) return false;
    const auto start = pos + 22;
    const auto end = request.find("\r\n", start);
    if (end == std::string::npos) return false;
    const std::string token = request.substr(start, end - start);
    return TokenAllowsScope(token, scope);
}

bool EdgeGateway::AllowRequest() {
    const uint64_t now_sec = static_cast<uint64_t>(std::time(nullptr));
    uint64_t prev = rate_window_sec_.load();
    if (prev != now_sec) {
        rate_window_sec_.store(now_sec);
        rate_window_count_.store(0);
    }
    const uint32_t c = rate_window_count_.fetch_add(1) + 1;
    return c <= rate_limit_per_sec_;
}

bool EdgeGateway::HasIdempotencyKey(const std::string& key) {
    if (key.empty()) return false;
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    std::lock_guard<std::mutex> lock(idem_mu_);
    for (auto it = seen_idempotency_keys_.begin(); it != seen_idempotency_keys_.end();) {
        if (it->second <= now) {
            it = seen_idempotency_keys_.erase(it);
        } else {
            ++it;
        }
    }
    return seen_idempotency_keys_.find(key) != seen_idempotency_keys_.end();
}

void EdgeGateway::RememberIdempotencyKey(const std::string& key) {
    if (key.empty()) return;
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    bool inserted = false;
    {
        std::lock_guard<std::mutex> lock(idem_mu_);
        const auto it = seen_idempotency_keys_.find(key);
        if (it == seen_idempotency_keys_.end()) {
            inserted = true;
        }
        seen_idempotency_keys_[key] = now + idempotency_ttl_sec_;
    }
    if (inserted) {
        AppendIdempotencyLog(key);
        (void)CompactIdempotencyLogIfNeeded();
    }
}

Status EdgeGateway::LoadIdempotencyLog() {
    namespace fs = std::filesystem;
    const fs::path p(idempotency_log_path_);
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    if (ec) return Status::IOError("failed to create gateway dir");

    std::ifstream in(idempotency_log_path_);
    if (!in.good()) return Status::Ok();
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    std::string line;
    std::lock_guard<std::mutex> lock(idem_mu_);
    while (std::getline(in, line)) {
        // Keep only sane keys to tolerate partial/corrupt lines.
        if (!line.empty() && line.size() <= 256 && line.find('\n') == std::string::npos && line.find('\r') == std::string::npos) {
            seen_idempotency_keys_[line] = now + idempotency_ttl_sec_;
        } else if (!line.empty()) {
            idempotency_log_errors_total_.fetch_add(1);
        }
    }
    return Status::Ok();
}

void EdgeGateway::AppendIdempotencyLog(const std::string& key) {
    if (idempotency_log_path_.empty()) return;
    std::ofstream out(idempotency_log_path_, std::ios::app);
    if (!out.good()) {
        idempotency_log_errors_total_.fetch_add(1);
        return;
    }
    out << key << "\n";
    out.flush();
    ++idempotency_append_count_;
}

Status EdgeGateway::CompactIdempotencyLogIfNeeded() {
    if (idempotency_append_count_ < idempotency_compact_every_) return Status::Ok();
    idempotency_append_count_ = 0;
    const std::string tmp_path = idempotency_log_path_ + ".tmp";
    std::ofstream out(tmp_path, std::ios::trunc);
    if (!out.good()) return Status::IOError("failed to compact idempotency log");
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    {
        std::lock_guard<std::mutex> lock(idem_mu_);
        for (auto it = seen_idempotency_keys_.begin(); it != seen_idempotency_keys_.end();) {
            if (it->second <= now) {
                it = seen_idempotency_keys_.erase(it);
                continue;
            }
            out << it->first << "\n";
            ++it;
        }
    }
    out.flush();
    out.close();
    std::error_code ec;
    std::filesystem::rename(tmp_path, idempotency_log_path_, ec);
    if (ec) {
        idempotency_log_errors_total_.fetch_add(1);
        return Status::IOError("failed to rotate idempotency log");
    }
    idempotency_compactions_total_.fetch_add(1);
    return Status::Ok();
}

Status EdgeGateway::RotateAuditLogIfNeeded() {
    if (audit_append_count_ < audit_rotate_every_) return Status::Ok();
    audit_append_count_ = 0;
    namespace fs = std::filesystem;
    const fs::path p(audit_log_path_);
    std::error_code ec;
    const fs::path rotated = p.string() + ".1";
    if (fs::exists(rotated, ec)) {
        fs::remove(rotated, ec);
    }
    fs::rename(p, rotated, ec);
    if (ec) {
        audit_log_errors_total_.fetch_add(1);
        return Status::IOError("failed to rotate audit log");
    }
    audit_log_rotations_total_.fetch_add(1);
    return Status::Ok();
}

void EdgeGateway::AppendAuditLog(std::string_view event, std::string_view detail) {
    if (audit_log_path_.empty()) return;
    std::ofstream out(audit_log_path_, std::ios::app);
    if (!out.good()) {
        audit_log_errors_total_.fetch_add(1);
        return;
    }
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    out << now << "|" << event << "|" << detail << "\n";
    out.flush();
    ++audit_append_count_;
    (void)RotateAuditLogIfNeeded();
}

Status EdgeGateway::LoadSeqState() {
    std::ifstream in(seq_state_path_);
    if (!in.good()) return Status::Ok();
    uint64_t v = 0;
    in >> v;
    ingest_seq_.store(v);
    return Status::Ok();
}

Status EdgeGateway::SaveSeqState() {
    std::ofstream out(seq_state_path_, std::ios::trunc);
    if (!out.good()) return Status::IOError("failed to save seq state");
    out << ingest_seq_.load();
    out.flush();
    return Status::Ok();
}

std::string EdgeGateway::JsonResponse(int code, std::string_view reason, std::string_view status, std::string_view message, std::string_view err_code) const {
    std::ostringstream body;
    body << "{\"status\":\"" << status << "\",\"code\":\"" << err_code << "\",\"message\":\"" << message << "\"}";
    const std::string body_s = body.str();
    std::ostringstream head;
    head << "HTTP/1.1 " << code << " " << reason << "\r\n";
    head << "Content-Type: application/json\r\n";
    head << "Content-Length: " << body_s.size() << "\r\n\r\n";
    return head.str() + body_s;
}

std::string EdgeGateway::HealthGradeJson() const {
    std::string grade = "healthy";
    std::string reason = "ok";
    if (audit_log_errors_total_.load() > 0 || idempotency_log_errors_total_.load() > 0) {
        grade = "degraded";
        reason = "log_io_errors";
    }
    if (http_errors_total_.load() + ingest_errors_total_.load() > (http_requests_total_.load() + ingest_requests_total_.load()) / 2 + 10) {
        grade = "unhealthy";
        reason = "high_error_ratio";
    }
    if (sync_fail_total_.load() > sync_success_total_.load() + 10) {
        grade = "degraded";
        reason = "upstream_sync_failures";
    }
    std::ostringstream body;
    body << "{\"grade\":\"" << grade << "\",\"reason\":\"" << reason << "\"}";
    return body.str();
}

void EdgeGateway::EnqueueSyncEvent(SyncEvent ev) {
    if (!manager_->GetOptions().gateway_upstream_sync_enabled || manager_->GetOptions().gateway_upstream_sync_url.empty()) return;
    std::lock_guard<std::mutex> lock(sync_mu_);
    if (sync_queue_.size() >= sync_max_queue_) {
        sync_queue_.pop_front();
        sync_backlog_drops_total_.fetch_add(1);
    }
    sync_queue_.push_back(std::move(ev));
    sync_cv_.notify_one();
}

void EdgeGateway::SyncLoop() {
    std::ifstream in(sync_checkpoint_path_);
    if (in.good()) {
        uint64_t ck = 0;
        in >> ck;
        sync_checkpoint_seq_.store(ck);
    }
    std::string host, path;
    uint16_t port = 80;
    if (!ParseHttpUrl(manager_->GetOptions().gateway_upstream_sync_url, &host, &port, &path)) {
        AppendAuditLog("sync_disabled", "invalid_upstream_url");
        return;
    }
    while (running_.load()) {
        SyncEvent ev;
        {
            std::unique_lock<std::mutex> lock(sync_mu_);
            sync_cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() { return !sync_queue_.empty() || !running_.load(); });
            if (!running_.load()) break;
            if (sync_queue_.empty()) continue;
            ev = sync_queue_.front();
        }
        if (ev.seq <= sync_checkpoint_seq_.load()) {
            std::lock_guard<std::mutex> lock(sync_mu_);
            if (!sync_queue_.empty() && sync_queue_.front().seq == ev.seq) sync_queue_.pop_front();
            continue;
        }
        std::ostringstream body;
        body << "{\"seq\":" << ev.seq << ",\"type\":\"" << ev.type << "\",\"membrane\":\"" << ev.membrane << "\",\"id\":" << ev.id << "}";
        sync_attempts_total_.fetch_add(1);
        const bool ok = HttpPostJson(host, port, path, body.str());
        if (ok) {
            sync_success_total_.fetch_add(1);
            sync_checkpoint_seq_.store(ev.seq);
            std::ofstream out(sync_checkpoint_path_, std::ios::trunc);
            if (out.good()) {
                out << ev.seq;
                out.flush();
            }
            std::lock_guard<std::mutex> lock(sync_mu_);
            if (!sync_queue_.empty() && sync_queue_.front().seq == ev.seq) sync_queue_.pop_front();
        } else {
            sync_fail_total_.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(sync_retry_ms_));
        }
    }
}

void EdgeGateway::HttpLoop(uint16_t port) {
    const int server_fd = OpenListenSocket(port);
    if (server_fd < 0) {
        running_.store(false);
        return;
    }
    http_fd_.store(server_fd);
    while (running_.load()) {
        int client = ::accept(server_fd, nullptr, nullptr);
        if (client < 0) continue;
        http_requests_total_.fetch_add(1);
        if (!AllowRequest()) {
            const std::string too_many = JsonResponse(429, "Too Many Requests", "error", "rate_limited");
            (void)::send(client, too_many.data(), too_many.size(), 0);
            ::close(client);
            http_errors_total_.fetch_add(1);
            rate_limited_total_.fetch_add(1);
            AppendAuditLog("http_rate_limited", "request");
            continue;
        }
        char buf[2048];
        const ssize_t n = ::recv(client, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            std::string req(buf, static_cast<size_t>(n));
            std::string response = JsonResponse(200, "OK", "ok", "healthy");
            const bool health = (req.rfind("GET /health ", 0) == 0) || (req.rfind("GET /v1/health ", 0) == 0);
            const bool metrics = (req.rfind("GET /metrics ", 0) == 0) || (req.rfind("GET /v1/metrics ", 0) == 0);
            const bool healthz = (req.rfind("GET /healthz ", 0) == 0) || (req.rfind("GET /v1/healthz ", 0) == 0);
            if (health) {
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
                continue;
            }
            if (healthz) {
                const std::string body = HealthGradeJson();
                const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
                const std::string full = head + body;
                (void)::send(client, full.data(), full.size(), 0);
                ::close(client);
                continue;
            }
            if (metrics) {
                if (!IsAuthorizedForScope(req, "admin:ops")) {
                    response = JsonResponse(401, "Unauthorized", "error", "missing admin scope", "auth_scope_denied");
                    auth_denied_total_.fetch_add(1);
                    (void)::send(client, response.data(), response.size(), 0);
                    ::close(client);
                    continue;
                }
                std::ostringstream oss;
                oss << "http_requests_total " << http_requests_total_.load() << "\n";
                oss << "http_errors_total " << http_errors_total_.load() << "\n";
                oss << "ingest_requests_total " << ingest_requests_total_.load() << "\n";
                oss << "ingest_errors_total " << ingest_errors_total_.load() << "\n";
                oss << "ingest_duplicates_total " << ingest_duplicates_total_.load() << "\n";
                oss << "rate_limited_total " << rate_limited_total_.load() << "\n";
                oss << "idempotency_compactions_total " << idempotency_compactions_total_.load() << "\n";
                oss << "idempotency_log_errors_total " << idempotency_log_errors_total_.load() << "\n";
                oss << "audit_log_errors_total " << audit_log_errors_total_.load() << "\n";
                oss << "audit_log_rotations_total " << audit_log_rotations_total_.load() << "\n";
                oss << "auth_denied_total " << auth_denied_total_.load() << "\n";
                oss << "mtls_denied_total " << mTLS_denied_total_.load() << "\n";
                oss << "ingest_seq " << ingest_seq_.load() << "\n";
                oss << "sync_attempts_total " << sync_attempts_total_.load() << "\n";
                oss << "sync_success_total " << sync_success_total_.load() << "\n";
                oss << "sync_fail_total " << sync_fail_total_.load() << "\n";
                oss << "sync_backlog_drops_total " << sync_backlog_drops_total_.load() << "\n";
                oss << "sync_checkpoint_seq " << sync_checkpoint_seq_.load() << "\n";
                const std::string body = oss.str();
                const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n";
                const std::string full = head + body;
                (void)::send(client, full.data(), full.size(), 0);
                ::close(client);
                continue;
            }
            if (manager_->GetOptions().gateway_require_mtls_proxy_header &&
                req.find(manager_->GetOptions().gateway_mtls_proxy_header) == std::string::npos) {
                response = JsonResponse(401, "Unauthorized", "error", "mtls proxy header missing", "mtls_required");
                mTLS_denied_total_.fetch_add(1);
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
                continue;
            }
            if (!IsAuthorizedForScope(req, "ingest:write")) {
                response = JsonResponse(401, "Unauthorized", "error", "missing ingest scope", "auth_scope_denied");
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
                http_errors_total_.fetch_add(1);
                auth_denied_total_.fetch_add(1);
                AppendAuditLog("http_unauthorized", "request");
                continue;
            }
            const auto key_pos = req.find("X-Idempotency-Key:");
            std::string idempotency_key;
            if (key_pos != std::string::npos) {
                const auto eol = req.find("\r\n", key_pos);
                if (eol != std::string::npos) {
                    idempotency_key = req.substr(key_pos + 18, eol - (key_pos + 18));
                    while (!idempotency_key.empty() && (idempotency_key.front() == ' ' || idempotency_key.front() == '\t')) {
                        idempotency_key.erase(idempotency_key.begin());
                    }
                }
            }
            if (HasIdempotencyKey(idempotency_key)) {
                response = JsonResponse(200, "OK", "duplicate", "idempotency_key_seen");
                (void)::send(client, response.data(), response.size(), 0);
                ::close(client);
                ingest_duplicates_total_.fetch_add(1);
                AppendAuditLog("http_duplicate", idempotency_key);
                continue;
            }
            // Tiny endpoints:
            // - GET /health
            // - POST /ingest/meta/<membrane>/<gid> body=value
            // - POST /ingest/vector/<membrane>/<id> body=f1,f2,f3
            const bool meta_ingest = (req.rfind("POST /ingest/meta/", 0) == 0) || (req.rfind("POST /v1/ingest/meta/", 0) == 0);
            const bool vec_ingest = (req.rfind("POST /ingest/vector/", 0) == 0) || (req.rfind("POST /v1/ingest/vector/", 0) == 0);
            if (meta_ingest) {
                const auto sp1 = req.find(' ');
                const auto sp2 = req.find(' ', sp1 + 1);
                const std::string path = req.substr(sp1 + 1, sp2 - (sp1 + 1));
                const auto body_pos = req.find("\r\n\r\n");
                const std::string body = (body_pos == std::string::npos) ? std::string() : req.substr(body_pos + 4);
                const std::string prefix = (path.rfind("/v1/ingest/meta/", 0) == 0) ? "/v1/ingest/meta/" : "/ingest/meta/";
                const auto rem = path.substr(prefix.size());
                const auto slash = rem.find('/');
                if (slash != std::string::npos) {
                    const auto membrane = rem.substr(0, slash);
                    const auto gid = rem.substr(slash + 1);
                    const auto st = manager_->MetaPut(membrane, gid, body);
                    if (st.ok()) {
                        RememberIdempotencyKey(idempotency_key);
                        AppendAuditLog("http_meta_ok", std::string(membrane) + "/" + std::string(gid));
                        response = JsonResponse(200, "OK", "ok", "meta_ingested", "accepted");
                    } else {
                        AppendAuditLog("http_meta_err", st.message());
                        response = JsonResponse(400, "Bad Request", "error", st.message(), "write_failed");
                        http_errors_total_.fetch_add(1);
                    }
                } else {
                    AppendAuditLog("http_meta_err", "invalid_meta_path");
                    response = JsonResponse(400, "Bad Request", "error", "invalid_meta_path", "invalid_path");
                    http_errors_total_.fetch_add(1);
                }
            } else if (vec_ingest) {
                const auto sp1 = req.find(' ');
                const auto sp2 = req.find(' ', sp1 + 1);
                const std::string path = req.substr(sp1 + 1, sp2 - (sp1 + 1));
                const auto body_pos = req.find("\r\n\r\n");
                const std::string body = (body_pos == std::string::npos) ? std::string() : req.substr(body_pos + 4);
                const std::string prefix = (path.rfind("/v1/ingest/vector/", 0) == 0) ? "/v1/ingest/vector/" : "/ingest/vector/";
                const auto rem = path.substr(prefix.size());
                const auto slash = rem.find('/');
                if (slash != std::string::npos) {
                    const auto membrane = rem.substr(0, slash);
                    const auto id_s = rem.substr(slash + 1);
                    const uint64_t id = static_cast<uint64_t>(std::strtoull(id_s.c_str(), nullptr, 10));
                    std::vector<float> v;
                    std::istringstream viss(body);
                    std::string tok;
                    while (std::getline(viss, tok, ',')) {
                        v.push_back(static_cast<float>(std::strtod(tok.c_str(), nullptr)));
                    }
                    if (!v.empty()) {
                        const auto st = manager_->PutVector(membrane, id, v);
                        if (st.ok()) {
                            RememberIdempotencyKey(idempotency_key);
                            AppendAuditLog("http_vector_ok", std::string(membrane) + "/" + std::to_string(id));
                            response = JsonResponse(200, "OK", "ok", "vector_ingested", "accepted");
                        } else {
                            AppendAuditLog("http_vector_err", st.message());
                            response = JsonResponse(400, "Bad Request", "error", st.message(), "write_failed");
                            http_errors_total_.fetch_add(1);
                        }
                    } else {
                        AppendAuditLog("http_vector_err", "empty_vector");
                        response = JsonResponse(400, "Bad Request", "error", "empty_vector", "bad_vector");
                        http_errors_total_.fetch_add(1);
                    }
                } else {
                    AppendAuditLog("http_vector_err", "invalid_vector_path");
                    response = JsonResponse(400, "Bad Request", "error", "invalid_vector_path", "invalid_path");
                    http_errors_total_.fetch_add(1);
                }
            } else {
                AppendAuditLog("http_not_found", "endpoint");
                response = JsonResponse(404, "Not Found", "error", "endpoint_not_found", "not_found");
                http_errors_total_.fetch_add(1);
            }
            (void)::send(client, response.data(), response.size(), 0);
        }
        ::close(client);
    }
    if (http_fd_.exchange(-1) >= 0) ::close(server_fd);
}

void EdgeGateway::IngestLoop(uint16_t port) {
    const int server_fd = OpenListenSocket(port);
    if (server_fd < 0) {
        running_.store(false);
        return;
    }
    ingest_fd_.store(server_fd);
    while (running_.load()) {
        int client = ::accept(server_fd, nullptr, nullptr);
        if (client < 0) continue;
        ingest_requests_total_.fetch_add(1);
        if (!AllowRequest()) {
            static const std::string nack = "ERR|rate_limited\n";
            (void)::send(client, nack.data(), nack.size(), 0);
            ::close(client);
            ingest_errors_total_.fetch_add(1);
            rate_limited_total_.fetch_add(1);
            AppendAuditLog("ingest_rate_limited", "request");
            continue;
        }
        char buf[4096];
        const ssize_t n = ::recv(client, buf, sizeof(buf) - 1, 0);
        if (n > 0) {
            buf[n] = '\0';
            // Simple line protocol for MQTT/WebSocket gateway compatibility:
            // No-auth: MQTT|<membrane>|<id>|f1,f2,f3
            // Auth:    MQTT|<token>|<membrane>|<id>|f1,f2,f3
            // Same shape for WS|...
            std::string line(buf, static_cast<size_t>(n));
            std::istringstream iss(line);
            std::string proto;
            if (!std::getline(iss, proto, '|')) {
                ::close(client);
                ingest_errors_total_.fetch_add(1);
                continue;
            }
            std::string token_or_membrane;
            std::string membrane;
            std::string id_s;
            std::string vec_s;
            if (!std::getline(iss, token_or_membrane, '|')) {
                ::close(client);
                continue;
            }
            if (auth_token_.empty()) {
                membrane = token_or_membrane;
                if (!std::getline(iss, id_s, '|') || !std::getline(iss, vec_s, '|')) {
                    ::close(client);
                    ingest_errors_total_.fetch_add(1);
                    continue;
                }
            } else {
                if (token_or_membrane != auth_token_) {
                    ::close(client);
                    ingest_errors_total_.fetch_add(1);
                    continue;
                }
                if (!std::getline(iss, membrane, '|') || !std::getline(iss, id_s, '|') || !std::getline(iss, vec_s, '|')) {
                    ::close(client);
                    ingest_errors_total_.fetch_add(1);
                    continue;
                }
            }
            if ((proto == "MQTT" || proto == "WS") && !membrane.empty() && !id_s.empty()) {
                const uint64_t id = static_cast<uint64_t>(std::strtoull(id_s.c_str(), nullptr, 10));
                std::vector<float> v;
                std::istringstream viss(vec_s);
                std::string tok;
                while (std::getline(viss, tok, ',')) {
                    v.push_back(static_cast<float>(std::strtod(tok.c_str(), nullptr)));
                }
                if (!v.empty()) {
                    // Optional fields:
                    // 5th token: idempotency key
                    // 6th token: durability ("D")
                    std::string idem_key;
                    std::string durability;
                    (void)std::getline(iss, idem_key, '|');
                    (void)std::getline(iss, durability, '|');
                    if (HasIdempotencyKey(idem_key)) {
                        const std::string reply = "ACK|duplicate\n";
                        ingest_duplicates_total_.fetch_add(1);
                        AppendAuditLog("ingest_duplicate", idem_key);
                        (void)::send(client, reply.data(), reply.size(), 0);
                        ::close(client);
                        continue;
                    }
                    auto st = manager_->PutVector(membrane, id, v);
                    if (st.ok() && durability == "D") {
                        st = manager_->FlushAll();
                    }
                    uint64_t seq = 0;
                    if (st.ok()) {
                        seq = ingest_seq_.fetch_add(1) + 1;
                        (void)SaveSeqState();
                    }
                    const std::string reply = st.ok()
                        ? (durability == "D" ? "ACK|durable_ack|seq=" + std::to_string(seq) + "\n"
                                             : "ACK|accepted|seq=" + std::to_string(seq) + "\n")
                        : "ERR|write_failed\n";
                    if (!st.ok()) {
                        ingest_errors_total_.fetch_add(1);
                        AppendAuditLog("ingest_err", st.message());
                    } else {
                        RememberIdempotencyKey(idem_key);
                        AppendAuditLog("ingest_ok", std::string(membrane) + "/" + std::to_string(id));
                        EnqueueSyncEvent(SyncEvent{seq, "vector_put", membrane, id});
                    }
                    (void)::send(client, reply.data(), reply.size(), 0);
                } else {
                    const std::string reply = "ERR|bad_vector\n";
                    ingest_errors_total_.fetch_add(1);
                    (void)::send(client, reply.data(), reply.size(), 0);
                }
            } else {
                const std::string reply = "ERR|bad_protocol\n";
                ingest_errors_total_.fetch_add(1);
                (void)::send(client, reply.data(), reply.size(), 0);
            }
        }
        ::close(client);
    }
    if (ingest_fd_.exchange(-1) >= 0) ::close(server_fd);
}

} // namespace pomai::core
