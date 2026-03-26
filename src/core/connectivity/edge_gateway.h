#pragma once

#include <cstdint>
#include <deque>
#include <set>
#include <string>
#include <unordered_map>
#include <poll.h>
#include <fcntl.h>

#include "pomai/status.h"

namespace pomai::core {
class MembraneManager;

class EdgeGateway {
public:
    explicit EdgeGateway(MembraneManager* manager);
    ~EdgeGateway();

    Status Start(uint16_t http_port, uint16_t ingest_port, std::string auth_token = {});
    Status Stop();
    bool Running() const { return running_; }

    /**
     * @brief Process pending network I/O and internal state updates.
     * Must be called periodically from the main loop.
     */
    void Tick();

private:
    struct TokenRecord {
        uint64_t exp_unix = 0;
        std::set<std::string> scopes;
    };
    struct SyncEvent {
        uint64_t seq = 0;
        std::string type;
        std::string membrane;
        uint64_t id = 0;
        uint64_t id2 = 0;
        uint32_t u32_a = 0;
        uint32_t u32_b = 0;
        std::string aux_k;
        std::string aux_v;
    };

    bool IsAuthorized(const std::string& request) const;
    bool IsAuthorizedForScope(const std::string& request, std::string_view scope);
    bool TokenAllowsScope(const std::string& token, std::string_view scope) const;
    void ReloadTokensIfNeeded();
    bool AllowRequest();
    bool HasIdempotencyKey(const std::string& key);
    void RememberIdempotencyKey(const std::string& key);
    Status LoadIdempotencyLog();
    void AppendIdempotencyLog(const std::string& key);
    Status CompactIdempotencyLogIfNeeded();
    Status RotateAuditLogIfNeeded();
    Status LoadSeqState();
    Status SaveSeqState();
    void AppendAuditLog(std::string_view event, std::string_view detail);
    std::string JsonResponse(int code, std::string_view reason, std::string_view status, std::string_view message, std::string_view err_code = "ok") const;
    std::string HealthGradeJson() const;

    // Single-threaded non-blocking processing
    void ProcessHttp();
    void ProcessIngest();
    void ProcessSync();
    void ProcessAcceptedClient(int client);

    void EnqueueSyncEvent(SyncEvent ev);
    void ServeHttpConnection(int client, const std::string& req);
    void ServeIngestConnection(int client, const std::string& line);

    MembraneManager* manager_ = nullptr;
    bool running_ = false;
    bool unified_mode_ = false;
    std::string auth_token_;
    std::string token_file_path_;
    int http_fd_ = -1;
    int ingest_fd_ = -1;
    uint64_t http_requests_total_ = 0;
    uint64_t http_errors_total_ = 0;
    uint64_t ingest_requests_total_ = 0;
    uint64_t ingest_errors_total_ = 0;
    uint64_t ingest_duplicates_total_ = 0;
    uint64_t rate_limited_total_ = 0;
    uint64_t idempotency_compactions_total_ = 0;
    uint64_t idempotency_log_errors_total_ = 0;
    uint64_t audit_log_errors_total_ = 0;
    uint64_t audit_log_rotations_total_ = 0;
    uint64_t auth_denied_total_ = 0;
    uint64_t mTLS_denied_total_ = 0;
    uint64_t rate_window_sec_ = 0;
    uint32_t rate_window_count_ = 0;
    uint32_t rate_limit_per_sec_ = 2000;
    std::string idempotency_log_path_;
    std::string audit_log_path_;
    std::string seq_state_path_;
    uint64_t idempotency_append_count_ = 0;
    uint64_t idempotency_compact_every_ = 1000;
    uint64_t audit_append_count_ = 0;
    uint64_t audit_rotate_every_ = 5000;
    std::unordered_map<std::string, uint64_t> seen_idempotency_keys_;
    std::unordered_map<std::string, TokenRecord> token_store_;
    uint64_t token_file_mtime_sec_ = 0;
    uint64_t token_reload_every_sec_ = 5;
    uint64_t token_last_reload_sec_ = 0;
    uint64_t ingest_seq_ = 0;
    uint64_t sync_attempts_total_ = 0;
    uint64_t sync_success_total_ = 0;
    uint64_t sync_fail_total_ = 0;
    uint64_t sync_backlog_drops_total_ = 0;
    uint64_t sync_retry_ms_ = 1000;
    uint64_t sync_max_queue_ = 10000;
    std::deque<SyncEvent> sync_queue_;
    std::string sync_checkpoint_path_;
    uint64_t sync_checkpoint_seq_ = 0;
    uint64_t idempotency_ttl_sec_ = 300;
    uint64_t last_sync_attempt_unix_ms_ = 0;
};

} // namespace pomai::core
