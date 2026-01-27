Concrete high-priority fixes (P0) — implement these first

    Make durability semantics explicit:
        Option A: Offer an immediate synchronous fdatasync path in AppendUpserts when wait_durable==true (do fdatasync while holding mutex or with careful ordering).
        Option B: Document that writes may be lost until WaitDurable returns, and recommend wait_durable=true for critical writes.
    Ensure truncation is durable:
        After ftruncate(wal), call fdatasync(wal_fd) and fsync the directory (open(dir) + fsync(dirfd)).
    Add graceful shutdown handler:
        Trap SIGINT/SIGTERM in main to call server.Stop() so Ctrl+C attempts to flush WAL before exit.
    Add unit + integration tests for WAL replay:
        Good records, truncated record at EOF, bad CRC, kill-9 during append, restart and assert recovery and truncation behavior.
    Add a small ReplayStats return value from Wal::ReplayToSeed (records_applied, vectors_applied, last_lsn, truncated_bytes) and log it from Shard::Start.

Important operational items (P1)

    Snapshot + WAL rotation:
        Implement periodic snapshot (persist a snapshot of seed store), perform fsync of snapshot file, then rotate wal: rename wal to wal.N or start new wal, fsync directory, and delete old wal safely.
    Metrics & telemetry:
        Expose metrics (written_lsn, durable_lsn, replayed_records, wal_size, queue sizes) via a /metrics HTTP endpoint or Prometheus client.
    Health & status API:
        Add admin endpoints returning last LSN, durable LSN, memtable size, index build backlog.
    Add logging improvements:
        Structured logs with severity, timestamp, component and include ReplayStats and truncation offsets.

Resilience & scaling (P2)

    Add HA/replication (async/sync or integrate Raft) if you need no-single-point-of-failure.
    Add tests under heavy load, run benchmarks with different fsync strategies, measure tail latencies.

Security & deployment (P2)

    TLS+auth for server RPCs, secure default unix socket perms, file ownership, sensitive data policy.
    Container builds, systemd unit, resource limits and liveness/readiness probes.
