#!/usr/bin/env python3
"""
benchmarks/benchmark_vs.py

Comprehensive benchmark runner for Pomai server (Raw vs Synapse modes).

Features:
- Start/stop server binary with environment overrides.
- Insert (VSET), Get (VGET), Delete (VDEL), and Search (VSEARCH) benchmarks.
- Supports concurrent search clients to measure throughput & latency distributions.
- Collects process memory/cpu (via psutil) and writes CSV summary.
- Configurable via CLI args and environment variables.

Usage examples:
  # Run default benchmark (1M vectors, dim 512) for both Raw and Synapse modes
  python3 benchmarks/benchmark_vs.py --vectors 1000000 --dim 512

  # Only run Synapse mode with 100k vectors
  python3 benchmarks/benchmark_vs.py --mode syn --vectors 100000 --dim 128

Notes:
- Requires psutil (pip install psutil) for memory/cpu stats.
- Ensure SERVER_BIN points to built server binary (default ./build/pomai-server).
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except Exception:
    psutil = None

# ---------------- Default configuration ----------------
DEFAULT_HOST = os.environ.get("POMAI_BENCH_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("POMAI_BENCH_PORT", "7777"))
DEFAULT_SERVER_BIN = os.environ.get("POMAI_SERVER_BIN", "./build/pomai-server")

# ---------------- RESP helpers ----------------
def build_resp_command(parts: List[bytes]) -> bytes:
    # parts: list of byte strings (each is an argument)
    out = bytearray()
    out += b"*" + str(len(parts)).encode() + b"\r\n"
    for p in parts:
        out += b"$" + str(len(p)).encode() + b"\r\n"
        out += p + b"\r\n"
    return bytes(out)

# ---------------- Simple TCP client ----------------
class PomaiClient:
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.lock = threading.Lock()

    def connect(self, retries: int = 10, wait: float = 0.5) -> bool:
        for _ in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((self.host, self.port))
                s.settimeout(self.timeout)
                self.sock = s
                return True
            except Exception:
                time.sleep(wait)
        return False

    def close(self) -> None:
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def send(self, data: bytes) -> None:
        if not self.sock:
            raise RuntimeError("socket not connected")
        mv = memoryview(data)
        sent = 0
        while sent < len(data):
            n = self.sock.send(mv[sent:])
            if n == 0:
                raise RuntimeError("socket send returned 0")
            sent += n

    def recv_some(self, maxbytes: int = 65536) -> bytes:
        if not self.sock:
            return b""
        try:
            return self.sock.recv(maxbytes) or b""
        except socket.timeout:
            return b""
        except Exception:
            return b""

    def send_and_recv(self, data: bytes, expect_lines: int = 1, timeout: float = 5.0) -> bytes:
        # thread-safe send+recv for simple usages
        with self.lock:
            self.send(data)
            # collect until we see at least expect_lines newlines or timeout
            deadline = time.time() + timeout
            buf = bytearray()
            while time.time() < deadline and buf.count(b"\n") < expect_lines:
                chunk = self.recv_some(65536)
                if chunk:
                    buf.extend(chunk)
                else:
                    time.sleep(0.001)
            return bytes(buf)

# ---------------- Process utils ----------------
def start_server(bin_path: str, env: Dict[str, str], stdout_path: str = "pomai_server_stdout.log",
                 stderr_path: str = "pomai_server_stderr.log") -> subprocess.Popen:
    env_copy = os.environ.copy()
    env_copy.update(env)
    # ensure port env variable if provided
    stdout_f = open(stdout_path, "ab")
    stderr_f = open(stderr_path, "ab")
    proc = subprocess.Popen([bin_path], env=env_copy, stdout=stdout_f, stderr=stderr_f)
    return proc

def stop_server(proc: subprocess.Popen, timeout: float = 3.0) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

def get_process_stats(pid: int) -> Tuple[float, float]:
    if psutil is None:
        return (0.0, 0.0)
    try:
        p = psutil.Process(pid)
        mem_mb = p.memory_info().rss / (1024.0 * 1024.0)
        # cpu_percent requires a prior interval; use a small sample
        cpu_pct = p.cpu_percent(interval=0.1)
        return (mem_mb, cpu_pct)
    except Exception:
        return (0.0, 0.0)

def vset_batch(client: PomaiClient, start_idx: int, count: int, dim: int, key_prefix: str = "key",
               max_retries: int = 3) -> None:
    """
    Send count VSET commands in a single TCP stream. On BrokenPipe/socket error,
    try to reconnect and resend the batch up to max_retries times.
    """
    parts = []
    for i in range(start_idx, start_idx + count):
        key = f"{key_prefix}:{i}".encode()
        vec = [random.random() for _ in range(dim)]
        vec_bytes = struct.pack(f"{dim}f", *vec)
        cmd = build_resp_command([b"VSET", key, vec_bytes])
        parts.append(cmd)
    data = b"".join(parts)

    retries = 0
    while True:
        try:
            client.send(data)
            # read back responses (one line per VSET). We'll read until we've got 'count' lines or timeout.
            deadline = time.time() + max(10.0, 0.05 * count)
            lines = 0
            buf = bytearray()
            while time.time() < deadline and lines < count:
                chunk = client.recv_some(65536)
                if not chunk:
                    time.sleep(0.001)
                    continue
                buf.extend(chunk)
                lines = buf.count(b"\n")
            return
        except BrokenPipeError:
            retries += 1
            if retries > max_retries:
                raise
            # attempt reconnect
            client.close()
            time.sleep(0.2)
            if not client.connect(retries=5, wait=0.2):
                raise RuntimeError("vset_batch: failed to reconnect to server after BrokenPipe")
            # on reconnect, retry sending the batch
        except OSError as e:
            # Generic socket error: try reconnect similarly
            retries += 1
            if retries > max_retries:
                raise
            client.close()
            time.sleep(0.2)
            if not client.connect(retries=5, wait=0.2):
                raise RuntimeError(f"vset_batch: socket error and reconnect failed: {e}")


def insert_vectors_streaming(client: PomaiClient, total: int, dim: int, batch_size: int,
                             key_prefix: str = "key") -> Tuple[float, List[float]]:
    """
    Insert `total` vectors via VSET in batches. On batch failure we attempt reconnect and retry
    each batch up to a few times. Returns overall QPS (based on successfully sent vectors) and per-batch durations.
    """
    batch_times = []
    t_start = time.time()
    sent = 0
    while sent < total:
        cur = min(batch_size, total - sent)
        t0 = time.time()
        try:
            vset_batch(client, sent, cur, dim, key_prefix)
        except Exception as e:
            # If server died, propagate to caller after printing helpful diagnostics.
            print(f"[bench] Error sending batch starting at {sent}: {e!r}")
            print("[bench] Check server logs: pomai_server_stdout.log and pomai_server_stderr.log")
            # compute partial QPS from what we sent successfully so far
            t_now = time.time()
            elapsed = max(1e-6, t_now - t_start)
            qps = sent / elapsed
            return qps, batch_times
        t1 = time.time()
        batch_times.append(t1 - t0)
        sent += cur
    t_end = time.time()
    total_time = max(1e-6, t_end - t_start)
    qps = total / total_time
    return qps, batch_times

def vget_one(client: PomaiClient, label_or_key: str, expect_bytes: Optional[int] = None) -> Tuple[float, bool]:
    cmd = build_resp_command([b"VGET", label_or_key.encode()])
    t0 = time.time()
    client.send_and_recv(cmd, expect_lines=1, timeout=2.0)
    t1 = time.time()
    return (t1 - t0) * 1000.0, True

def vdel_one(client: PomaiClient, label_or_key: str) -> Tuple[float, bool]:
    cmd = build_resp_command([b"VDEL", label_or_key.encode()])
    t0 = time.time()
    client.send_and_recv(cmd, expect_lines=1, timeout=2.0)
    t1 = time.time()
    return (t1 - t0) * 1000.0, True

def vsearch_one(client: PomaiClient, dim: int, topk: int) -> Tuple[float, bool]:
    query = struct.pack(f"{dim}f", *[random.random() for _ in range(dim)])
    cmd = build_resp_command([b"VSEARCH", query, str(topk).encode()])
    t0 = time.time()
    resp = client.send_and_recv(cmd, expect_lines=1, timeout=5.0)
    t1 = time.time()
    # crude check: response should start with '*' for array
    ok = resp.startswith(b"*")
    return (t1 - t0) * 1000.0, ok

def concurrent_search_benchmark(host: str, port: int, dim: int, topk: int, iters: int, concurrency: int) -> Dict:
    """
    Run `iters` searches distributed among `concurrency` worker threads; measure latencies and throughput.
    Returns dict with latencies list and total_ops/sec.
    """
    latencies: List[float] = []
    successes = 0
    lock = threading.Lock()

    def worker(run_count: int):
        nonlocal successes
        c = PomaiClient(host, port)
        if not c.connect(retries=3):
            return
        for _ in range(run_count):
            try:
                lat_ms, ok = vsearch_one(c, dim, topk)
                with lock:
                    latencies.append(lat_ms)
                    if ok:
                        successes += 1
            except Exception:
                with lock:
                    latencies.append(9999.0)
        c.close()

    per_thread = max(1, iters // concurrency)
    threads = []
    t0 = time.time()
    for i in range(concurrency):
        t = threading.Thread(target=worker, args=(per_thread,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    t1 = time.time()
    duration = max(1e-6, t1 - t0)
    total_ops = len(latencies)
    qps = total_ops / duration
    return {"latencies_ms": latencies, "qps": qps, "successes": successes}

# ---------------- Reporting ----------------
def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {}
    sorted_l = sorted(latencies)
    n = len(sorted_l)
    def pct(p):
        idx = min(n - 1, max(0, int(p * n) - 1))
        return sorted_l[idx]
    return {
        "count": n,
        "mean_ms": statistics.mean(sorted_l),
        "median_ms": statistics.median(sorted_l),
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "max_ms": max(sorted_l),
    }

def save_csv_summary(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved CSV summary to {path}")

# ---------------- Orchestrate a test case (mode) ----------------
def run_mode(mode: str, server_bin: str, port: int, vectors: int, dim: int, batch: int,
             search_iters: int, search_concurrency: int, topk: int,
             arena_mb: Optional[int]) -> Dict:
    """
    mode: "raw" or "syn"
    """
    print(f"\n=== Mode: {mode} ===")
    env = {"POMAI_PORT": str(port)}
    if arena_mb:
        env["POMAI_ARENA_MB"] = str(arena_mb)
    if mode == "raw":
        env["POMAI_DISABLE_SYNAPSE"] = "1"
    else:
        env.pop("POMAI_DISABLE_SYNAPSE", None)
    # Start server
    proc = start_server(server_bin, env)
    time.sleep(2.0)  # wait for server to initialize
    if proc.poll() is not None:
        print("Server process exited unexpectedly; check logs.")
        return {}

    client = PomaiClient(DEFAULT_HOST, port)
    if not client.connect(retries=10):
        print("Failed to connect client to server")
        stop_server(proc)
        return {}

    # Warmup / ensure server ready
    # send a small VSET
    try:
        key = b"__bench_init__"
        vec = struct.pack(f"{dim}f", *([0.0] * dim))
        client.send(build_resp_command([b"VSET", key, vec]))
        _ = client.recv_some(4096)
    except Exception:
        pass

    # 1) Insert benchmark
    t_ins_start = time.time()
    qps_ins, batch_times = insert_vectors_streaming(client, vectors, dim, batch)
    t_ins_end = time.time()
    mem_mb_before, cpu_before = get_process_stats(proc.pid)
    print(f"Insert done: qps_insert={qps_ins:.2f}, elapsed={t_ins_end - t_ins_start:.2f}s")

    # 2) Search benchmark (concurrent)
    search_res = concurrent_search_benchmark(DEFAULT_HOST, port, dim, topk, search_iters, search_concurrency)
    lat_summary = summarize_latencies(search_res["latencies_ms"])
    print(f"Search: qps_search={search_res['qps']:.2f}, mean_lat={lat_summary.get('mean_ms',0):.2f}ms p95={lat_summary.get('p95_ms',0):.2f}ms")

    # 3) VGET spot-check (10 random)
    get_lats = []
    for _ in range(10):
        idx = random.randrange(0, vectors)
        key = f"key:{idx}"
        lm, ok = vget_one(client, key)
        get_lats.append(lm)

    # 4) VDEL spot-check (10 random)
    del_lats = []
    for _ in range(10):
        idx = random.randrange(0, vectors)
        key = f"key:{idx}"
        lm, ok = vdel_one(client, key)
        del_lats.append(lm)

    mem_mb_after, cpu_after = get_process_stats(proc.pid)

    client.close()
    stop_server(proc)

    result = {
        "mode": mode,
        "insert_qps": qps_ins,
        "insert_time_s": (t_ins_end - t_ins_start),
        "search_qps": search_res["qps"],
        "search_mean_ms": lat_summary.get("mean_ms", 0.0),
        "search_p95_ms": lat_summary.get("p95_ms", 0.0),
        "get_mean_ms": statistics.mean(get_lats) if get_lats else 0.0,
        "del_mean_ms": statistics.mean(del_lats) if del_lats else 0.0,
        "mem_mb_before": mem_mb_before,
        "mem_mb_after": mem_mb_after,
        "cpu_before": cpu_before,
        "cpu_after": cpu_after,
    }
    return result

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Pomai benchmark runner (Raw vs Synapse)")
    p.add_argument("--server-bin", default=DEFAULT_SERVER_BIN, help="Path to pomai-server binary")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--mode", choices=["both", "raw", "syn"], default="both")
    p.add_argument("--vectors", type=int, default=100000, help="Number of vectors to insert for the test")
    p.add_argument("--dim", type=int, default=128, help="Vector dimensionality")
    p.add_argument("--batch", type=int, default=200, help="Insert batch size")
    p.add_argument("--search-iters", type=int, default=1000, help="Total search operations for benchmark")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrent search clients")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--arena-mb-raw", type=int, default=4096, help="Arena MB for raw mode")
    p.add_argument("--arena-mb-syn", type=int, default=1024, help="Arena MB for syn mode")
    p.add_argument("--csv-out", default="pomai_benchmark_summary.csv")
    return p.parse_args()

def main():
    args = parse_args()
    global DEFAULT_HOST, DEFAULT_PORT
    DEFAULT_HOST = args.host
    DEFAULT_PORT = args.port

    modes = []
    if args.mode == "both":
        modes = ["raw", "syn"]
    else:
        modes = [args.mode]

    results = []
    for m in modes:
        arena_mb = args.arena_mb_raw if m == "raw" else args.arena_mb_syn
        res = run_mode(m, args.server_bin, args.port, args.vectors, args.dim, args.batch,
                       args.search_iters, args.concurrency, args.topk, arena_mb)
        if res:
            results.append(res)

    # Save CSV
    rows = []
    for r in results:
        row = {
            "mode": r.get("mode", ""),
            "insert_qps": r.get("insert_qps", 0.0),
            "insert_time_s": r.get("insert_time_s", 0.0),
            "search_qps": r.get("search_qps", 0.0),
            "search_mean_ms": r.get("search_mean_ms", 0.0),
            "search_p95_ms": r.get("search_p95_ms", 0.0),
            "get_mean_ms": r.get("get_mean_ms", 0.0),
            "del_mean_ms": r.get("del_mean_ms", 0.0),
            "mem_mb_before": r.get("mem_mb_before", 0.0),
            "mem_mb_after": r.get("mem_mb_after", 0.0),
            "cpu_before": r.get("cpu_before", 0.0),
            "cpu_after": r.get("cpu_after", 0.0),
        }
        rows.append(row)
    save_csv_summary(rows, args.csv_out)
    print("Done.")

if __name__ == "__main__":
    main()