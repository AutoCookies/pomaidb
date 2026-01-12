#!/usr/bin/env python3
"""
benchmarks/benchmark_vs.py (Optimized V2)

Robust benchmark comparing Raw Float (RAM) vs Synapse (4-bit) storage modes.

OPTIMIZATIONS V2:
- Uses a Data Pool to avoid calling random() 500 million times in Python.
- Streams batches instead of pre-allocating huge lists (saves Client RAM).
"""

from __future__ import annotations

import socket
import struct
import time
import random
import os
import sys
import subprocess
import psutil
import statistics
import csv
import argparse
from typing import List, Tuple, Dict

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except Exception:
    MATPLOTLIB = False

# ---------------- Config ----------------
HOST = os.environ.get("POMAI_BENCH_HOST", "127.0.0.1")
PORT = int(os.environ.get("POMAI_BENCH_PORT", "7777"))
SERVER_BIN = os.environ.get("POMAI_SERVER_BIN", "./build/pomai-server")

NUM_VECTORS = int(os.environ.get("POMAI_NUM_VECTORS", "1000000"))
DIMENSIONS = int(os.environ.get("POMAI_DIM", "512"))
BATCH_SIZE = int(os.environ.get("POMAI_BATCH", "200"))
SEARCH_TOP_K = int(os.environ.get("POMAI_TOPK", "10"))
SEARCH_ITERS = int(os.environ.get("POMAI_SEARCH_ITERS", "200"))

SERVER_STARTUP_WAIT = 2.0
BATCH_SEND_TIMEOUT = 10.0

# ---------------- Utility helpers ----------------

def get_process_stats(pid: int) -> Tuple[float, float]:
    try:
        p = psutil.Process(pid)
        mem_mb = p.memory_info().rss / (1024.0 * 1024.0)
        cpu = p.cpu_percent(interval=0.1)
        return mem_mb, cpu
    except Exception:
        return 0.0, 0.0

# ---------------- Robust client ----------------

class PomaiClient:
    def __init__(self, host: str, port: int, sock: socket.socket | None = None):
        self.host = host
        self.port = port
        self.sock = sock

    def connect(self, retries: int = 5, wait: float = 1.0) -> bool:
        for _ in range(retries):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.connect((self.host, self.port))
                self.sock.settimeout(None)
                return True
            except ConnectionRefusedError:
                time.sleep(wait)
        return False

    def close(self) -> None:
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def recv_some(self, maxbytes: int = 65536, timeout: float = 1.0) -> bytes:
        if not self.sock:
            return b""
        try:
            self.sock.settimeout(timeout)
            data = self.sock.recv(maxbytes)
            return data or b""
        except (socket.timeout, BlockingIOError):
            return b""
        except Exception:
            return b""
        finally:
            try:
                self.sock.settimeout(None)
            except Exception:
                pass

    def robust_send_all(self, buf: bytes) -> None:
        if not self.sock:
            raise RuntimeError("socket not connected")
        mv = memoryview(buf)
        total = len(buf)
        sent = 0
        while sent < total:
            try:
                n = self.sock.send(mv[sent:])
                if n == 0:
                    raise RuntimeError("socket send returned 0")
                sent += n
            except (BlockingIOError, InterruptedError):
                _ = self.recv_some(65536, timeout=0.05)
                time.sleep(0.001)
            except socket.timeout:
                _ = self.recv_some(65536, timeout=0.05)
            except Exception:
                raise

    def recv_until_min_lines(self, min_lines: int, max_total_wait: float = 10.0) -> bytes:
        if min_lines <= 0:
            return b""
        buf = bytearray()
        deadline = time.time() + max_total_wait
        while time.time() < deadline:
            chunk = self.recv_some(65536, timeout=0.5)
            if chunk:
                buf.extend(chunk)
                if buf.count(b"\n") >= min_lines:
                    break
            else:
                time.sleep(0.001)
        return bytes(buf)

    def insert_vectors(self, count: int, dim: int, batch_size: int = BATCH_SIZE) -> Tuple[float, List[float]]:
        # 1. Generate Data Pool (Speed up Python)
        # Reuse 5000 vectors to simulate load without spending minutes generating floats
        pool_size = min(count, 5000)
        print(f" [Client] Pre-generating pool of {pool_size} vectors...")
        pool: List[bytes] = []
        for _ in range(pool_size):
            vec = [random.random() for _ in range(dim)]
            pool.append(struct.pack(f"{dim}f", *vec))
        
        print(f" [Client] Sending {count} vectors in batches of {batch_size}...")
        batch_times: List[float] = []
        t0 = time.time()
        
        # 2. Stream Batches (Low Memory Footprint)
        for i in range(0, count, batch_size):
            current_batch_size = min(batch_size, count - i)
            batch_parts = []
            
            for j in range(current_batch_size):
                idx = i + j
                # Round-robin pick from pool
                packed_vec = pool[idx % pool_size]
                key = f"key:{idx}".encode()
                
                cmd = (
                    b"*3\r\n$4\r\nVSET\r\n"
                    + f"${len(key)}\r\n".encode() + key + b"\r\n"
                    + f"${len(packed_vec)}\r\n".encode() + packed_vec + b"\r\n"
                )
                batch_parts.append(cmd)
            
            batch_data = b"".join(batch_parts)
            
            bt0 = time.time()
            self.robust_send_all(batch_data)
            
            # Wait for ACKs
            _ = self.recv_until_min_lines(current_batch_size, max_total_wait=BATCH_SEND_TIMEOUT)
            bt1 = time.time()
            batch_times.append(bt1 - bt0)
            
            if i > 0 and i % 50000 == 0:
                print(f"   ...sent {i}/{count}...")

        t1 = time.time()
        total_time = t1 - t0 if t1 > t0 else 1e-6
        qps = count / total_time
        return qps, batch_times

    def benchmark_search_latencies(self, dim: int, iters: int, topk: int = SEARCH_TOP_K) -> List[float]:
        print(f" [Client] Benchmarking Search ({iters} iters)...")
        latencies: List[float] = []
        for _ in range(iters):
            query = struct.pack(f"{dim}f", *[random.random() for _ in range(dim)])
            cmd = (
                b"*3\r\n$7\r\nVSEARCH\r\n"
                + f"${len(query)}\r\n".encode() + query + b"\r\n"
                + f"${len(str(topk))}\r\n{topk}\r\n".encode()
            )
            t0 = time.time()
            self.robust_send_all(cmd)
            _ = self.recv_some(65536, timeout=2.0)
            t1 = time.time()
            latencies.append((t1 - t0) * 1000.0)
        return latencies

# ---------------- Runner ----------------

def run_server(env: Dict[str, str]) -> Tuple[subprocess.Popen, bool]:
    try:
        subprocess.run(["fuser", "-k", f"{PORT}/tcp"], stderr=subprocess.DEVNULL)
    except Exception:
        pass

    if not os.path.exists(SERVER_BIN):
        print("Server binary not found:", SERVER_BIN)
        return None, False

    env_copy = os.environ.copy()
    env_copy.update(env)
    stdout_f = open("pomai_server_stdout.log", "ab")
    stderr_f = open("pomai_server_stderr.log", "ab")
    proc = subprocess.Popen([SERVER_BIN], env=env_copy, stdout=stdout_f, stderr=stderr_f)
    return proc, True

def run_case(name: str, env_vars: Dict[str, str], num_vectors: int, dim: int, batch_size: int) -> Dict:
    print(f"\n=== RUN: {name} ===")
    proc, ok = run_server(env_vars)
    if not ok or proc is None:
        print("Failed to start server")
        return {}

    time.sleep(SERVER_STARTUP_WAIT)

    client = PomaiClient(HOST, PORT)
    if not client.connect(retries=10, wait=0.5):
        print("Client failed to connect to server")
        proc.terminate()
        proc.wait(timeout=3.0)
        return {}

    # Warmup / Init
    warm = struct.pack(f"{dim}f", *([0.0] * dim))
    init_cmd = (
        b"*3\r\n$4\r\nVSET\r\n$4\r\ninit\r\n"
        + f"${len(warm)}\r\n".encode() + warm + b"\r\n"
    )
    client.robust_send_all(init_cmd)
    _ = client.recv_some(4096, timeout=2.0)
    time.sleep(0.5)

    try:
        qps, batch_durations = client.insert_vectors(num_vectors, dim, batch_size)
    except Exception as e:
        print("Error during insert:", e)
        client.close()
        proc.terminate()
        try: proc.wait(timeout=3.0) 
        except: proc.kill()
        return {}

    mem_mb, cpu_pct = get_process_stats(proc.pid)
    latencies = client.benchmark_search_latencies(dim, SEARCH_ITERS, SEARCH_TOP_K)

    client.close()
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()

    result = {
        "name": name,
        "qps": qps,
        "mem_mb": mem_mb,
        "cpu_pct": cpu_pct,
        "latencies_ms": latencies,
    }
    print(f" -> RESULTS: qps={qps:.0f}, mem={mem_mb:.1f}MB, cpu={cpu_pct:.1f}%")
    return result

# ---------------- Reporting ----------------

def summarize(latencies: List[float]) -> Dict[str, float]:
    if not latencies: return {}
    s = {}
    s["mean_ms"] = statistics.mean(latencies)
    s["median_ms"] = statistics.median(latencies)
    s["p95_ms"] = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
    return s

def save_summary_csv(results: List[Dict], path: str = "pomai_benchmark_summary.csv") -> None:
    rows = []
    for r in results:
        summ = summarize(r.get("latencies_ms", []))
        rows.append({
            "name": r.get("name", ""),
            "qps": r.get("qps", 0.0),
            "mem_mb": r.get("mem_mb", 0.0),
            "lat_mean_ms": summ.get("mean_ms", 0.0),
            "lat_p95_ms": summ.get("p95_ms", 0.0),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote CSV summary to", path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=NUM_VECTORS)
    parser.add_argument("--dim", type=int, default=DIMENSIONS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"Benchmark: {args.vectors} vectors, dim {args.dim}, batch {args.batch}")

    # Set bigger arena for Raw mode to avoid instant crash on small machines
    # (Though it might still crash if physical RAM is exhausted)
    raw_env = {"POMAI_DISABLE_SYNAPSE": "1", "POMAI_PORT": str(PORT), "POMAI_ARENA_MB": "4096"}
    syn_env = {"POMAI_PORT": str(PORT), "POMAI_ARENA_MB": "1024"}

    raw = run_case("Raw Float", raw_env, args.vectors, args.dim, args.batch)
    syn = run_case("Synapse 4-bit", syn_env, args.vectors, args.dim, args.batch)

    if raw:
        raw_s = summarize(raw.get("latencies_ms", []))
        print(f"\nRAW : QPS={raw['qps']:.0f}, MEM={raw['mem_mb']:.1f}MB, P95={raw_s.get('p95_ms',0):.2f}ms")
    else:
        print("\nRAW : CRASHED / FAILED")

    if syn:
        syn_s = summarize(syn.get("latencies_ms", []))
        print(f"SYN : QPS={syn['qps']:.0f}, MEM={syn['mem_mb']:.1f}MB, P95={syn_s.get('p95_ms',0):.2f}ms")
    else:
        print("SYN : CRASHED / FAILED")

    valid_results = [r for r in [raw, syn] if r]
    if valid_results:
        save_summary_csv(valid_results)

if __name__ == "__main__":
    main()