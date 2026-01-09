#!/usr/bin/env python3
"""
bench_query.py -- concurrent query benchmark (VSEARCH).

Usage:
  python3 ./benchmarks/vector_query.py --host 160.30.192.118 --dim 512 --topk 5 --qps 10 --clients 1 --duration 10

Behavior:
  - Spawns `clients` worker threads
  - Each worker sends queries in a loop, target total qps ~= qps (best-effort)
  - Reports p50/p90/p99/avg latency and total qps at the end
"""
import argparse
import socket
import struct
import time
import random
import threading
import statistics
import sys

PWP_MAGIC = ord('P')
OP_VSEARCH = 11

def read_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket closed")
        buf += chunk
    return buf

def do_query(host, port, topk, vec_bytes, timeout=5.0):
    body = struct.pack("!I", topk) + vec_bytes
    hdr = struct.pack("!BBHIII", PWP_MAGIC, OP_VSEARCH, 0, 0, len(body), 0)
    start = time.time()
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(hdr + body)
        resp_hdr = read_exact(s, 16)
        magic, op, status, rklen, rvlen, reserved = struct.unpack("!BBHIII", resp_hdr)
        if magic != PWP_MAGIC:
            raise RuntimeError("bad magic")
        if status != 0:
            raise RuntimeError(f"server status {status}")
        if rvlen:
            # consume but ignore result content
            read_exact(s, rvlen)
    return (time.time() - start)

def worker_thread(host, port, topk, dim, per_thread_qps, duration, stats_list, seed):
    random.seed(seed)
    sleep_per_query = 1.0 / per_thread_qps if per_thread_qps > 0 else 0
    vec_bytes_cache = None
    t_end = time.time() + duration
    latencies = []
    while time.time() < t_end:
        # reuse a small random query of given dim
        vals = [random.random() for _ in range(dim)]
        vec_bytes = b''.join(struct.pack('f', float(x)) for x in vals)
        try:
            lat = do_query(host, port, topk, vec_bytes)
            latencies.append(lat)
        except Exception as e:
            # track as a very large latency or skip
            print(f"[WARN] query fail: {e}", file=sys.stderr)
        if sleep_per_query > 0:
            time.sleep(sleep_per_query)
    stats_list.append(latencies)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--qps", type=int, default=100, help="total target QPS across clients")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--duration", type=int, default=30, help="seconds")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    per_thread_qps = float(args.qps) / max(1, args.clients)
    threads = []
    stats_list = []
    t0 = time.time()
    for i in range(args.clients):
        t = threading.Thread(target=worker_thread,
                             args=(args.host, args.port, args.topk, args.dim, per_thread_qps, args.duration, stats_list, args.seed + i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    elapsed = time.time() - t0

    # flatten stats
    all_lats = [x for lst in stats_list for x in lst]
    total_q = len(all_lats)
    avg = statistics.mean(all_lats) if all_lats else 0.0
    p50 = statistics.median(all_lats) if all_lats else 0.0
    p90 = sorted(all_lats)[int(0.9 * total_q)] if total_q else 0.0
    p99 = sorted(all_lats)[min(int(0.99 * total_q), max(0, total_q-1))] if total_q else 0.0
    print("Benchmark results:")
    print(f"  duration: {elapsed:.2f}s  total_queries: {total_q}  achieved_qps: {total_q/elapsed:.2f}")
    print(f"  latencies (s): avg={avg:.4f} p50={p50:.4f} p90={p90:.4f} p99={p99:.4f}")

if __name__ == "__main__":
    main()