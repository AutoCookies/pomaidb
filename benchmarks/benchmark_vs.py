#!/usr/bin/env python3
"""
benchmarks/benchmark_q1_paper.py
--------------------------------
POMAIDB SCIENTIFIC BENCHMARK SUITE - TURBO EDITION (v2.2)

This is an extended benchmark that now measures:
 - INSERT (batched + pipelined)
 - SEARCH (baseline + metadata-filtered)
 - GET (existing / post-delete)
 - DELETE (soft-delete)
 - Basic DDL checks (CREATE/DROP/SHOW)
 - System telemetry (cpu, mem) and detailed CSV+plots

Usage:
  - Ensure server (pomai_server) is running and listening at HOST:PORT.
  - Requires python3 with numpy, matplotlib, psutil
"""

from __future__ import annotations
import socket
import time
import threading
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import random
import math

try:
    import psutil
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing dependencies:", e)
    sys.exit(1)

# -------------------------
# Config
# -------------------------
HOST = "127.0.0.1"
PORT = 7777
SERVER_BIN_NAME = "pomai_server"

DIM = 512
MILESTONES = [10000, 50000, 100000, 200000, 500000, 1000000, 5000000, 10000000, 20000000]  # practical defaults; extend if you can
WARMUP_VECTORS = 2000
INSERT_BATCH_SIZE = 2000   # number of tuples per SQL when tags identical
PIPELINED_BATCH_SIZE = 500 # when tags differ we send this many single-insert statements in one send()
SEARCH_TRIALS = 2000
SEARCH_CONCURRENCY = 32    # number of parallel clients issuing searches

OUTPUT_DIR = "pomai_scientific_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# System monitor (client-side)
# -------------------------
class SystemMonitor(threading.Thread):
    def __init__(self, pid_hint=None, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.history = {"ts": [], "cpu": [], "mem": [], "phase": []}
        self.phase = "init"
        self.stop_ev = threading.Event()
        self.pid_hint = pid_hint
        self.proc = None
        self._attach_proc()

    def _attach_proc(self):
        if self.pid_hint:
            try:
                self.proc = psutil.Process(int(self.pid_hint))
                return
            except Exception:
                self.proc = None
        # fallback: find by name
        for p in psutil.process_iter(['pid','name']):
            if SERVER_BIN_NAME in (p.info.get('name') or ""):
                try:
                    self.proc = psutil.Process(p.info['pid'])
                    return
                except Exception:
                    self.proc = None

    def set_phase(self, s):
        self.phase = s

    def run(self):
        start = time.time()
        while not self.stop_ev.is_set():
            now = time.time() - start
            cpu, mem = 0.0, 0.0
            if self.proc:
                try:
                    cpu = self.proc.cpu_percent(interval=None)
                    mem = self.proc.memory_info().rss / (1024*1024)
                except Exception:
                    self.proc = None
            self.history["ts"].append(now)
            self.history["cpu"].append(cpu)
            self.history["mem"].append(mem)
            self.history["phase"].append(self.phase)
            time.sleep(self.interval)

    def stop(self):
        self.stop_ev.set()

# -------------------------
# Networking client (thread-safe) + pool helper
# -------------------------
class SimpleClient:
    def __init__(self, host=HOST, port=PORT, timeout=60.0):
        self.host = host; self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.recv_buf = bytearray()
        self.lock = threading.Lock()

    def connect(self):
        self.sock.connect((self.host, self.port))

    def close(self):
        try:
            self.sock.close()
        except:
            pass

    # send many bytes, single writer at a time
    def send_all(self, data: bytes):
        with self.lock:
            self.sock.sendall(data)

    # receive until marker, single reader at a time
    def recv_response_exact(self) -> bytes:
        marker = b"<END>\n"
        with self.lock:
            # reuse buffer across calls
            while True:
                if marker in self.recv_buf:
                    idx = self.recv_buf.find(marker)
                    resp = bytes(self.recv_buf[:idx])
                    del self.recv_buf[:idx+len(marker)]
                    return resp
                try:
                    chunk = self.sock.recv(65536)
                except socket.timeout:
                    raise TimeoutError("recv timed out")
                if not chunk:
                    raise ConnectionError("socket closed")
                self.recv_buf.extend(chunk)

# client pool: one client per thread (simple)
def make_client_pool(n):
    pool = []
    for _ in range(n):
        c = SimpleClient()
        c.connect()
        pool.append(c)
    return pool

def close_client_pool(pool):
    for c in pool:
        c.close()

# -------------------------
# Utilities: vector & tag generation
# -------------------------
def generate_vector_pool(size, dim):
    data = np.random.rand(size, dim).astype(np.float32)
    # store as CSV strings to avoid reformat cost in inner loop
    return [",".join(f"{x:.6f}" for x in row) for row in data]

def deterministic_tags(idx):
    tags = []
    if idx % 2 == 0:
        tags.append("selectivity:common")
    if idx % 10 == 0:
        tags.append("selectivity:medium")
    if idx % 100 == 0:
        tags.append("selectivity:rare")
    return tags

def tags_to_sql_paren(tags: List[str]) -> str:
    if not tags: return ""
    kvs = ", ".join(tags)
    return f"({kvs})"

# -------------------------
# Insertion: maximize use of multi-tuple INSERT when possible
# -------------------------
def batch_insert_membrane(client: SimpleClient, membr, start_idx, vectors: List[str], tags_batch: List[str]):
    """
    vectors: list of CSV vector strings
    tags_batch: list of tag-strings (each either "" or "(k:v,...)")
    Attempt to use large single multi-tuple INSERT when all tags identical,
    otherwise send pipelined single INSERT statements grouped for fewer syscalls.
    """
    n = len(vectors)
    if n == 0:
        return 0
    same_tag = all(t == tags_batch[0] for t in tags_batch)
    success = 0
    try:
        if same_tag and tags_batch[0] != "":
            tuples = []
            for j in range(n):
                idx = start_idx + j
                tuples.append(f"(k_{idx}, [{vectors[j]}])")
            sql = f"INSERT INTO {membr} VALUES " + ",".join(tuples) + f" TAGS {tags_batch[0]};\n"
            client.send_all(sql.encode())
            client.recv_response_exact()
            success = n
        elif same_tag:
            tuples = []
            for j in range(n):
                idx = start_idx + j
                tuples.append(f"(k_{idx}, [{vectors[j]}])")
            sql = f"INSERT INTO {membr} VALUES " + ",".join(tuples) + ";\n"
            client.send_all(sql.encode())
            client.recv_response_exact()
            success = n
        else:
            parts = []
            for j in range(n):
                idx = start_idx + j
                sql = f"INSERT INTO {membr} VALUES (k_{idx}, [{vectors[j]}])"
                if tags_batch[j]:
                    sql += f" TAGS {tags_batch[j]}"
                sql += ";"
                parts.append(sql)
            payload = "\n".join(parts) + "\n"
            client.send_all(payload.encode())
            for _ in range(n):
                client.recv_response_exact()
                success += 1
    except Exception as e:
        # best-effort; return what we managed
        print("[!] insert error:", e)
    return success

# -------------------------
# GET / DELETE tests
# -------------------------
def test_get_delete(client: SimpleClient, membr: str, label_indices: List[int]) -> Dict:
    """
    Performs GET on a list of label indices, then DELETE, then GET again to confirm deletion.
    Returns latency stats and success/failure counts.
    """
    get_lat = []
    get_ok = 0
    get_fail = 0

    # GET existing
    for idx in label_indices:
        sql = f"GET {membr} LABEL k_{idx};\n"
        t0 = time.time()
        try:
            client.send_all(sql.encode())
            resp = client.recv_response_exact()
            t1 = time.time()
            get_lat.append((t1 - t0) * 1000.0)
            text = resp.decode(errors='ignore').lower()
            if "vector " in text:
                get_ok += 1
            else:
                get_fail += 1
        except Exception:
            get_lat.append(10000.0)
            get_fail += 1

    # DELETE existing
    del_lat = []
    del_ok = 0
    del_fail = 0
    for idx in label_indices:
        sql = f"DELETE {membr} LABEL k_{idx};\n"
        t0 = time.time()
        try:
            client.send_all(sql.encode())
            resp = client.recv_response_exact()
            t1 = time.time()
            del_lat.append((t1 - t0) * 1000.0)
            text = resp.decode(errors='ignore').lower()
            if text.startswith("ok"):
                del_ok += 1
            else:
                del_fail += 1
        except Exception:
            del_lat.append(10000.0)
            del_fail += 1

    # GET after delete (should fail)
    get2_lat = []
    get2_fail = 0
    for idx in label_indices:
        sql = f"GET {membr} LABEL k_{idx};\n"
        t0 = time.time()
        try:
            client.send_all(sql.encode())
            resp = client.recv_response_exact()
            t1 = time.time()
            get2_lat.append((t1 - t0) * 1000.0)
            text = resp.decode(errors='ignore').lower()
            if "not found" in text or "err" in text:
                get2_fail += 1
        except Exception:
            get2_lat.append(10000.0)
            get2_fail += 1

    def p50(xs): return float(np.percentile(xs, 50)) if xs else 0.0

    return {
        "get_p50": p50(get_lat),
        "get_succ": get_ok,
        "get_fail": get_fail,
        "del_p50": p50(del_lat),
        "del_succ": del_ok,
        "del_fail": del_fail,
        "get2_p50": p50(get2_lat),
        "get2_fail": get2_fail
    }

# -------------------------
# Search helpers: use concurrent clients to stress test
# -------------------------
def search_one(client: SimpleClient, membr: str, qvec_csv: str, where_clause: str = "", topk: int = 10):
    """
    Send single search and return (latency_ms, results_count)
    """
    sql = f"SEARCH {membr} QUERY ([{qvec_csv}])"
    if where_clause:
        sql += f" WHERE {where_clause}"
    sql += f" TOP {topk};\n"
    t0 = time.time()
    client.send_all(sql.encode())
    resp = client.recv_response_exact()
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000.0
    try:
        line = resp.decode(errors='ignore').splitlines()[0]
        if line.startswith("RESULTS"):
            parts = line.split()
            cnt = int(parts[1]) if len(parts) > 1 else 0
        else:
            cnt = 0
    except Exception:
        cnt = 0
    return latency_ms, cnt

def run_concurrent_searches(client_pool: List[SimpleClient], membr: str, qvecs: List[str],
                            where_clause: str, trials: int, concurrency: int, topk: int = 10):
    latencies = []
    results_counts = []
    pool_size = len(client_pool)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for i in range(trials):
            client = client_pool[i % pool_size]
            q = qvecs[i % len(qvecs)]
            futures.append(ex.submit(search_one, client, membr, q, where_clause, topk))
        for f in as_completed(futures):
            try:
                lat, cnt = f.result()
                latencies.append(lat)
                results_counts.append(cnt)
            except Exception as e:
                latencies.append(10000.0)
                results_counts.append(0)
    return latencies, results_counts

# -------------------------
# Top-level flow
# -------------------------
def summarize_stats(latencies: List[float]) -> Dict:
    if not latencies:
        return {"n":0, "avg":0, "p50":0, "p90":0, "p99":0}
    a = np.array(latencies)
    return {"n": len(a), "avg": float(np.mean(a)), "p50": float(np.percentile(a,50)),
            "p90": float(np.percentile(a,90)), "p99": float(np.percentile(a,99))}

def run_scientific_suite():
    monitor = SystemMonitor()
    monitor.start()

    ctl_client = SimpleClient()
    ctl_client.connect()

    client_pool = make_client_pool(max(SEARCH_CONCURRENCY, 8))

    MEMBR = "pomai_research_db"
    results = []

    try:
        # init DDL
        print("[+] Resetting membrance (DROP/CREATE)")
        try:
            ctl_client.send_all(f"DROP MEMBRANCE {MEMBR};\n".encode())
            # swallow response if not present
            try:
                ctl_client.recv_response_exact()
            except Exception:
                pass
        except Exception:
            pass

        ctl_client.send_all(f"CREATE MEMBRANCE {MEMBR} DIM {DIM} RAM 4096;\n".encode())
        rsp = ctl_client.recv_response_exact()
        print("[+] Create response:", rsp.decode(errors='ignore').strip())

        current = 0

        # Pre-generate vector pool
        gen_pool = generate_vector_pool(10000, DIM)

        for target in MILESTONES:
            monitor.set_phase(f"idle_{target}")
            time.sleep(0.5)
            need = target - current
            if need > 0:
                monitor.set_phase(f"insert_{target}")
                t0 = time.time()
                ptr = 0
                while ptr < need:
                    block = min(INSERT_BATCH_SIZE, need - ptr)
                    vectors = [gen_pool[(current + ptr + i) % len(gen_pool)] for i in range(block)]
                    tags_batch = []
                    for i in range(block):
                        idx = current + ptr + i
                        tags = deterministic_tags(idx)
                        tags_batch.append(tags_to_sql_paren(tags) if tags else "")

                    from collections import Counter
                    cnt = Counter(tags_batch)
                    most_tag, most_count = cnt.most_common(1)[0]
                    if most_count >= 0.9 * block:
                        batch_insert_membrane(ctl_client, MEMBR, current + ptr, vectors, [most_tag]*block)
                        ptr += block
                    else:
                        sub_ptr = 0
                        while sub_ptr < block:
                            sub = min(PIPELINED_BATCH_SIZE, block - sub_ptr)
                            subv = vectors[sub_ptr:sub_ptr+sub]
                            subtags = tags_batch[sub_ptr:sub_ptr+sub]
                            batch_insert_membrane(ctl_client, MEMBR, current + ptr + sub_ptr, subv, subtags)
                            sub_ptr += sub
                        ptr += block

                dur = time.time() - t0
                insert_qps = need / dur if dur > 0 else 0.0
                current = target
                print(f"[+] Inserted up to {current:,} total vectors (throughput {insert_qps:.0f} vec/s)")
            else:
                insert_qps = 0.0

            # small sample of labels for GET/DELETE tests (avoid huge cost)
            sample_size = min(200, max(10, current // 1000))
            label_samples = random.sample(range(max(0, current - need), current) if current>0 else [], sample_size) if current>0 else []
            # Warmup
            monitor.set_phase(f"warmup_{target}")
            warm_qs = [gen_pool[i % len(gen_pool)] for i in range(100)]
            _ = run_concurrent_searches(client_pool, MEMBR, warm_qs, "", trials=200, concurrency=8)

            # Baseline search
            monitor.set_phase(f"baseline_{target}")
            qpool = [gen_pool[i % len(gen_pool)] for i in range(min(SEARCH_TRIALS, len(gen_pool)))]
            lat_base, _ = run_concurrent_searches(client_pool, MEMBR, qpool, "", trials=min(SEARCH_TRIALS, len(qpool)), concurrency=min(SEARCH_CONCURRENCY, len(client_pool)))
            stats_base = summarize_stats(lat_base)
            print(f"[+] Baseline (N={target}): p50={stats_base['p50']:.3f}ms p99={stats_base['p99']:.3f}ms avg={stats_base['avg']:.3f}ms")

            # Filtered
            monitor.set_phase(f"filtered_{target}")
            filter_stats = {}
            for sel, expected_pct in [("common", 50.0), ("medium", 10.0), ("rare", 1.0)]:
                where = f"selectivity='{sel}'"
                lat_f, _ = run_concurrent_searches(client_pool, MEMBR, qpool, where, trials=min(1000, len(qpool)), concurrency=min(SEARCH_CONCURRENCY, len(client_pool)))
                stats_f = summarize_stats(lat_f)
                filter_stats[sel] = stats_f
                print(f"    Filter {sel} (~{expected_pct}%): p50={stats_f['p50']:.3f}ms p99={stats_f['p99']:.3f}ms")

            # GET / DELETE tests (on ctl_client serially)
            getdel_stats = {}
            if label_samples:
                monitor.set_phase(f"getdel_{target}")
                getdel_stats = test_get_delete(ctl_client, MEMBR, label_samples)
                print(f"    GET p50={getdel_stats['get_p50']:.3f}ms succ={getdel_stats['get_succ']} fail={getdel_stats['get_fail']}")
                print(f"    DEL p50={getdel_stats['del_p50']:.3f}ms succ={getdel_stats['del_succ']} fail={getdel_stats['del_fail']}")
            else:
                getdel_stats = {"get_p50":0,"get_succ":0,"get_fail":0,"del_p50":0,"del_succ":0,"del_fail":0,"get2_p50":0,"get2_fail":0}

            # Snapshot system metrics
            last_cpu = monitor.history["cpu"][-1] if monitor.history["cpu"] else 0.0
            last_mem = monitor.history["mem"][-1] if monitor.history["mem"] else 0.0

            results.append({
                "vectors": target,
                "insert_qps": insert_qps,
                "baseline": stats_base,
                "filtered": filter_stats,
                "cpu": last_cpu,
                "mem": last_mem,
                "raw_base_lat": lat_base,
                **getdel_stats
            })

    except KeyboardInterrupt:
        print("[!] Interrupted")
    finally:
        monitor.stop()
        close_client_pool(client_pool)
        try:
            ctl_client.close()
        except:
            pass
        time.sleep(0.1)

    return results, monitor.history

# -------------------------
# Reporting
# -------------------------
def generate_report(results, sys_hist):
    if not results:
        print("No results to report")
        return

    vectors = [r['vectors'] for r in results]
    p50s = [r['baseline']['p50'] for r in results]
    p99s = [r['baseline']['p99'] for r in results]

    # Scalability plot
    plt.figure(figsize=(8,5))
    plt.plot(vectors, p50s, 'o-', label='P50')
    plt.plot(vectors, p99s, 's-', label='P99')
    plt.xscale('log')
    plt.xlabel("Dataset size")
    plt.ylabel("Latency (ms)")
    plt.title("Baseline Search Scalability (concurrent)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, "scalability_baseline.png"), dpi=150)
    plt.close()

    # Filter selectivity chart (last)
    last = results[-1]
    labels = ['NoFilter', 'Common(50%)', 'Medium(10%)', 'Rare(1%)']
    values = [
        last['baseline']['p50'],
        last['filtered']['common']['p50'],
        last['filtered']['medium']['p50'],
        last['filtered']['rare']['p50']
    ]
    plt.figure(figsize=(8,5))
    bars = plt.bar(labels, values, color=['gray','blue','purple','green'])
    for b in bars:
        plt.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f"{b.get_height():.2f}ms", ha='center', va='bottom')
    plt.title(f"Filter Selectivity (N={last['vectors']:,})")
    plt.ylabel("P50 latency (ms)")
    plt.savefig(os.path.join(OUTPUT_DIR, "filter_selectivity.png"), dpi=150)
    plt.close()

    # GET/DELETE summaries across milestones
    get_p50s = [r.get('get_p50',0) for r in results]
    del_p50s = [r.get('del_p50',0) for r in results]
    plt.figure(figsize=(8,5))
    plt.plot(vectors, get_p50s, 'o-', label='GET P50')
    plt.plot(vectors, del_p50s, 's-', label='DEL P50')
    plt.xscale('log')
    plt.xlabel("Dataset size")
    plt.ylabel("Latency (ms)")
    plt.title("GET / DELETE Latency")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(OUTPUT_DIR, "get_del_latency.png"), dpi=150)
    plt.close()

    # Save CSV summary
    csvfile = os.path.join(OUTPUT_DIR, "results_summary.csv")
    with open(csvfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "vectors","insert_qps","base_p50","base_p99","cpu_pct","mem_mb",
            "filter_common_p50","filter_common_p99","filter_medium_p50","filter_medium_p99","filter_rare_p50","filter_rare_p99",
            "get_p50","get_succ","get_fail","del_p50","del_succ","del_fail","get_after_del_p50","get_after_del_fail"
        ])
        for r in results:
            w.writerow([
                r['vectors'],
                f"{r['insert_qps']:.1f}",
                f"{r['baseline']['p50']:.3f}", f"{r['baseline']['p99']:.3f}",
                f"{r['cpu']:.2f}", f"{r['mem']:.1f}",
                f"{r['filtered']['common']['p50']:.3f}", f"{r['filtered']['common']['p99']:.3f}",
                f"{r['filtered']['medium']['p50']:.3f}", f"{r['filtered']['medium']['p99']:.3f}",
                f"{r['filtered']['rare']['p50']:.3f}", f"{r['filtered']['rare']['p99']:.3f}",
                f"{r.get('get_p50',0):.3f}", r.get('get_succ',0), r.get('get_fail',0),
                f"{r.get('del_p50',0):.3f}", r.get('del_succ',0), r.get('del_fail',0),
                f"{r.get('get2_p50',0):.3f}", r.get('get2_fail',0)
            ])
    print("[+] Saved CSV:", csvfile)
    print("[+] Saved plots in", OUTPUT_DIR)

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    print("=== POMAI SCIENTIFIC BENCHMARK (Turbo v2.2) ===")
    res, sys_hist = run_scientific_suite()
    if res:
        generate_report(res, sys_hist)
        print("Done. Results saved to", OUTPUT_DIR)
    else:
        print("No results collected.")