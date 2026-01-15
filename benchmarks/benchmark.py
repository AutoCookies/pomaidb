#!/usr/bin/env python3
"""
benchmarks/benchmark_q1_paper_v3.py

Robust Pomai scientific benchmark with enhanced monitoring and plots.

Usage example:
  python3 benchmarks/benchmark_q1_paper_v3.py --milestones 10000 50000 100000 200000 500000 1000000 2000000 5000000 10000000 20000000 50000000 100000000 --server-pid 3867

Dependencies: numpy, matplotlib, psutil
"""

from __future__ import annotations
import argparse
import socket
import time
import threading
import csv
import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import random
import math
import statistics
from collections import Counter

try:
    import psutil
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing dependencies:", e)
    sys.exit(1)

# ---- Defaults ----
HOST = "127.0.0.1"
PORT = 7777
SERVER_BIN_NAME = "pomai_server"

DEFAULT_DIM = 512
DEFAULT_MILESTONES = [10000, 50000, 100000, 200000, 500000, 1000000]
DEFAULT_INSERT_BATCH = 5000
DEFAULT_PIPELINED_BATCH = 5000
DEFAULT_GEN_POOL = 10000
DEFAULT_SEARCH_TRIALS = 2000
DEFAULT_SEARCH_CONCURRENCY = 32
OUTPUT_DIR_DEFAULT = "pomai_results"

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="PomaiDB Scientific Benchmark (v3)")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", default=PORT, type=int)
    p.add_argument("--dim", default=DEFAULT_DIM, type=int)
    p.add_argument("--milestones", nargs="+", type=int, default=DEFAULT_MILESTONES)
    p.add_argument("--insert-batch", default=DEFAULT_INSERT_BATCH, type=int)
    p.add_argument("--pipelined-batch", default=DEFAULT_PIPELINED_BATCH, type=int)
    p.add_argument("--gen-pool-size", default=DEFAULT_GEN_POOL, type=int)
    p.add_argument("--search-trials", default=DEFAULT_SEARCH_TRIALS, type=int)
    p.add_argument("--search-concurrency", default=DEFAULT_SEARCH_CONCURRENCY, type=int)
    p.add_argument("--timeout", default=30.0, type=float)
    p.add_argument("--output", default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--only-milestones", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--server-pid", default=None, type=int, help="Optional PID of the server to monitor")
    return p.parse_args()

# ---- Monitoring ----
class SystemMonitor(threading.Thread):
    """
    Best-effort monitor that attaches to server process (by PID, name/cmdline, or listening port).
    Records timestamped cpu% and RSS (MB).
    Notes:
      - psutil.cpu_percent() needs a priming call; we call it once when attaching.
      - If attach fails, monitor runs but will record zeros.
    """
    def __init__(self, pid_hint: Optional[int] = None, interval: float = 0.2, host: str = HOST, port: int = PORT):
        super().__init__(daemon=True)
        self.interval = interval
        self.history = {"ts": [], "cpu": [], "mem": [], "phase": []}
        self.phase = "init"
        self.stop_ev = threading.Event()
        self.pid_hint = pid_hint
        self.proc: Optional[psutil.Process] = None
        self.host = host
        self.port = port
        self._attach_proc()

    def _attach_proc(self):
        # explicit PID
        if self.pid_hint:
            try:
                self.proc = psutil.Process(int(self.pid_hint))
                try:
                    self.proc.cpu_percent(interval=None)
                except Exception:
                    pass
                print(f"[Monitor] attached to PID {self.pid_hint}")
                return
            except Exception:
                self.proc = None

        # try by name or cmdline
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                name = (p.info.get('name') or "").lower()
                cmd = " ".join(p.info.get('cmdline') or []).lower()
                if SERVER_BIN_NAME in name or SERVER_BIN_NAME in cmd:
                    try:
                        self.proc = psutil.Process(p.info['pid'])
                        try:
                            self.proc.cpu_percent(interval=None)
                        except Exception:
                            pass
                        print(f"[Monitor] attached to PID {p.info['pid']} (name/cmd detect)")
                        return
                    except Exception:
                        self.proc = None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # try by listening socket on port
        try:
            # iterate network connections and match listening local port
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr and conn.laddr.port == self.port:
                    pid = conn.pid
                    if pid:
                        try:
                            self.proc = psutil.Process(pid)
                            try:
                                self.proc.cpu_percent(interval=None)
                            except Exception:
                                pass
                            print(f"[Monitor] attached to PID {pid} via listening port {self.port}")
                            return
                        except Exception:
                            continue
        except Exception:
            # net_connections might require privileges on some systems
            pass

        print("[Monitor] warning: could not auto-attach to server process; CPU/mem stats will be zeros")

    def set_phase(self, s: str):
        self.phase = s

    def run(self):
        start = time.time()
        while not self.stop_ev.is_set():
            now = time.time() - start
            cpu, mem = 0.0, 0.0
            if self.proc:
                try:
                    cpu = self.proc.cpu_percent(interval=None)
                    mem = self.proc.memory_info().rss / (1024.0 * 1024.0)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.proc = None
                    cpu, mem = 0.0, 0.0
            self.history["ts"].append(now)
            self.history["cpu"].append(cpu)
            self.history["mem"].append(mem)
            self.history["phase"].append(self.phase)
            time.sleep(self.interval)

    def stop(self):
        self.stop_ev.set()

# ---- Networking client ----
class SimpleClient:
    """
    Minimal thread-safe TCP client with small recv buffer and marker-delimited responses.
    Adjust 'marker' if your server uses a different response terminator.
    """
    def __init__(self, host=HOST, port=PORT, timeout=30.0, marker: bytes = b"<END>\n"):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.marker = marker
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.recv_buf = bytearray()
        self.lock = threading.Lock()

    def connect(self):
        self.sock.connect((self.host, self.port))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def send_all(self, data: bytes):
        with self.lock:
            self.sock.sendall(data)

    def recv_response_exact(self) -> bytes:
        with self.lock:
            while True:
                if self.marker in self.recv_buf:
                    idx = self.recv_buf.find(self.marker)
                    resp = bytes(self.recv_buf[:idx])
                    del self.recv_buf[:idx + len(self.marker)]
                    return resp
                try:
                    chunk = self.sock.recv(65536)
                except socket.timeout:
                    raise TimeoutError("recv timed out")
                if not chunk:
                    raise ConnectionError("socket closed")
                self.recv_buf.extend(chunk)

def make_client_pool(n: int, host: str, port: int, timeout: float):
    pool = []
    for _ in range(n):
        c = SimpleClient(host, port, timeout)
        c.connect()
        pool.append(c)
    return pool

def close_client_pool(pool: List[SimpleClient]):
    for c in pool:
        try:
            c.close()
        except Exception:
            pass

# ---- Vector helpers ----
def generate_vector_pool(size: int, dim: int, seed: int):
    rnd = np.random.RandomState(seed)
    data = rnd.rand(size, dim).astype(np.float32)
    return [",".join(f"{x:.6f}" for x in row) for row in data]

def deterministic_tags(idx: int) -> List[str]:
    tags = []
    if idx % 2 == 0:
        tags.append("selectivity:common")
    if idx % 10 == 0:
        tags.append("selectivity:medium")
    if idx % 100 == 0:
        tags.append("selectivity:rare")
    return tags

def tags_to_sql_paren(tags: List[str]) -> str:
    if not tags:
        return ""
    return "(" + ", ".join(tags) + ")"

# ---- Inserts ----
def batch_insert_membrane(client: SimpleClient, membr: str, start_idx: int, vectors: List[str], tags_batch: List[str],
                          retries: int = 2) -> int:
    n = len(vectors)
    if n == 0:
        return 0
    same_tag = all(t == tags_batch[0] for t in tags_batch)
    success = 0
    attempt = 0
    while attempt <= retries:
        try:
            if same_tag and tags_batch[0] != "":
                tuples = [f"(k_{start_idx + j}, [{vectors[j]}])" for j in range(n)]
                sql = f"INSERT INTO {membr} VALUES " + ",".join(tuples) + f" TAGS {tags_batch[0]};\n"
                client.send_all(sql.encode())
                client.recv_response_exact()
                success = n
            elif same_tag:
                tuples = [f"(k_{start_idx + j}, [{vectors[j]}])" for j in range(n)]
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
            return success
        except (TimeoutError, ConnectionError, OSError) as e:
            attempt += 1
            time.sleep(0.05 * attempt)
            # reconnect best-effort
            try:
                client.close()
                client = SimpleClient(client.host, client.port, client.timeout)
                client.connect()
            except Exception:
                pass
            if attempt > retries:
                print("[!] insert failed after retries:", e, file=sys.stderr)
                return success
    return success

# ---- GET/DELETE micro bench ----
def test_get_delete(client: SimpleClient, membr: str, label_indices: List[int]) -> Dict:
    get_lat = []; get_ok = 0; get_fail = 0
    for idx in label_indices:
        sql = f"GET {membr} LABEL k_{idx};\n"
        t0 = time.time()
        try:
            client.send_all(sql.encode())
            resp = client.recv_response_exact()
            dt = (time.time() - t0) * 1000.0
            get_lat.append(dt)
            if b"VECTOR" in resp.upper() or b"VECTOR" in resp:
                get_ok += 1
            else:
                get_fail += 1
        except Exception:
            get_lat.append(1e4); get_fail += 1

    del_lat = []; del_ok = 0; del_fail = 0
    for idx in label_indices:
        sql = f"DELETE {membr} LABEL k_{idx};\n"
        t0 = time.time()
        try:
            client.send_all(sql.encode())
            resp = client.recv_response_exact()
            dt = (time.time() - t0) * 1000.0
            del_lat.append(dt)
            if resp.strip().upper().startswith(b"OK"):
                del_ok += 1
            else:
                del_fail += 1
        except Exception:
            del_lat.append(1e4); del_fail += 1

    get2_lat = []; get2_fail = 0
    for idx in label_indices:
        sql = f"GET {membr} LABEL k_{idx};\n"
        t0 = time.time()
        try:
            client.send_all(sql.encode())
            resp = client.recv_response_exact()
            dt = (time.time() - t0) * 1000.0
            get2_lat.append(dt)
            txt = resp.decode(errors='ignore').lower()
            if "not found" in txt or "err" in txt:
                get2_fail += 1
        except Exception:
            get2_lat.append(1e4); get2_fail += 1

    def s(xs):
        if not xs:
            return {"p50":0.0, "p90":0.0}
        a = np.array(xs)
        return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}
    return {
        "get_p50": s(get_lat)["p50"], "get_succ": get_ok, "get_fail": get_fail,
        "del_p50": s(del_lat)["p50"], "del_succ": del_ok, "del_fail": del_fail,
        "get2_p50": s(get2_lat)["p50"], "get2_fail": get2_fail
    }

# ---- Search helpers ----
def search_one(client: SimpleClient, membr: str, qcsv: str, where_clause: str = "", topk: int = 10) -> Tuple[float,int]:
    sql = f"SEARCH {membr} QUERY ([{qcsv}])"
    if where_clause:
        sql += f" WHERE {where_clause}"
    sql += f" TOP {topk};\n"
    t0 = time.time()
    client.send_all(sql.encode())
    resp = client.recv_response_exact()
    dt = (time.time() - t0) * 1000.0
    try:
        first = resp.decode(errors='ignore').splitlines()[0]
        if first.startswith("RESULTS"):
            parts = first.split()
            cnt = int(parts[1]) if len(parts) > 1 else 0
        else:
            cnt = 0
    except Exception:
        cnt = 0
    return dt, cnt

def run_concurrent_searches(client_pool: List[SimpleClient], membr: str, qvecs: List[str], where_clause: str,
                            trials: int, concurrency: int, topk: int = 10) -> Tuple[List[float], List[int]]:
    latencies = []
    counts = []
    pool_size = len(client_pool)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for i in range(trials):
            client = client_pool[i % pool_size]
            q = qvecs[i % len(qvecs)]
            futures.append(ex.submit(search_one, client, membr, q, where_clause, topk))
        for f in as_completed(futures):
            try:
                dt, cnt = f.result()
                latencies.append(dt); counts.append(cnt)
            except Exception:
                latencies.append(1e4); counts.append(0)
    return latencies, counts

# ---- Stats & plotting ----
def percentile_stats(latencies: List[float]) -> Dict:
    if not latencies:
        return {"n":0,"min":0,"max":0,"mean":0,"std":0,"p50":0,"p90":0,"p95":0,"p99":0,"p999":0}
    a = np.array(latencies)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p50": float(np.percentile(a,50)),
        "p90": float(np.percentile(a,90)),
        "p95": float(np.percentile(a,95)),
        "p99": float(np.percentile(a,99)),
        "p999": float(np.percentile(a,99.9))
    }

def plot_cdf(latencies: List[float], title: str, outpath: str):
    if not latencies: return
    a = np.sort(np.array(latencies))
    p = np.arange(1, a.size+1) / a.size
    plt.figure(figsize=(6,4))
    plt.plot(a, p, linewidth=1)
    plt.xscale('log')
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_histogram(latencies: List[float], title: str, outpath: str):
    if not latencies: return
    plt.figure(figsize=(6,4))
    a = np.array(latencies)
    # log bins to show tail
    bins = np.logspace(np.log10(max(0.001, a.min())), np.log10(max(1.0, a.max()+1e-6)), 80)
    plt.hist(a + 1e-6, bins=bins, density=False, alpha=0.8)
    plt.xscale('log')
    plt.xlabel("Latency (ms)")
    plt.ylabel("Counts")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_percentiles_vs_N(Ns, p50s, p90s, p95s, p99s, p999s, outpath):
    plt.figure(figsize=(9,5))
    plt.plot(Ns, p50s, 'o-', label='p50')
    plt.plot(Ns, p90s, 's-', label='p90')
    plt.plot(Ns, p95s, 'd-', label='p95')
    plt.plot(Ns, p99s, 'x-', label='p99')
    plt.plot(Ns, p999s, '^--', label='p999')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Dataset size (N)"); plt.ylabel("Latency (ms)")
    plt.title("Search latency percentiles vs dataset size")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_throughput_vs_N(Ns, tps, outpath):
    plt.figure(figsize=(8,4))
    plt.plot(Ns, tps, 'o-')
    plt.xscale('log'); plt.xlabel("Dataset size (N)"); plt.ylabel("Insert QPS")
    plt.title("Insertion throughput (per milestone)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_recall_vs_latency(recalls, med_latencies, outpath):
    plt.figure(figsize=(6,5))
    plt.scatter(med_latencies, recalls, c='C2', alpha=0.8)
    plt.xlabel("Median search latency (ms)"); plt.ylabel("Recall proxy (@10)")
    plt.title("Recall vs latency (per milestone)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_sys_time_series(sys_hist: Dict, outdir: str):
    if not sys_hist or not sys_hist.get("ts"):
        return
    ts = np.array(sys_hist["ts"])
    cpu = np.array(sys_hist["cpu"])
    mem = np.array(sys_hist["mem"])
    phases = sys_hist.get("phase", [""] * len(ts))

    plt.figure(figsize=(10,4))
    plt.plot(ts, cpu, label="CPU%")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU%")
    plt.title("Server CPU over time")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sys_cpu_time.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(ts, mem, label="RSS MiB", color='C1')
    plt.xlabel("Time (s)")
    plt.ylabel("RSS (MiB)")
    plt.title("Server memory RSS over time")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sys_mem_time.png"), dpi=150)
    plt.close()

# ---- Main experiment ----
def run_scientific_suite(args) -> Tuple[List[Dict], Dict]:
    monitor = SystemMonitor(pid_hint=args.server_pid, interval=0.2, host=args.host, port=args.port)
    monitor.start()

    ctl = SimpleClient(args.host, args.port, timeout=args.timeout)
    ctl.connect()

    pool_clients = make_client_pool(max(args.search_concurrency, 8), args.host, args.port, args.timeout)

    MEMBR = "pomai_research_db"
    results = []

    try:
        print("[*] Resetting membrance (DROP/CREATE)")
        try:
            ctl.send_all(f"DROP MEMBRANCE {MEMBR};\n".encode())
            try:
                ctl.recv_response_exact()
            except Exception:
                pass
        except Exception:
            pass

        ctl.send_all(f"CREATE MEMBRANCE {MEMBR} DIM {args.dim} RAM 12288;\n".encode())
        rsp = ctl.recv_response_exact()
        print("[*] Create response:", rsp.decode(errors='ignore').strip())

        current = 0
        gen_pool = generate_vector_pool(args.gen_pool_size, args.dim, args.seed)

        for target in args.milestones:
            monitor.set_phase(f"idle_{target}")
            time.sleep(0.15)
            need = target - current
            insert_qps = 0.0
            if need > 0:
                monitor.set_phase(f"insert_{target}")
                t0 = time.time()
                ptr = 0
                printed_progress = 0
                while ptr < need:
                    block = min(args.insert_batch, need - ptr)
                    vectors = [gen_pool[(current + ptr + i) % len(gen_pool)] for i in range(block)]
                    tags_batch = [tags_to_sql_paren(deterministic_tags(current + ptr + i)) for i in range(block)]

                    cnt = Counter(tags_batch)
                    top_tag, top_count = cnt.most_common(1)[0]
                    if top_count >= 0.9 * block:
                        inserted = batch_insert_membrane(ctl, MEMBR, current + ptr, vectors, [top_tag] * block)
                        ptr += inserted
                    else:
                        sub_ptr = 0
                        while sub_ptr < block:
                            sub = min(args.pipelined_batch, block - sub_ptr)
                            subv = vectors[sub_ptr:sub_ptr+sub]
                            subtags = tags_batch[sub_ptr:sub_ptr+sub]
                            ins = batch_insert_membrane(ctl, MEMBR, current + ptr + sub_ptr, subv, subtags)
                            sub_ptr += sub
                        ptr += block

                    if (current + ptr) // 50000 > printed_progress:
                        printed_progress = (current + ptr) // 50000
                        print(f"  [progress] inserted {current+ptr:,} / {target:,}")

                dur = time.time() - t0
                insert_qps = need / dur if dur > 0 else 0.0
                current = target
                print(f"[+] Inserted up to {current:,} vectors (throughput {insert_qps:.0f} vec/s)")

            # sample labels for get/delete
            sample_size = min(200, max(10, current // 1000))
            label_samples = random.sample(range(max(0, current - need), current), sample_size) if current > 0 else []

            # warmup
            monitor.set_phase(f"warmup_{target}")
            warm_qs = [gen_pool[i % len(gen_pool)] for i in range(min(200, len(gen_pool)))]
            _ = run_concurrent_searches(pool_clients, MEMBR, warm_qs, "", trials=200, concurrency=min(8, args.search_concurrency))

            # baseline searches
            monitor.set_phase(f"baseline_{target}")
            qpool = [gen_pool[i % len(gen_pool)] for i in range(min(args.search_trials, len(gen_pool)))]
            lat_base, counts = run_concurrent_searches(pool_clients, MEMBR, qpool, "", trials=min(args.search_trials, len(qpool)), concurrency=args.search_concurrency)
            stats_base = percentile_stats(lat_base)
            # use counts average as proxy for recall
            recall_proxy = float(np.mean(counts)) if counts else 0.0
            print(f"[+] Baseline N={target}: p50={stats_base['p50']:.3f}ms p99={stats_base['p99']:.3f}ms mean={stats_base['mean']:.3f}ms")

            # filtered searches
            monitor.set_phase(f"filtered_{target}")
            filters = [("selectivity:common", "common"), ("selectivity:medium", "medium"), ("selectivity:rare", "rare")]
            filter_stats = {}
            for clause, name in filters:
                where = f"{clause}"
                lat_f, cnts = run_concurrent_searches(pool_clients, MEMBR, qpool, where, trials=min(1000, len(qpool)), concurrency=args.search_concurrency)
                fs = percentile_stats(lat_f)
                fs["mean_count"] = float(np.mean(cnts)) if cnts else 0.0
                filter_stats[name] = fs
                print(f"    Filter {name}: p50={fs['p50']:.3f}ms p99={fs['p99']:.3f}ms mean_count={fs['mean_count']:.2f}")

            # get/delete tests
            monitor.set_phase(f"getdel_{target}")
            getdel_stats = {}
            if label_samples:
                getdel_stats = test_get_delete(ctl, MEMBR, label_samples)
                print(f"    GET p50={getdel_stats['get_p50']:.3f}ms succ={getdel_stats['get_succ']} fail={getdel_stats['get_fail']}")
                print(f"    DEL p50={getdel_stats['del_p50']:.3f}ms succ={getdel_stats['del_succ']} fail={getdel_stats['del_fail']}")
            else:
                getdel_stats = {"get_p50":0,"get_succ":0,"get_fail":0,"del_p50":0,"del_succ":0,"del_fail":0}

            last_cpu = monitor.history["cpu"][-1] if monitor.history["cpu"] else 0.0
            last_mem = monitor.history["mem"][-1] if monitor.history["mem"] else 0.0

            results.append({
                "vectors": target,
                "insert_qps": insert_qps,
                "baseline": stats_base,
                "recall_proxy": recall_proxy,
                "filtered": filter_stats,
                "cpu": last_cpu,
                "mem": last_mem,
                "raw_base_lat": lat_base,
                **getdel_stats
            })

    except KeyboardInterrupt:
        print("[!] Interrupted by user")
    finally:
        monitor.stop()
        close_client_pool(pool_clients)
        try: ctl.close()
        except: pass
        time.sleep(0.05)

    return results, monitor.history

# ---- Reporting ----
def generate_report(results: List[Dict], sys_hist: Dict, outdir: str, skip_plots: bool = False):
    if not results:
        print("No results to report")
        return
    os.makedirs(outdir, exist_ok=True)

    Ns = [r['vectors'] for r in results]
    p50s = [r['baseline']['p50'] for r in results]
    p90s = [r['baseline']['p90'] for r in results]
    p95s = [r['baseline']['p95'] for r in results]
    p99s = [r['baseline']['p99'] for r in results]
    p999s = [r['baseline']['p999'] for r in results]
    tps = [r['insert_qps'] for r in results]
    recalls = [r['recall_proxy'] for r in results]
    med_lat = [r['baseline']['p50'] for r in results]

    last = results[-1]
    if not skip_plots:
        plot_cdf(last['raw_base_lat'], f"Baseline CDF (N={last['vectors']})", os.path.join(outdir, "baseline_cdf_last.png"))
        plot_histogram(last['raw_base_lat'], f"Latency histogram (N={last['vectors']})", os.path.join(outdir, "latency_hist_last.png"))
        plot_percentiles_vs_N(Ns, p50s, p90s, p95s, p99s, p999s, os.path.join(outdir, "latency_percentiles_vs_N.png"))
        plot_throughput_vs_N(Ns, tps, os.path.join(outdir, "throughput_vs_N.png"))
        plot_recall_vs_latency(recalls, med_lat, os.path.join(outdir, "recall_vs_latency.png"))
        plot_sys_time_series(sys_hist, outdir)

    # CSV summary
    csvfile = os.path.join(outdir, "results_summary_v3.csv")
    with open(csvfile, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["vectors","insert_qps","p50","p90","p95","p99","p999","mean","std","cpu_pct","mem_mb","recall_proxy"]
        w.writerow(hdr)
        for r in results:
            b = r['baseline']
            row = [r['vectors'], f"{r['insert_qps']:.1f}", f"{b['p50']:.3f}", f"{b['p90']:.3f}",
                   f"{b['p95']:.3f}", f"{b['p99']:.3f}", f"{b['p999']:.3f}", f"{b['mean']:.3f}", f"{b['std']:.3f}",
                   f"{r['cpu']:.2f}", f"{r['mem']:.1f}", f"{r.get('recall_proxy',0):.3f}"]
            w.writerow(row)

    # JSON raw
    jsonfile = os.path.join(outdir, "results_raw_v3.json")
    with open(jsonfile, "w") as f:
        json.dump({"results": results, "system": sys_hist}, f, indent=2, default=lambda o: (list(o) if isinstance(o, np.ndarray) else str(o)))

    print("[+] Saved CSV:", csvfile)
    print("[+] Saved JSON:", jsonfile)
    if not skip_plots:
        print("[+] Saved plots to", outdir)

# ---- Entrypoint ----
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    print("=== POMAI SCIENTIFIC BENCHMARK v3 ===")
    print(f"Milestones: {args.milestones}, dim={args.dim}, gen_pool={args.gen_pool_size}, insert_batch={args.insert_batch}")
    res, sys_hist = run_scientific_suite(args)
    if res:
        generate_report(res, sys_hist, args.output, skip_plots=args.skip_plots)
        print("Done. Results saved to", args.output)
    else:
        print("No results collected.")

if __name__ == "__main__":
    main()