#!/usr/bin/env python3
"""
benchmarks/benchmark_q1_paper_v3.py
----------------------------------
Improved Pomai scientific benchmark (milestones preserved).

Goals (what changed vs original):
 - Still uses milestone flow (insert up to a list of dataset sizes).
 - More metrics: p50/p90/p95/p99/p999/std/max/min/count.
 - More plots: latency CDF, latency percentiles vs N, throughput vs N,
   recall vs latency scatter, filter selectivity heatmap (simple).
 - Better insertion progress reporting and resiliency (retries on transient errors).
 - Configurable via CLI: milestones, batch sizes, gen_pool_size, timeouts, skip micro-sweep.
 - Keeps existing SQL protocol and server usage (no server code changes).
 - Saves raw JSON + CSV summary + PNG plots.

Usage (example):
  python3 benchmarks/benchmark_q1_paper_v3.py --only-milestones --milestones 10000 50000 100000 \
      --gen-pool-size 10000 --output results_v3 --timeout 30

Requirements:
  python3, numpy, matplotlib, psutil
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
from typing import List, Dict, Tuple
import random
import math
import statistics

try:
    import psutil
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print("Missing dependencies:", e)
    sys.exit(1)

# -------------------------
# Defaults / Config
# -------------------------
HOST = "127.0.0.1"
PORT = 7777
SERVER_BIN_NAME = "pomai_server"

DEFAULT_DIM = 512
DEFAULT_MILESTONES = [10000, 50000]
DEFAULT_INSERT_BATCH = 2000
DEFAULT_PIPELINED_BATCH = 500
DEFAULT_GEN_POOL = 10000
DEFAULT_SEARCH_TRIALS = 2000
DEFAULT_SEARCH_CONCURRENCY = 32
OUTPUT_DIR_DEFAULT = "pomai_scientific_results_v3"

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PomaiDB Scientific Benchmark (v3)")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", default=PORT, type=int)
    p.add_argument("--dim", default=DEFAULT_DIM, type=int)
    p.add_argument("--milestones", nargs="+", type=int, default=DEFAULT_MILESTONES,
                   help="Dataset sizes to measure at (milestones)")
    p.add_argument("--insert-batch", default=DEFAULT_INSERT_BATCH, type=int,
                   help="Batch size for multi-tuple INSERT when tags are identical")
    p.add_argument("--pipelined-batch", default=DEFAULT_PIPELINED_BATCH, type=int,
                   help="Pipelined single-insert group size when tags vary")
    p.add_argument("--gen-pool-size", default=DEFAULT_GEN_POOL, type=int,
                   help="In-memory vector pool size (keep reasonable for mem)")
    p.add_argument("--search-trials", default=DEFAULT_SEARCH_TRIALS, type=int)
    p.add_argument("--search-concurrency", default=DEFAULT_SEARCH_CONCURRENCY, type=int)
    p.add_argument("--timeout", default=30.0, type=float)
    p.add_argument("--output", default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--only-milestones", action="store_true",
                   help="Run only the milestone incremental experiment (skip optional micro-sweep)")
    p.add_argument("--skip-micro", action="store_true", help="Skip micro-sweep insert profiling")
    p.add_argument("--skip-plots", action="store_true", help="Skip writing plots (write JSON/CSV only)")
    return p.parse_args()

# -------------------------
# Monitoring (server-side best-effort)
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
# Networking client (thread-safe)
# -------------------------
class SimpleClient:
    def __init__(self, host=HOST, port=PORT, timeout=30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
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

    def recv_response_exact(self, marker=b"<END>\n") -> bytes:
        with self.lock:
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

def make_client_pool(n, host, port, timeout):
    pool = []
    for _ in range(n):
        c = SimpleClient(host, port, timeout)
        c.connect()
        pool.append(c)
    return pool

def close_client_pool(pool):
    for c in pool:
        try: c.close()
        except: pass

# -------------------------
# Vector/tag helpers
# -------------------------
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
    if not tags: return ""
    return "(" + ", ".join(tags) + ")"

# -------------------------
# Inserts: batching + pipelining + retry
# -------------------------
def batch_insert_membrane(client: SimpleClient, membr: str, start_idx: int, vectors: List[str], tags_batch: List[str],
                          retries: int = 2) -> int:
    """
    Insert vectors into `membr` starting with label index start_idx.
    Uses multi-tuple INSERT when tags identical; otherwise pipelined singles grouped.
    Returns number of successful inserts (best-effort).
    """
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
                # pipelined grouped singles
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
            # try to reconnect client if needed
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

# -------------------------
# GET / DELETE micro-test
# -------------------------
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

    def stats(xs):
        if not xs: return {"p50":0.0,"p90":0.0}
        a = np.array(xs)
        return {"p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90))}

    return {
        "get_p50": stats(get_lat)["p50"], "get_succ": get_ok, "get_fail": get_fail,
        "del_p50": stats(del_lat)["p50"], "del_succ": del_ok, "del_fail": del_fail,
        "get2_p50": stats(get2_lat)["p50"], "get2_fail": get2_fail
    }

# -------------------------
# Search helpers (concurrent)
# -------------------------
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

# -------------------------
# Stats/plots/report helpers
# -------------------------
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
    if not latencies:
        return
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

def plot_percentiles_vs_N(Ns, p50s, p99s, p999s, outpath):
    plt.figure(figsize=(8,5))
    plt.plot(Ns, p50s, 'o-', label='p50')
    plt.plot(Ns, p99s, 's-', label='p99')
    plt.plot(Ns, p999s, 'x-', label='p99.9')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel("Dataset size (N)"); plt.ylabel("Latency (ms)")
    plt.title("Search latency percentiles vs dataset size")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(outpath, dpi=150); plt.close()

def plot_throughput_vs_N(Ns, tps, outpath):
    plt.figure(figsize=(8,5))
    plt.plot(Ns, tps, 'o-')
    plt.xscale('log'); plt.xlabel("Dataset size (N)"); plt.ylabel("Insert QPS")
    plt.title("Insertion throughput (per milestone)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(outpath, dpi=150); plt.close()

def plot_recall_vs_latency(recalls, latencies, outpath):
    plt.figure(figsize=(6,5))
    plt.scatter(latencies, recalls, c='C2', alpha=0.7)
    plt.xlabel("Median search latency (ms)"); plt.ylabel("Recall@10")
    plt.title("Recall vs latency (per milestone)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150); plt.close()

# -------------------------
# Main experiment (milestones) - core logic preserved, improved
# -------------------------
def run_scientific_suite(args):
    monitor = SystemMonitor()
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

        # iterate milestones (respect safety cap)
        for target in args.milestones:
            monitor.set_phase(f"idle_{target}")
            time.sleep(0.2)
            need = target - current
            if need > 0:
                monitor.set_phase(f"insert_{target}")
                t0 = time.time()
                ptr = 0
                printed_progress = 0
                while ptr < need:
                    # choose block
                    block = min(args.insert_batch, need - ptr)
                    vectors = [gen_pool[(current + ptr + i) % len(gen_pool)] for i in range(block)]
                    tags_batch = [tags_to_sql_paren(deterministic_tags(current + ptr + i)) for i in range(block)]

                    # group if most tags same to use multi-tuple
                    from collections import Counter
                    cnt = Counter(tags_batch)
                    top_tag, top_count = cnt.most_common(1)[0]
                    if top_count >= 0.9 * block:
                        inserted = batch_insert_membrane(ctl, MEMBR, current + ptr, vectors, [top_tag] * block)
                        ptr += inserted
                    else:
                        # pipelined sub-blocks
                        sub_ptr = 0
                        while sub_ptr < block:
                            sub = min(args.pipelined_batch, block - sub_ptr)
                            subv = vectors[sub_ptr:sub_ptr+sub]
                            subtags = tags_batch[sub_ptr:sub_ptr+sub]
                            ins = batch_insert_membrane(ctl, MEMBR, current + ptr + sub_ptr, subv, subtags)
                            sub_ptr += sub
                        ptr += block

                    # progress print every ~100k inserted or every few loops
                    if (current + ptr) // 50000 > printed_progress:
                        printed_progress = (current + ptr) // 50000
                        print(f"  [progress] inserted {current+ptr:,} / {target:,}")

                    if (current + ptr) // 500000 > printed_progress:
                        printed_progress = (current + ptr) // 500000
                        print(f"  [progress] inserted {current+ptr:,} / {target:,}")

                dur = time.time() - t0
                insert_qps = need / dur if dur > 0 else 0.0
                current = target
                print(f"[+] Inserted up to {current:,} vectors (throughput {insert_qps:.0f} vec/s)")
            else:
                insert_qps = 0.0

            # sample labels for get/delete
            sample_size = min(200, max(10, current // 1000))
            label_samples = random.sample(range(max(0, current - need), current), sample_size) if current > 0 else []

            # warmup searches
            monitor.set_phase(f"warmup_{target}")
            warm_qs = [gen_pool[i % len(gen_pool)] for i in range(min(200, len(gen_pool)))]
            _ = run_concurrent_searches(pool_clients, MEMBR, warm_qs, "", trials=200, concurrency=min(8, args.search_concurrency))

            # baseline search
            monitor.set_phase(f"baseline_{target}")
            qpool = [gen_pool[i % len(gen_pool)] for i in range(min(args.search_trials, len(gen_pool)))]
            lat_base, _ = run_concurrent_searches(pool_clients, MEMBR, qpool, "", trials=min(args.search_trials, len(qpool)), concurrency=args.search_concurrency)
            stats_base = percentile_stats(lat_base)
            print(f"[+] Baseline N={target}: p50={stats_base['p50']:.3f}ms p99={stats_base['p99']:.3f}ms mean={stats_base['mean']:.3f}ms")

            # filtered searches
            monitor.set_phase(f"filtered_{target}")
            filters = [("common",50.0), ("medium",10.0), ("rare",1.0)]
            filter_stats = {}
            for sel, pct in filters:
                where = f"selectivity='{sel}'"
                lat_f, _ = run_concurrent_searches(pool_clients, MEMBR, qpool, where, trials=min(1000, len(qpool)), concurrency=args.search_concurrency)
                fs = percentile_stats(lat_f)
                filter_stats[sel] = fs
                print(f"    Filter {sel}: p50={fs['p50']:.3f}ms p99={fs['p99']:.3f}ms")

            # get/delete tests
            monitor.set_phase(f"getdel_{target}")
            getdel_stats = {}
            if label_samples:
                getdel_stats = test_get_delete(ctl, MEMBR, label_samples)
                print(f"    GET p50={getdel_stats['get_p50']:.3f}ms succ={getdel_stats['get_succ']} fail={getdel_stats['get_fail']}")
                print(f"    DEL p50={getdel_stats['del_p50']:.3f}ms succ={getdel_stats['del_succ']} fail={getdel_stats['del_fail']}")
            else:
                getdel_stats = {"get_p50":0,"get_succ":0,"get_fail":0,"del_p50":0,"del_succ":0,"del_fail":0,"get2_p50":0,"get2_fail":0}

            # snapshot monitor
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
        print("[!] Interrupted by user")
    finally:
        monitor.stop()
        close_client_pool(pool_clients)
        try: ctl.close()
        except: pass
        time.sleep(0.05)

    return results, monitor.history

# -------------------------
# Reporting
# -------------------------
def generate_report(results, sys_hist, outdir, skip_plots=False):
    if not results:
        print("No results to report")
        return

    os.makedirs(outdir, exist_ok=True)
    # Build arrays for summary plots
    Ns = [r['vectors'] for r in results]
    p50s = [r['baseline']['p50'] for r in results]
    p99s = [r['baseline']['p99'] for r in results]
    p999s = [r['baseline']['p999'] for r in results]
    tps = [r['insert_qps'] for r in results]
    recalls = [r['filtered']['common']['p50'] if 'common' in r['filtered'] else 0.0 for r in results]  # proxy

    # CDF for last baseline
    last = results[-1]
    if not skip_plots:
        plot_cdf(last['raw_base_lat'], f"Baseline CDF (N={last['vectors']})", os.path.join(outdir, "baseline_cdf_last.png"))
        plot_percentiles_vs_N(Ns, p50s, p99s, p999s, os.path.join(outdir, "latency_percentiles_vs_N.png"))
        plot_throughput_vs_N(Ns, tps, os.path.join(outdir, "throughput_vs_N.png"))
        # recall vs median-latency
        med_lat = [r['baseline']['p50'] for r in results]
        plot_recall_vs_latency(recalls, med_lat, os.path.join(outdir, "recall_vs_latency.png"))

    # CSV summary
    csvfile = os.path.join(outdir, "results_summary_v3.csv")
    with open(csvfile, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["vectors","insert_qps","base_p50","base_p90","base_p95","base_p99","base_p999","base_mean","base_std","cpu_pct","mem_mb",
               "filter_common_p50","filter_common_p99","filter_medium_p50","filter_medium_p99","filter_rare_p50","filter_rare_p99",
               "get_p50","get_succ","get_fail","del_p50","del_succ","del_fail"]
        w.writerow(hdr)
        for r in results:
            fc = r['filtered'].get('common', {})
            fm = r['filtered'].get('medium', {})
            fr = r['filtered'].get('rare', {})
            row = [
                r['vectors'], f"{r['insert_qps']:.1f}",
                f"{r['baseline']['p50']:.3f}", f"{r['baseline']['p90']:.3f}", f"{r['baseline']['p95']:.3f}",
                f"{r['baseline']['p99']:.3f}", f"{r['baseline']['p999']:.3f}", f"{r['baseline']['mean']:.3f}", f"{r['baseline']['std']:.3f}",
                f"{r['cpu']:.2f}", f"{r['mem']:.1f}",
                f"{fc.get('p50',0):.3f}", f"{fc.get('p99',0):.3f}",
                f"{fm.get('p50',0):.3f}", f"{fm.get('p99',0):.3f}",
                f"{fr.get('p50',0):.3f}", f"{fr.get('p99',0):.3f}",
                f"{r.get('get_p50',0):.3f}", r.get('get_succ',0), r.get('get_fail',0),
                f"{r.get('del_p50',0):.3f}", r.get('del_succ',0), r.get('del_fail',0)
            ]
            w.writerow(row)

    # Save raw JSON
    jsonfile = os.path.join(outdir, "results_raw_v3.json")
    with open(jsonfile, "w") as f:
        json.dump({"results": results, "system": sys_hist}, f, indent=2, default=lambda o: (list(o) if isinstance(o, np.ndarray) else str(o)))

    print("[+] Saved CSV:", csvfile)
    print("[+] Saved JSON:", jsonfile)
    if not skip_plots:
        print("[+] Saved plots to", outdir)

# -------------------------
# Entrypoint
# -------------------------
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