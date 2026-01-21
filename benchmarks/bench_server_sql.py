#!/usr/bin/env python3
"""
bench_server_sql_multi.py

Extended benchmarking driver to run insert experiments across multiple
"milestone" sizes (e.g. 5k, 100k, 1M, 100M) and collect latency / throughput /
and resource (CPU / RSS) measurements for each milestone. Produces CSVs and
plots suitable for inclusion in a research-style report.

Notes & caveats:
 - Inserting 100M vectors over TCP will take a very long time and may be
   impractical on a single machine. Use this script cautiously for very large
   milestones; prefer using distributed ingestion, faster batch APIs, or
   synthetic summaries (i.e. extrapolation) for extremely large counts.
 - The script can launch a local server (--run-server). If you prefer to
   benchmark an existing server, run without --run-server and point --host/--port.
 - Resource monitoring requires `psutil`. Install with `pip install psutil`.
 - The server SQL protocol must accept batch INSERTs in the form:
     INSERT INTO <name> VALUES (label,[v...]),(...),...;
   This script uses that form to reduce per-vector RTT overhead.

Usage examples:
  # Run three milestones locally (server binary available)
  python3 bench_server_sql_multi.py --run-server --server-bin ./build/pomai_server \
      --milestones 5000 100000 1000000 --concurrency 8 --batch-size 16

  # Run against remote server, collect resource stats by PID lookup
  python3 bench_server_sql_multi.py --host 10.0.0.2 --port 7777 --milestones 5000 50000

If no --milestones are provided the script will use sensible defaults so you can
run it immediately without specifying flags.

Author: Generated helper
"""

from __future__ import annotations

import argparse
import socket
import time
import os
import sys
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple
from tqdm import tqdm

try:
    import psutil
except Exception:
    psutil = None

import numpy as np
import pandas as pd

# Force a headless non-GUI backend to avoid tkinter/GUI errors when plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.max_open_warning": 0})

# -------------------------
# Defaults
# -------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7777
DEFAULT_RUN_SERVER = False
DEFAULT_SERVER_BIN = "./build/pomai_server"
DEFAULT_OUTDIR = Path("bench_multi_results") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
DEFAULT_DIM = 512
DEFAULT_CONCURRENCY = 8
DEFAULT_BATCH_SIZE = 8           # number of vectors per batch SQL statement
DEFAULT_MONITOR_INTERVAL = 0.5   # seconds
SOCKET_TIMEOUT = 10.0

# sensible default milestones so script is runnable without flags
DEFAULT_MILESTONES = [5000, 100000, 1000000]

# -------------------------
# SQL socket client
# -------------------------
class SQLClient:
    def __init__(self, host: str, port: int, timeout: float = SOCKET_TIMEOUT):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.lock = threading.Lock()

    def connect(self, retries: int = 5, backoff: float = 0.2):
        for attempt in range(retries):
            try:
                s = socket.create_connection((self.host, self.port), timeout=self.timeout)
                s.settimeout(self.timeout)
                self.sock = s
                return
            except Exception as e:
                if attempt + 1 == retries:
                    raise
                time.sleep(backoff * (1 + 0.5 * attempt))
        raise RuntimeError("connect failed")

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        self.sock = None

    def send_sql(self, sql: str) -> str:
        """
        Send SQL (must end with ';') and read until "<END>\n".
        Returns response without trailing marker.
        """
        if not sql.endswith(";"):
            sql = sql + ";"
        payload = sql.encode("utf-8")
        with self.lock:
            self.sock.sendall(payload)
            buf = bytearray()
            while True:
                chunk = self.sock.recv(8192)
                if not chunk:
                    break
                buf.extend(chunk)
                if b"<END>\n" in buf:
                    break
        text = buf.decode("utf-8", errors="replace")
        if "<END>\n" in text:
            text = text.replace("<END>\n", "")
        return text

# -------------------------
# Resource monitoring
# -------------------------
def monitor_process(pid: int, interval_s: float, stop_event: threading.Event, out_list: List[Dict]):
    if psutil is None:
        print("[monitor] psutil not installed; skipping resource monitoring")
        return
    try:
        p = psutil.Process(pid)
    except Exception as e:
        print(f"[monitor] cannot monitor pid {pid}: {e}")
        return

    # prime CPU percent
    try:
        p.cpu_percent(interval=None)
    except Exception:
        pass

    while not stop_event.is_set():
        ts = time.time()
        try:
            cpu = p.cpu_percent(interval=None)
            mem = p.memory_info().rss
            # optionally capture threads, open files, io counters
            io = {}
            try:
                ioc = p.io_counters()
                io = {"read_bytes": getattr(ioc, "read_bytes", 0), "write_bytes": getattr(ioc, "write_bytes", 0)}
            except Exception:
                io = {"read_bytes": 0, "write_bytes": 0}

            out_list.append({"ts": ts, "cpu_percent": cpu, "mem_rss": mem, **io})
        except Exception:
            out_list.append({"ts": ts, "cpu_percent": 0.0, "mem_rss": 0, "read_bytes": 0, "write_bytes": 0})
        stop_event.wait(interval_s)

# -------------------------
# vector utils
# -------------------------
def gen_vector(dim: int, seed_hint: int = 0) -> List[float]:
    # deterministic simple vector (avoid RNG overhead): small ramp + seed offset
    base = (seed_hint % 100) / 100.0
    return [base + 0.1 * i for i in range(dim)]

def vec_to_sqbr(vec: List[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"

# -------------------------
# batch insert worker
# -------------------------
def batch_insert_worker(host: str, port: int, membr: str, dim: int,
                        labels_slice: List[int], batch_size: int,
                        lat_per_op: List[float], ts_per_op: List[float],
                        result_ok: List[Tuple[int, bool]], worker_id: int):
    """
    Each worker keeps a persistent connection and issues batch INSERT statements.
    It slices labels_slice into contiguous batches of batch_size and sends them.
    We attribute each vector in a batch the per-op latency = batch_latency / batch_count.
    """
    client = SQLClient(host, port)
    try:
        client.connect()
    except Exception as e:
        print(f"[worker {worker_id}] connect failed: {e}")
        for lbl in labels_slice:
            result_ok.append((lbl, False))
        return

    try:
        # process labels in groups
        idx = 0
        total = len(labels_slice)
        while idx < total:
            take = min(batch_size, total - idx)
            batch_labels = labels_slice[idx: idx + take]
            # build multi-row VALUES clause
            values_parts = []
            for lbl in batch_labels:
                v = gen_vector(dim, seed_hint=lbl)
                values_parts.append(f"({lbl}, {vec_to_sqbr(v)})")
            sql = f"INSERT INTO {membr} VALUES " + ",".join(values_parts) + ";"
            t0 = time.time()
            try:
                resp = client.send_sql(sql)
                t1 = time.time()
                batch_lat_ms = (t1 - t0) * 1000.0
                per_op = batch_lat_ms / take
                ts = t0
                for lbl in batch_labels:
                    lat_per_op.append(per_op)
                    ts_per_op.append(ts)
                    ok = ("OK: inserted" in resp) or ("OK" in resp and "ERR" not in resp)
                    result_ok.append((lbl, ok))
            except Exception as e:
                # mark all as failed
                t1 = time.time()
                per_op = float("nan")
                ts = t0
                for lbl in batch_labels:
                    lat_per_op.append(per_op)
                    ts_per_op.append(ts)
                    result_ok.append((lbl, False))
                print(f"[worker {worker_id}] batch insert failed: {e}")
            idx += take
    finally:
        client.close()

# -------------------------
# orchestrator per milestone
# -------------------------
def run_milestone(host: str, port: int, membr: str, dim: int,
                  target_count: int, concurrency: int, batch_size: int,
                  monitor_pid: int | None, monitor_interval: float,
                  outdir: Path):
    """
    Inserts target_count vectors (labels 1..target_count) across concurrent workers.
    Returns summary dict and paths to saved CSVs/plots.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[milestone] start inserting {target_count} vectors (concurrency={concurrency}, batch={batch_size})")

    # prepare label slices
    labels = list(range(1, target_count + 1))
    # partition evenly
    chunks = [[] for _ in range(concurrency)]
    for i, lbl in enumerate(labels):
        chunks[i % concurrency].append(lbl)

    # shared collectors (thread-safe append semantics)
    lat_per_op: List[float] = []
    ts_per_op: List[float] = []
    result_ok: List[Tuple[int, bool]] = []

    # start monitor if pid provided
    monitor_series: List[Dict] = []
    stop_mon = threading.Event()
    mon_thread = None
    if monitor_pid and psutil is not None:
        mon_thread = threading.Thread(target=monitor_process, args=(monitor_pid, monitor_interval, stop_mon, monitor_series))
        mon_thread.start()
        print(f"[milestone] started resource monitor for pid {monitor_pid}")

    # run workers
    start_wall = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = []
        for wid in range(concurrency):
            if not chunks[wid]:
                continue
            futures.append(ex.submit(batch_insert_worker, host, port, membr, dim,
                                     chunks[wid], batch_size,
                                     lat_per_op, ts_per_op, result_ok, wid))
        for fut in tqdm(as_completed(futures), total=len(futures), desc="workers"):
            try:
                fut.result()
            except Exception as e:
                print("worker exception:", e)
    end_wall = time.time()

    # stop monitor
    if mon_thread:
        stop_mon.set()
        mon_thread.join(timeout=5.0)

    # summarize
    # filter NaN latencies out of numeric summary
    numeric_lats = [x for x in lat_per_op if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    summary = {
        "count_requested": target_count,
        "count_attempted": len(lat_per_op),
        "count_successful": sum(1 for _, ok in result_ok if ok),
        "wall_time_s": end_wall - start_wall,
        "throughput_ops_per_s": (sum(1 for _, ok in result_ok if ok) / max(1e-9, (end_wall - start_wall))),
        "latency": {}  # filled below
    }
    if numeric_lats:
        arr = np.array(numeric_lats)
        summary["latency"] = {
            "count": int(arr.size),
            "mean_ms": float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p90_ms": float(np.percentile(arr, 90)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "std_ms": float(arr.std(ddof=1) if arr.size > 1 else 0.0)
        }
    else:
        summary["latency"] = {}

    # Save per-op CSV
    df_ops = pd.DataFrame({"label": [lbl for lbl, _ in result_ok],
                           "ok": [ok for _, ok in result_ok],
                           "lat_ms": lat_per_op,
                           "ts": ts_per_op})
    ops_csv = outdir / f"inserts_{target_count}_ops.csv"
    df_ops.to_csv(ops_csv, index=False)

    # Save monitor series
    monitor_csv = None
    if monitor_series:
        monitor_csv = outdir / f"monitor_{target_count}.csv"
        pd.DataFrame(monitor_series).to_csv(monitor_csv, index=False)

    # Plots: latency CDF, hist, throughput time series (1s buckets), resource timeseries
    if numeric_lats:
        lat_cdf_png = outdir / f"latency_cdf_{target_count}.png"
        arr_sorted = np.sort(np.array(numeric_lats))
        p = 100.0 * np.arange(1, arr_sorted.size + 1) / arr_sorted.size
        plt.figure(figsize=(6,4))
        plt.plot(arr_sorted, p, linewidth=2)
        plt.xlabel("Latency (ms)")
        plt.ylabel("Cumulative %")
        plt.title(f"Insert Latency CDF (N={target_count})")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(lat_cdf_png)
        plt.close()

        lat_hist_png = outdir / f"latency_hist_{target_count}.png"
        plt.figure(figsize=(6,4))
        plt.hist(arr_sorted, bins=100, color='tab:blue', alpha=0.8)
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count")
        plt.title(f"Insert Latency Histogram (N={target_count})")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(lat_hist_png)
        plt.close()
    else:
        lat_cdf_png = None
        lat_hist_png = None

    # throughput time series
    if ts_per_op:
        t0 = min(ts_per_op)
        last = max(ts_per_op)
        window = 1.0
        bins = int(((last - t0) / window) + 3)
        counts, edges = np.histogram(ts_per_op, bins=bins, range=(t0 - 0.5, last + 0.5))
        centers = (edges[:-1] + edges[1:]) / 2.0 - t0
        rates = counts / window
        thr_png = outdir / f"throughput_{target_count}.png"
        plt.figure(figsize=(8,3))
        plt.plot(centers, rates, drawstyle='steps-mid')
        plt.xlabel("Time since start (s)")
        plt.ylabel("Throughput (ops/sec)")
        plt.title(f"Insert Throughput (N={target_count})")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(thr_png)
        plt.close()
    else:
        thr_png = None

    # resource timeseries plot
    res_png = None
    if monitor_series:
        dfm = pd.DataFrame(monitor_series)
        t0m = dfm['ts'].min()
        dfm['t_rel'] = dfm['ts'] - t0m
        res_png = outdir / f"resources_{target_count}.png"
        fig, ax1 = plt.subplots(figsize=(8,3))
        ax1.plot(dfm['t_rel'], dfm['cpu_percent'], color='tab:orange', label='CPU %')
        ax1.set_ylabel('CPU %', color='tab:orange')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(dfm['t_rel'], dfm['mem_rss'] / (1024.0*1024.0), color='tab:blue', label='RSS MB')
        ax2.set_ylabel('RSS (MB)', color='tab:blue')
        plt.title(f"Server CPU% and RSS (N={target_count})")
        fig.tight_layout()
        plt.savefig(res_png)
        plt.close()

    # Save summary JSON
    summary_path = outdir / f"summary_{target_count}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary": summary,
        "ops_csv": str(ops_csv),
        "monitor_csv": str(monitor_csv) if monitor_csv else None,
        "lat_cdf_png": str(lat_cdf_png) if lat_cdf_png else None,
        "lat_hist_png": str(lat_hist_png) if lat_hist_png else None,
        "throughput_png": str(thr_png) if thr_png else None,
        "resources_png": str(res_png) if res_png else None,
        "summary_json": str(summary_path)
    }

# -------------------------
# helpers: find server pid by port
# -------------------------
def find_pid_by_port(port: int) -> int:
    if psutil is None:
        return 0
    for p in psutil.process_iter(["pid", "name", "connections"]):
        try:
            for c in p.connections(kind='inet'):
                if c.laddr and c.laddr.port == port:
                    return p.pid
        except Exception:
            continue
    return 0

# -------------------------
# script entrypoint
# -------------------------
def main(argv):
    parser = argparse.ArgumentParser(description="Multi-milestone server SQL benchmark")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--run-server", action="store_true", default=DEFAULT_RUN_SERVER)
    parser.add_argument("--server-bin", default=DEFAULT_SERVER_BIN)
    parser.add_argument("--data-root", default=None, help="data root for server when launching")
    parser.add_argument("--milestones", nargs="+", type=int, required=False,
                        help="List of milestone insertion counts (e.g. --milestones 5000 100000 1000000). "
                             "If omitted the script will use sensible defaults.")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--monitor-interval", type=float, default=DEFAULT_MONITOR_INTERVAL)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    args = parser.parse_args(argv)

    # Use default milestones if none provided on the CLI
    if not args.milestones:
        print(f"[main] no --milestones provided; using defaults: {DEFAULT_MILESTONES}")
        args.milestones = DEFAULT_MILESTONES

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": args.host,
        "port": args.port,
        "milestones": args.milestones,
        "dim": args.dim,
        "concurrency": args.concurrency,
        "batch_size": args.batch_size,
        "run_server": args.run_server,
        "server_bin": args.server_bin
    }
    with open(outdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    server_proc = None
    data_root = args.data_root
    if args.run_server:
        if not os.path.exists(args.server_bin):
            print("[error] server binary not found:", args.server_bin)
            sys.exit(2)
        if data_root is None:
            data_root = tempfile.mkdtemp(prefix="pomai_bench_")
        print("[main] launching server:", args.server_bin, "data_root=", data_root)
        # Launch server as background process; logs saved in data_root
        env = os.environ.copy()
        env["POMAI_DB_DIR"] = str(data_root)
        env["POMAI_PORT"] = str(args.port)
        outlog = open(os.path.join(data_root, "server.stdout.log"), "wb")
        errlog = open(os.path.join(data_root, "server.stderr.log"), "wb")
        server_proc = subprocess.Popen([args.server_bin], env=env, stdout=outlog, stderr=errlog)
        # wait for the socket to be ready
        deadline = time.time() + 30.0
        while time.time() < deadline:
            try:
                s = socket.create_connection((args.host, args.port), timeout=1.0)
                s.close()
                break
            except Exception:
                time.sleep(0.2)
        else:
            print("[error] server did not become ready in time")
            if server_proc:
                server_proc.terminate()
            sys.exit(3)

    # detect server pid for monitoring
    server_pid = server_proc.pid if server_proc else find_pid_by_port(args.port)
    if server_pid:
        print("[main] monitoring server pid:", server_pid)
    else:
        print("[main] server pid unknown; resource monitoring will be skipped (psutil required)")

    # Create a reusable membrance for the experiments. We'll reuse the same membrance
    # name but it's safe to drop and recreate between milestones if desired.
    membr_name = "bench_multi_membr"
    # connect client and create membrance
    cli = SQLClient(args.host, args.port)
    cli.connect()
    print("[main] creating membrance", membr_name, "dim", args.dim)
    resp = cli.send_sql(f"CREATE MEMBRANCE {membr_name} DIM {args.dim};")
    print("[main] CREATE ->", (resp.splitlines()[0] if resp else "<empty>"))
    cli.close()

    all_results = {}
    # iterate milestones. For each milestone we will:
    #  - if required, drop & recreate membrance to have clean state (optional)
    #  - perform inserts up to target count
    #  - collect metrics
    cumulative = 0
    for m in args.milestones:
        if m <= cumulative:
            print(f"[warn] milestone {m} <= previous cumulative {cumulative}; skipping")
            continue
        target_for_run = m
        # Optionally, for very large milestones we could skip re-inserting already inserted vectors.
        # For simplicity this script inserts labels [1..m] each run (so for larger m it repeats previous work).
        # For long experiments you may prefer incremental insertion (only insert from cumulative+1 .. m).
        print(f"\n=== Running milestone: {m} vectors ===")
        milestone_dir = outdir / f"milestone_{m}"
        milestone_dir.mkdir(parents=True, exist_ok=True)

        # For large runs it's better to insert only the delta (from cumulative+1 to m)
        start_label = cumulative + 1
        end_label = m
        count_to_do = end_label - start_label + 1
        if count_to_do <= 0:
            print("[main] nothing to do for this milestone")
            cumulative = m
            continue

        # For simplicity we will instruct workers with labels start..end
        # Build label list
        labels = list(range(start_label, end_label + 1))
        # Partition labels into worker chunks
        chunks = [[] for _ in range(args.concurrency)]
        for i, lbl in enumerate(labels):
            chunks[i % args.concurrency].append(lbl)

        # prepare shared collectors
        lat_all = []
        ts_all = []
        results_all = []

        # start monitor
        monitor_series = []
        stop_mon = threading.Event()
        mon_thread = None
        if server_pid and psutil is not None:
            mon_thread = threading.Thread(target=monitor_process, args=(server_pid, args.monitor_interval, stop_mon, monitor_series))
            mon_thread.start()

        # start workers
        print(f"[main] inserting labels {start_label}..{end_label} (count={count_to_do})")
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = []
            for wid in range(args.concurrency):
                if not chunks[wid]:
                    continue
                futures.append(ex.submit(batch_insert_worker, args.host, args.port, membr_name, args.dim,
                                         chunks[wid], args.batch_size,
                                         lat_all, ts_all, results_all, wid))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="milestone workers"):
                try:
                    fut.result()
                except Exception as e:
                    print("[main] worker exception:", e)

        # stop monitor
        if mon_thread:
            stop_mon.set()
            mon_thread.join(timeout=5.0)

        # save per-milestone results
        df = pd.DataFrame({"label": [lbl for lbl, _ in results_all],
                           "ok": [ok for _, ok in results_all],
                           "lat_ms": lat_all,
                           "ts": ts_all})
        df.to_csv(milestone_dir / "ops.csv", index=False)

        # resource series
        if monitor_series:
            pd.DataFrame(monitor_series).to_csv(milestone_dir / "monitor.csv", index=False)

        # compute summary
        numeric_lats = [x for x in lat_all if isinstance(x, (float, int)) and not (np.isnan(x) or np.isinf(x))]
        summary = {
            "milestone": m,
            "requested": count_to_do,
            "attempted": len(lat_all),
            "successful": int(sum(1 for _, ok in results_all if ok)),
            "wall_time_s": None,
            "throughput_ops_per_s": None,
            "lat_summary": {}
        }
        if ts_all:
            t0 = min(ts_all)
            t1 = max(ts_all)
            wall = t1 - t0 if t1 > t0 else 0.0
            summary["wall_time_s"] = wall
            summary["throughput_ops_per_s"] = (summary["successful"] / wall) if wall > 0 else None
        if numeric_lats:
            arr = np.array(numeric_lats)
            summary["lat_summary"] = {
                "count": int(arr.size),
                "mean_ms": float(arr.mean()),
                "median_ms": float(np.median(arr)),
                "p90_ms": float(np.percentile(arr, 90)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
                "std_ms": float(arr.std(ddof=1) if arr.size > 1 else 0.0)
            }
        # Save JSON summary
        with open(milestone_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        all_results[m] = {"dir": str(milestone_dir), "summary": summary}

        # Create basic plots for milestone (latency CDF, throughput timeline)
        if numeric_lats:
            arr_sorted = np.sort(np.array(numeric_lats))
            plt.figure(figsize=(6,4))
            plt.plot(arr_sorted, 100.0 * np.arange(1, arr_sorted.size + 1) / arr_sorted.size, linewidth=2)
            plt.xlabel("Latency (ms)")
            plt.ylabel("Cumulative %")
            plt.title(f"Insert Latency CDF (N={m})")
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(milestone_dir / "latency_cdf.png")
            plt.close()

        if ts_all:
            t0 = min(ts_all)
            last = max(ts_all)
            window = 1.0
            bins = int(((last - t0) / window) + 3)
            counts, edges = np.histogram(ts_all, bins=bins, range=(t0 - 0.5, last + 0.5))
            centers = (edges[:-1] + edges[1:]) / 2.0 - t0
            rates = counts / window
            plt.figure(figsize=(8,3))
            plt.plot(centers, rates, drawstyle='steps-mid')
            plt.xlabel("Time since start (s)")
            plt.ylabel("Throughput (ops/sec)")
            plt.title(f"Insert Throughput (N={m})")
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(milestone_dir / "throughput.png")
            plt.close()

        if monitor_series:
            dfm = pd.DataFrame(monitor_series)
            t0m = dfm['ts'].min()
            dfm['t_rel'] = dfm['ts'] - t0m
            plt.figure(figsize=(8,3))
            plt.plot(dfm['t_rel'], dfm['cpu_percent'], color='tab:orange', label='CPU %')
            plt.ylabel('CPU %')
            plt.xlabel('Time (s)')
            ax2 = plt.twinx()
            ax2.plot(dfm['t_rel'], dfm['mem_rss'] / (1024.0*1024.0), color='tab:blue', label='RSS MB')
            ax2.set_ylabel('RSS MB')
            plt.title(f"Resources during insertion N={m}")
            plt.tight_layout()
            plt.savefig(milestone_dir / "resources.png")
            plt.close()

        # advance cumulative
        cumulative = m

    # final cleanup: drop membrance
    try:
        cli2 = SQLClient(args.host, args.port)
        cli2.connect()
        drop = cli2.send_sql(f"DROP MEMBRANCE {membr_name};")
        print("[main] DROP ->", (drop.splitlines()[0] if drop else "<empty>"))
        cli2.close()
    except Exception:
        pass

    # shutdown server if started
    if server_proc:
        print("[main] terminating server process...")   
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5.0)
        except Exception:
            server_proc.kill()

    # Save aggregated overview
    with open(outdir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("[main] done. results directory:", outdir)


if __name__ == "__main__":
    main(sys.argv[1:])