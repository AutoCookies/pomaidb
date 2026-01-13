#!/usr/bin/env python3
"""
benchmarks/benchmark_q1_paper.py
--------------------------------
POMAIDB SCIENTIFIC BENCHMARK SUITE (Q1 PAPER GRADE)
--------------------------------
Mục tiêu: Tạo ra bộ dữ liệu và biểu đồ đạt chuẩn công bố khoa học (IEEE/ACM).

Các chỉ số đo lường (Metrics):
1. Throughput (QPS): Scalability theo kích thước dữ liệu.
2. Latency Profile: P50, P95, P99, P99.9 (Tail Latency), StdDev.
3. Resource Efficiency: CPU/RAM time-series correlation.
4. Latency Distribution: CDF (Cumulative Distribution Function).

Yêu cầu: python3-numpy, python3-matplotlib, python3-psutil
"""

import socket
import time
import random
import threading
import csv
import os
import sys
import statistics
import datetime
from typing import List, Dict, Tuple

try:
    import psutil
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"ERROR: Thiếu thư viện khoa học. Hãy cài đặt: {e.name}")
    print("Run: sudo apt install python3-numpy python3-matplotlib python3-psutil")
    sys.exit(1)

# --- CONFIGURATION ---
HOST = "127.0.0.1"
PORT = 7777
SERVER_BIN_NAME = "pomai_server" # Tên process chính xác của server

# Experimental Parameters
DIM = 512
WARMUP_VECTORS = 2000
MILESTONES = [10000, 50000, 100000, 200000, 500000, 1000000] # Tăng lên nếu máy mạnh: [100k, 500k, 1M]
SEARCH_TRIALS = 5000  # Số lần search để lấy mẫu thống kê (N > 1000 cho định lý giới hạn trung tâm)
CONCURRENCY = 1       # Single-thread client để đo Raw Latency chính xác nhất (Latency-focus)

# Output
OUTPUT_DIR = "pomai_scientific_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HIGH-PRECISION SYSTEM MONITOR ---
class SystemMonitor(threading.Thread):
    """
    Thu thập dữ liệu hệ thống với độ phân giải cao (High-resolution Telemetry)
    Chạy song song với Benchmark để tương quan giữa Load và Resource.
    """
    def __init__(self, target_process_name, interval=0.1):
        super().__init__()
        self.target_name = target_process_name
        self.interval = interval
        self.stop_event = threading.Event()
        self.history = {
            "timestamp": [],
            "cpu_percent": [],
            "ram_mb": [],
            "phase": [] # Ghi chú giai đoạn (Insert/Search)
        }
        self.current_phase = "Init"
        self.process = None
        self._find_process()

    def _find_process(self):
        for proc in psutil.process_iter(['pid', 'name']):
            if self.target_name in proc.info['name']:
                self.process = proc
                print(f"[SystemMonitor] Attached to PID {proc.info['pid']}")
                return
        print(f"[SystemMonitor] WARNING: Process '{self.target_name}' not found. RAM/CPU will be 0.")

    def set_phase(self, phase_name):
        self.current_phase = phase_name

    def run(self):
        start_time = time.time()
        while not self.stop_event.is_set():
            now = time.time() - start_time
            cpu = 0.0
            mem = 0.0
            if self.process:
                try:
                    # with_children=True để tính cả các thread con
                    cpu = self.process.cpu_percent(interval=None) 
                    mem = self.process.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self.process = None # Process died?

            self.history["timestamp"].append(now)
            self.history["cpu_percent"].append(cpu)
            self.history["ram_mb"].append(mem)
            self.history["phase"].append(self.current_phase)
            
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

# --- OPTIMIZED SQL CLIENT ---
class ScientificClient:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # Tắt Nagle's alg cho low latency
        self.sock.settimeout(60.0)
        self.host = host
        self.port = port
        self.recv_buf = bytearray()

    def connect(self):
        try:
            self.sock.connect((self.host, self.port))
            return True
        except OSError as e:
            print(f"[Client] Connection failed: {e}")
            return False

    def close(self):
        self.sock.close()

    def send_raw(self, data: bytes):
        self.sock.sendall(data)

    def recv_response_exact(self):
        """Đọc chính xác một phản hồi (dựa vào marker <END>)"""
        marker = b"<END>\n"
        while True:
            if marker in self.recv_buf:
                idx = self.recv_buf.find(marker)
                resp = self.recv_buf[:idx]
                self.recv_buf = self.recv_buf[idx+len(marker):]
                return resp
            
            chunk = self.sock.recv(65536)
            if not chunk: raise ConnectionError("Socket closed unexpected")
            self.recv_buf.extend(chunk)

# --- EXPERIMENT LOGIC ---

def generate_vector_pool(size, dim):
    """Sinh vector trước để không tốn CPU lúc đo"""
    print(f"[Generator] Pre-generating {size} vectors (Dim={dim})...")
    # Sử dụng numpy để sinh nhanh nếu có, fallback về list comprehension
    try:
        data = np.random.rand(size, dim).astype(np.float32)
        # Convert to string format for SQL
        return [",".join(map(lambda x: f"{x:.4f}", row)) for row in data]
    except:
        return [",".join(f"{random.random():.4f}" for _ in range(dim)) for _ in range(size)]

def calculate_statistics(latencies_ms: List[float]) -> Dict:
    """Tính toán các chỉ số thống kê nâng cao"""
    arr = np.array(latencies_ms)
    return {
        "min": np.min(arr),
        "max": np.max(arr),
        "avg": np.mean(arr),
        "std_dev": np.std(arr),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
        "p99_9": np.percentile(arr, 99.9), # Tail latency quan trọng
        "samples": len(arr)
    }

def run_scientific_suite():
    monitor = SystemMonitor(SERVER_BIN_NAME)
    monitor.start()
    
    client = ScientificClient(HOST, PORT)
    if not client.connect():
        monitor.stop()
        return None

    MEMBR = "pomai_research_db"
    results = [] # List of dicts storing metrics per milestone

    try:
        # 1. INIT
        print("=== [PHASE 0] Initialization & Warmup ===")
        client.send_raw(f"DROP MEMBRANCE {MEMBR};\n".encode())
        client.recv_response_exact() # Ignore err if not exists
        
        # Cấp phát bộ nhớ lớn để tránh realloc trong quá trình đo
        client.send_raw(f"CREATE MEMBRANCE {MEMBR} DIM {DIM} RAM 2048;\n".encode())
        print(client.recv_response_exact().decode())

        # Warmup: Nạp một ít dữ liệu để nóng cache CPU/OS
        monitor.set_phase("Warmup")
        w_pool = generate_vector_pool(WARMUP_VECTORS, DIM)
        for i, v in enumerate(w_pool):
            client.send_raw(f"INSERT INTO {MEMBR} VALUES (w_{i}, [{v}]);\n".encode())
            client.recv_response_exact()
        print("-> Warmup complete.")

        current_count = WARMUP_VECTORS
        
        # 2. MAIN LOOP
        query_pool = generate_vector_pool(100, DIM) # 100 queries mẫu tái sử dụng

        for target in MILESTONES:
            monitor.set_phase(f"Idle_{target}")
            time.sleep(2.0) # Cool-down giữa các mốc để CPU hạ nhiệt, ổn định baseline

            needed = target - current_count
            if needed > 0:
                print(f"\n=== [MILESTONE] Scaling to {target:,} vectors (Adding {needed:,}) ===")
                monitor.set_phase(f"Insert_{target}")
                
                # --- INSERT MEASUREMENT (Throughput Focus) ---
                # Dùng batch lớn để đo max throughput của server
                batch_pool = generate_vector_pool(min(needed, 5000), DIM)
                pool_len = len(batch_pool)
                
                t_start = time.time()
                batch_size = 200 # Pipeline depth
                
                for i in range(0, needed, batch_size):
                    cmds = []
                    this_batch = min(batch_size, needed - i)
                    for j in range(this_batch):
                        idx = current_count + i + j
                        vec = batch_pool[idx % pool_len]
                        cmds.append(f"INSERT INTO {MEMBR} VALUES (k_{idx}, [{vec}]);")
                    
                    payload = "\n".join(cmds) + "\n"
                    client.send_raw(payload.encode())
                    
                    # Consume responses
                    for _ in range(this_batch):
                        client.recv_response_exact()
                
                dur = time.time() - t_start
                insert_qps = needed / dur
                print(f"-> Insert Done. Throughput: {insert_qps:.2f} vectors/s")
                current_count = target
            
            # --- SEARCH MEASUREMENT (Latency Focus) ---
            print(f"=== [MEASURE] Searching @ {target:,} vectors ===")
            monitor.set_phase(f"Search_{target}")
            
            latencies = []
            
            # Đo từng query một (Ping-Pong) để lấy chính xác Latency phía Client
            # Không dùng Pipelining ở đây vì muốn đo độ trễ thực tế của từng request.
            for k in range(SEARCH_TRIALS):
                q_vec = query_pool[k % 100]
                cmd = f"SEARCH {MEMBR} QUERY ([{q_vec}]) TOP 10;\n"
                
                t0 = time.time()
                client.send_raw(cmd.encode())
                resp = client.recv_response_exact()
                t1 = time.time()
                
                latencies.append((t1 - t0) * 1000.0) # ms
            
            stats = calculate_statistics(latencies)
            print(f"-> Latency Profile: Avg={stats['avg']:.2f}ms | P99={stats['p99']:.2f}ms | Tail(P99.9)={stats['p99_9']:.2f}ms")
            
            # Snapshot Resource usage tại thời điểm này
            # Lấy trung bình RAM trong giai đoạn search vừa rồi
            idx_start = -1
            try:
                # Tìm index trong history bắt đầu phase search hiện tại
                rev_phases = list(reversed(monitor.history["phase"]))
                offset = rev_phases.index(f"Search_{target}")
                idx_start = len(monitor.history["phase"]) - 1 - offset
            except ValueError:
                idx_start = -1
            
            ram_avg = 0
            if idx_start != -1:
                segment = monitor.history["ram_mb"][idx_start:]
                if segment: ram_avg = np.mean(segment)

            results.append({
                "vectors": target,
                "insert_qps": insert_qps if needed > 0 else 0,
                "search_qps": SEARCH_TRIALS / (sum(latencies)/1000.0), # Derived QPS from sequential latency
                "stats": stats,
                "ram_mb": ram_avg,
                "raw_latencies": latencies # Lưu lại để vẽ Boxplot/CDF
            })

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as e:
        print(f"\n[!] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[TEARDOWN] Stopping monitor and closing connection...")
        monitor.stop()
        monitor.join()
        try:
            client.send_raw(f"DROP MEMBRANCE {MEMBR};\n".encode())
            client.close()
        except: pass

    return results, monitor.history

# --- SCIENTIFIC PLOTTING (MATPLOTLIB) ---

def generate_report(res_data, sys_data):
    print(f"\n=== Generating Scientific Report in '{OUTPUT_DIR}' ===")
    
    # Style configuration for Paper (IEEE style ish)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    # Data prep
    vectors = [r['vectors'] for r in res_data]
    
    # ---------------------------------------------------------
    # FIG 1: Scalability & Throughput (QPS vs Data Size)
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title("Fig 1. Search Scalability Analysis")
    ax1.set_xlabel("Database Size (Vectors)")
    ax1.set_ylabel("Search Latency (ms)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Latencies
    p50 = [r['stats']['p50'] for r in res_data]
    p99 = [r['stats']['p99'] for r in res_data]
    p999 = [r['stats']['p99_9'] for r in res_data]

    l1, = ax1.plot(vectors, p50, 'o-', color='#2ca02c', label='P50 (Median)')
    l2, = ax1.plot(vectors, p99, 's-', color='#1f77b4', label='P99 Latency')
    l3, = ax1.plot(vectors, p999, '^--', color='#d62728', label='P99.9 (Tail)')

    ax1.legend(handles=[l1, l2, l3], loc='upper left')
    
    # Secondary Axis for QPS
    ax2 = ax1.twinx()
    search_qps = [r['search_qps'] for r in res_data]
    l4, = ax2.plot(vectors, search_qps, 'x:', color='#7f7f7f', label='Est. Sequential QPS', alpha=0.7)
    ax2.set_ylabel("Sequential Throughput (ops/s)")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_scalability.png", dpi=300)
    print("-> Generated Fig 1: Scalability")

    # ---------------------------------------------------------
    # FIG 2: Latency Distribution (Box Plot)
    # Shows the stability and jitter at each milestone
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Fig 2. Latency Distribution per Milestone")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Dataset Size")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    raw_data = [r['raw_latencies'] for r in res_data]
    ax.boxplot(raw_data, labels=[f"{v//1000}k" for v in vectors], showfliers=False) # Hide extreme outliers for clearer view
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_latency_boxplot.png", dpi=300)
    print("-> Generated Fig 2: Latency Boxplot")

    # ---------------------------------------------------------
    # FIG 3: Cumulative Distribution Function (CDF)
    # The most important chart for performance engineers
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Fig 3. Search Latency CDF (Tail Analysis)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Plot CDF for the largest dataset
    largest_run = res_data[-1]
    sorted_lat = np.sort(largest_run['raw_latencies'])
    yvals = np.arange(len(sorted_lat)) / float(len(sorted_lat) - 1)
    
    ax.plot(sorted_lat, yvals, label=f"N={largest_run['vectors']:,}", linewidth=2)
    # Zoom in to the "knee" (90% - 100%)
    ax.set_xlim(left=0, right=largest_run['stats']['p99_9'] * 1.2) 
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_latency_cdf.png", dpi=300)
    print("-> Generated Fig 3: Latency CDF")

    # ---------------------------------------------------------
    # FIG 4: System Resource Timeline
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title("Fig 4. System Resource Usage Timeline")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("CPU Usage (%)", color='tab:blue')
    
    t = sys_data['timestamp']
    cpu = sys_data['cpu_percent']
    ram = sys_data['ram_mb']
    
    ax1.fill_between(t, cpu, color='tab:blue', alpha=0.3)
    ax1.plot(t, cpu, color='tab:blue', linewidth=1, label="CPU %")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, max(100, max(cpu)*1.2))

    ax2 = ax1.twinx()
    ax2.set_ylabel("RAM Usage (MB)", color='tab:orange')
    ax2.plot(t, ram, color='tab:orange', linewidth=2, label="RAM MB")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Annotate Phases
    # Simple logic: find where phase changes
    phases = sys_data['phase']
    last_p = ""
    for i, p in enumerate(phases):
        if p != last_p and p != "Init":
            plt.axvline(x=t[i], color='k', linestyle=':', alpha=0.5)
            # Only label major phases to avoid clutter
            if "Insert" in p or "Search" in p:
                y_pos = ax2.get_ylim()[1] * 0.95
                ax1.text(t[i], 80, p.split('_')[0], rotation=90, verticalalignment='center', fontsize=8, alpha=0.7)
        last_p = p

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_resources.png", dpi=300)
    print("-> Generated Fig 4: Resource Timeline")

    # ---------------------------------------------------------
    # CSV DUMP (Raw Data for Peer Review)
    # ---------------------------------------------------------
    csv_path = f"{OUTPUT_DIR}/pomai_metrics_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Vectors", "Insert_QPS", "Search_Avg_Lat", "Search_P50", "Search_P99", "Search_P99.9", "RAM_MB"])
        for r in res_data:
            writer.writerow([
                r['vectors'], 
                f"{r['insert_qps']:.2f}", 
                f"{r['stats']['avg']:.3f}",
                f"{r['stats']['p50']:.3f}",
                f"{r['stats']['p99']:.3f}",
                f"{r['stats']['p99_9']:.3f}",
                f"{r['ram_mb']:.1f}"
            ])
    print(f"-> Saved Metrics Table: {csv_path}")

if __name__ == "__main__":
    print(f"--- STARTING SCIENTIFIC BENCHMARK SUITE ---")
    print(f"Target: {HOST}:{PORT}")
    print(f"Milestones: {MILESTONES}")
    
    res, sys_mon = run_scientific_suite()
    
    if res and len(res) > 0:
        generate_report(res, sys_mon)
        print("\n[SUCCESS] Benchmark complete. Report generated.")
    else:
        print("\n[FAILURE] No results generated.")