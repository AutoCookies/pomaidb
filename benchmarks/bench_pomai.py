#!/usr/bin/env python3
import argparse
import socket
import struct
import time
import random
import statistics
import math
from array import array

# ---- Binary protocol Constants ----
OP_PING = 1
OP_CREATE_COLLECTION = 2
OP_UPSERT_BATCH = 3
OP_SEARCH = 4

METRIC_L2 = 0
METRIC_COSINE = 1

# ---- Helper Functions ----

def percentile(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c: return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)

def frame(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload

def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: raise RuntimeError("server closed connection")
        buf.extend(chunk)
    return bytes(buf)

def recv_frame(sock: socket.socket) -> bytes:
    header = recv_exact(sock, 4)
    ln = struct.unpack("<I", header)[0]
    if ln == 0: return b""
    return recv_exact(sock, ln)

def put_string_u16(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<H", len(b)) + b

# ---- Protocol Requests ----

def req_ping(sock) -> None:
    sock.sendall(frame(struct.pack("<B", OP_PING)))
    resp = recv_frame(sock)
    if not resp or resp[0] != 1: raise RuntimeError("PING failed")

def req_create_collection(sock, name: str, dim: int, metric: str, shards: int) -> int:
    m = METRIC_COSINE if metric == "cosine" else METRIC_L2
    payload = bytearray()
    payload += struct.pack("<B", OP_CREATE_COLLECTION)
    payload += put_string_u16(name)
    payload += struct.pack("<H", dim)
    payload += struct.pack("<B", m)
    payload += struct.pack("<I", shards)
    payload += struct.pack("<I", 4096)
    sock.sendall(frame(payload))
    resp = recv_frame(sock)
    if not resp or resp[0] != 1: raise RuntimeError("CREATE failed")
    return struct.unpack_from("<I", resp, 1)[0]

def req_upsert_batch(sock, col_id: int, dim: int, ids, vec_f32: array) -> int:
    n = len(ids)
    payload = bytearray()
    payload += struct.pack("<B", OP_UPSERT_BATCH)
    payload += struct.pack("<I", col_id)
    payload += struct.pack("<I", n)
    payload += struct.pack("<H", dim)
    payload += struct.pack("<" + "Q" * n, *ids)
    payload += vec_f32.tobytes()
    sock.sendall(frame(payload))
    resp = recv_frame(sock)
    if not resp or resp[0] != 1: raise RuntimeError("UPSERT failed")
    return struct.unpack_from("<Q", resp, 1)[0]

def req_search(sock, col_id: int, dim: int, topk: int, q_f32: array):
    payload = bytearray()
    payload += struct.pack("<B", OP_SEARCH)
    payload += struct.pack("<I", col_id)
    payload += struct.pack("<I", topk)
    payload += struct.pack("<H", dim)
    payload += q_f32.tobytes()
    sock.sendall(frame(payload))
    resp = recv_frame(sock)
    if not resp or resp[0] != 1: raise RuntimeError("SEARCH failed")
    
    count = struct.unpack_from("<I", resp, 1)[0]
    off = 5
    ids = list(struct.unpack_from("<" + "Q" * count, resp, off))
    return ids

# ---- Ground Truth Calculation (Pure Python) ----

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2)) # Đã fix lỗi typo ở đây
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)

def l2_distance_sq(v1, v2):
    return sum((a - b) ** 2 for a, b in zip(v1, v2))

def compute_ground_truth(dataset, queries, topk, metric):
    print(f"[Client] Calculating Ground Truth for {len(queries)} queries (be patient)...")
    gt_results = []
    for q_vec in queries:
        scores = []
        for doc_id, doc_vec in dataset.items():
            if metric == "cosine":
                s = cosine_similarity(q_vec, doc_vec)
                scores.append((s, doc_id)) 
            else:
                s = l2_distance_sq(q_vec, doc_vec)
                scores.append((-s, doc_id))
        scores.sort(key=lambda x: x[0], reverse=True)
        top_ids = set(x[1] for x in scores[:topk])
        gt_results.append(top_ids)
    return gt_results

# ---- Main Benchmark ----

def main():
    ap = argparse.ArgumentParser(description="Pomai Stress Test with Recall")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7744) # Đã sửa default port thành 7744
    ap.add_argument("--vectors", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    print(f":: POMAI BENCHMARK ::")
    print(f"   Target: {args.host}:{args.port}")
    print(f"   Dataset: {args.vectors} vectors, Dim {args.dim}")
    print("-" * 50)

    try:
        sock = socket.create_connection((args.host, args.port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        print(f"[Error] Cannot connect: {e}")
        return

    # 1. Generate Data
    print("[Client] Generating random data...")
    rng = random.Random(42)
    local_dataset = {} 
    upsert_buffer_ids = []
    upsert_buffer_vecs = array("f")
    
    col_id = req_create_collection(sock, "stress_test", args.dim, "cosine", 4)
    start_time = time.perf_counter()
    
    for i in range(args.vectors):
        vec = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
        
        # Chỉ lưu 20k vector đầu để làm GT cho nhanh, nhưng query vẫn search trên full DB
        if i < 20000 + args.queries:
             local_dataset[i] = vec
             
        upsert_buffer_ids.append(i)
        upsert_buffer_vecs.extend(vec)
        
        if len(upsert_buffer_ids) >= args.batch:
            req_upsert_batch(sock, col_id, args.dim, upsert_buffer_ids, upsert_buffer_vecs)
            upsert_buffer_ids = []
            upsert_buffer_vecs = array("f")
            
    if upsert_buffer_ids:
        req_upsert_batch(sock, col_id, args.dim, upsert_buffer_ids, upsert_buffer_vecs)
        
    duration = time.perf_counter() - start_time
    print(f"[Server] Ingested {args.vectors} vectors in {duration:.2f}s ({args.vectors/duration:.1f} vec/s)")
    
    # 2. GT & Queries
    gt_dataset = local_dataset 
    query_vectors = []
    for _ in range(args.queries):
        query_vectors.append([rng.uniform(-1.0, 1.0) for _ in range(args.dim)])
        
    gt_ids = compute_ground_truth(gt_dataset, query_vectors, args.topk, "cosine")
    
    print("-" * 50)
    print(f"[Server] Warming up (Waiting for Index Build)...")
    time.sleep(2.0) 
    
    # 3. Benchmark
    latencies = []
    recalls = []
    
    print(f"[Server] Running {args.queries} queries...")
    t_start = time.perf_counter()
    
    for i, q_vec in enumerate(query_vectors):
        q_arr = array("f", q_vec)
        t0 = time.perf_counter()
        result_ids = req_search(sock, col_id, args.dim, args.topk, q_arr)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        true_set = gt_ids[i]
        found_set = set(result_ids)
        intersect = true_set.intersection(found_set)
        recalls.append(len(intersect) / args.topk)
        
    t_total = time.perf_counter() - t_start
    
    # 4. Report
    avg_lat = statistics.mean(latencies)
    avg_recall = statistics.mean(recalls) * 100.0
    throughput = args.queries / t_total
    
    print("-" * 50)
    print(":: RESULTS ::")
    print(f"Throughput : {throughput:,.1f} QPS")
    print(f"Latency    : Avg={avg_lat:.2f}ms | P95={percentile(latencies, 95):.2f}ms | P99={percentile(latencies, 99):.2f}ms")
    print(f"Recall@{args.topk} : {avg_recall:.2f}%")
    print("-" * 50)
    sock.close()

if __name__ == "__main__":
    main()