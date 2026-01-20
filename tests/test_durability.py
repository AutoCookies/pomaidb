#!/usr/bin/env python3
"""
tests/benchmark_checkpoint.py
Script benchmark hiệu năng Insert, Checkpoint và đo lường dung lượng đĩa.
"""

import socket
import time
import random
import sys
import os
import argparse

# --- CONFIG ---
HOST = '127.0.0.1'
PORT = 7777
DB_NAME = 'bench_db'
DIM = 128
BATCH_SIZE = 100       # Insert 100 vectors per command
WAL_PATH = "./data/orbit.wal" # Đường dẫn file WAL mặc định

def get_file_size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0

def send_cmd(sock, cmd):
    if not cmd.endswith('\n'): cmd += '\n'
    sock.sendall(cmd.encode('utf-8'))
    
    # Đọc response đơn giản (đến khi gặp newline hoặc timeout)
    data = b""
    while True:
        try:
            chunk = sock.recv(4096)
            if not chunk: break
            data += chunk
            if b'\n' in chunk: break 
        except socket.timeout:
            break
    return data.decode('utf-8', errors='ignore').strip()

def mode_ingest(count):
    print(f"\n[BENCHMARK] Target: {count} vectors, Dim: {DIM}")
    print(f"[BENCHMARK] Connecting to {HOST}:{PORT}...")
    
    try:
        sock = socket.create_connection((HOST, PORT), timeout=30)
    except Exception as e:
        print(f"Error connecting: {e}")
        sys.exit(1)

    # 1. Clean & Create DB
    send_cmd(sock, f"DROP MEMBRANCE {DB_NAME};")
    print(f"[1] Creating membrance '{DB_NAME}'...")
    # Tăng RAM lên 512MB cho test lớn
    resp = send_cmd(sock, f"CREATE MEMBRANCE {DB_NAME} DIM {DIM} DATA_TYPE float32 RAM 512;")
    print(f"    -> {resp}")

    # 2. Insert Loop
    print(f"[2] Inserting {count} vectors (Batch size {BATCH_SIZE})...")
    
    # Tạo mẫu vector random (dùng chung để tiết kiệm CPU client)
    sample_vec = ",".join([f"{random.random():.4f}" for _ in range(DIM)])
    
    insert_start = time.time()
    
    for i in range(0, count, BATCH_SIZE):
        batch_entries = []
        for j in range(BATCH_SIZE):
            idx = i + j
            if idx >= count: break
            # Format: (id, [v1,v2...])
            batch_entries.append(f"({idx}, [{sample_vec}])")
        
        if not batch_entries: break
        
        # Build SQL INSERT
        values_str = ",".join(batch_entries)
        sql = f"INSERT INTO {DB_NAME} VALUES {values_str};"
        
        send_cmd(sock, sql)
        
        if (i + BATCH_SIZE) % 5000 == 0:
            sys.stdout.write(f"    -> Inserted {i + BATCH_SIZE}/{count}...\r")
            sys.stdout.flush()

    insert_end = time.time()
    duration = insert_end - insert_start
    print(f"\n    -> Insert DONE in {duration:.2f}s")
    if duration > 0:
        print(f"    -> Throughput: {count / duration:.0f} vectors/sec")

    # 3. Check WAL Size BEFORE Checkpoint
    time.sleep(1) # Chờ OS flush file stat
    wal_size_before = get_file_size_mb(WAL_PATH)
    print(f"\n[3] WAL Size BEFORE Checkpoint: {wal_size_before:.2f} MB")
    
    # 4. Execute Checkpoint
    print(f"[4] Executing EXEC CHECKPOINT (Flush RAM -> Disk, Truncate WAL)...")
    cp_start = time.time()
    resp = send_cmd(sock, "EXEC CHECKPOINT;")
    cp_end = time.time()
    print(f"    -> Response: {resp}")
    print(f"    -> Checkpoint Duration: {cp_end - cp_start:.4f}s")

    # 5. Check WAL Size AFTER Checkpoint
    time.sleep(0.5)
    wal_size_after = get_file_size_mb(WAL_PATH)
    print(f"[5] WAL Size AFTER Checkpoint:  {wal_size_after:.2f} MB")

    if wal_size_after < 1.0:
        print(f"\n[PASS] WAL truncated successfully (Saved {wal_size_before - wal_size_after:.2f} MB disk space).")
    else:
        print(f"\n[WARN] WAL still seems large ({wal_size_after:.2f} MB). Check logic.")

    sock.close()

def mode_verify(count):
    print(f"\n[VERIFY] Checking data integrity after restart...")
    try:
        # Retry connect
        for i in range(10):
            try:
                sock = socket.create_connection((HOST, PORT), timeout=5)
                break
            except:
                time.sleep(0.5)
        else:
            print("Server not reachable.")
            sys.exit(1)
    except Exception as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    # 1. Check Info (Total Vectors)
    print("    -> Checking Membrance Info...")
    resp = send_cmd(sock, f"GET MEMBRANCE INFO {DB_NAME};")
    
    # Simple check for total count
    expected_str = f"total_vectors: {count}"
    if expected_str in resp or f"total_vectors:{count}" in resp.replace(" ", ""):
         print(f"    -> INFO Check OK: Server reports {count} vectors.")
    else:
         print(f"    -> INFO Check WARNING: Could not find '{expected_str}' in response.")
         print(f"Response:\n{resp}")

    # 2. Search Probe
    print("    -> Checking Search (Random Access)...")
    dummy_vec = ",".join(["0.5"] * DIM)
    start = time.time()
    resp = send_cmd(sock, f"SEARCH {DB_NAME} QUERY ({dummy_vec}) TOP 1;")
    end = time.time()
    
    if "OK" in resp:
        print(f"    -> SEARCH Latency: {(end-start)*1000:.2f} ms")
        print("[PASS] Data is accessible and index is working.")
    else:
        print(f"[FAIL] Search error: {resp}")
        sys.exit(1)
    
    sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ingest', 'verify'], required=True)
    parser.add_argument('--count', type=int, default=50000)
    args = parser.parse_args()

    if args.mode == 'ingest':
        mode_ingest(args.count)
    else:
        mode_verify(args.count)