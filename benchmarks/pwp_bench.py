#!/usr/bin/env python3
import socket
import struct
import time
import os
import multiprocessing
import random
import argparse

# --- CẤU HÌNH GIAO THỨC POMAI (PWP) ---
# Header: Magic(1) + Op(1) + Status(2) + KLen(4) + VLen(4) + Reserved(4) = 16 Bytes
# Use network (big-endian) byte order for integers so server sees correct values.
HEADER_FMT = "!BBHIII"
MAGIC = 0x50  # 'P'
OP_GET = 1
OP_SET = 2

def recv_exact(sock, n):
    """Receive exactly n bytes or raise."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed while reading")
        data.extend(chunk)
    return bytes(data)

def run_worker(worker_id, num_ops, host, port, payload_size):
    """A worker process: performs num_ops SET then num_ops GET"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((host, port))
        
        key_prefix = f"k:{worker_id}:"
        value_data = "X" * payload_size
        v_bytes = value_data.encode('utf-8')
        v_len = len(v_bytes)

        # SET phase
        start_write = time.time()
        for i in range(num_ops):
            key = f"{key_prefix}{i}".encode('utf-8')
            k_len = len(key)

            header = struct.pack(HEADER_FMT, MAGIC, OP_SET, 0, k_len, v_len, 0)
            sock.sendall(header + key + v_bytes)

            # read response header (16 bytes)
            resp = recv_exact(sock, 16)
            # we don't use the response fields for SET here
        end_write = time.time()

        # GET phase
        start_read = time.time()
        for i in range(num_ops):
            key = f"{key_prefix}{i}".encode('utf-8')
            k_len = len(key)
            header = struct.pack(HEADER_FMT, MAGIC, OP_GET, 0, k_len, 0, 0)
            sock.sendall(header + key)

            # read response header
            resp_hdr = recv_exact(sock, 16)
            _, _, status, _, r_vlen, _ = struct.unpack(HEADER_FMT, resp_hdr)

            # recv body if present
            if r_vlen > 0:
                _ = recv_exact(sock, r_vlen)
        end_read = time.time()

        sock.close()
        return (num_ops, end_write - start_write, end_read - start_read)

    except Exception as e:
        # Print to help debugging; return zeros so main can continue
        print(f"Worker {worker_id} died: {e}")
        return (0, 0.0, 0.0)

def main():
    parser = argparse.ArgumentParser(description="Pomai Cache Benchmark")
    parser.add_argument("-c", "--concurrency", type=int, default=4, help="Number of client worker processes")
    parser.add_argument("-n", "--requests", type=int, default=100000, help="Total number of SET (and GET) per run")
    parser.add_argument("-d", "--data_size", type=int, default=32, help="Value size in bytes")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=6379, help="Server port")
    args = parser.parse_args()

    if args.requests < args.concurrency:
        print("Total requests should be >= concurrency; adjusting.")
        args.requests = args.concurrency

    ops_per_worker = args.requests // args.concurrency

    print(f"--- POMAI BENCHMARK ---")
    print(f"Workers: {args.concurrency}")
    print(f"Total SET ops: {args.requests} (each worker will do {ops_per_worker} SETs and {ops_per_worker} GETs)")
    print(f"Payload: {args.data_size} bytes")
    print(f"Server: {args.host}:{args.port}")
    print("Running...")

    start_global = time.time()
    with multiprocessing.Pool(processes=args.concurrency) as pool:
        results = pool.starmap(run_worker, [
            (i, ops_per_worker, args.host, args.port, args.data_size)
            for i in range(args.concurrency)
        ])
    end_global = time.time()

    total_ops = sum(r[0] for r in results) * 2  # SET + GET
    total_time = end_global - start_global

    write_ops = sum(r[0] for r in results)
    total_write_time = sum(r[1] for r in results)
    total_read_time = sum(r[2] for r in results)

    avg_write_time = (total_write_time / args.concurrency) if total_write_time > 0 else 0.0
    avg_read_time = (total_read_time / args.concurrency) if total_read_time > 0 else 0.0

    print(f"\n--- RESULTS ---")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {int(total_ops / total_time) if total_time > 0 else 0:,} ops/sec (SET+GET)")
    if avg_write_time > 0:
        print(f"Write Speed (per worker avg): ~{int(write_ops / avg_write_time):,} ops/sec")
    else:
        print("Write Speed: N/A")
    if avg_read_time > 0:
        print(f"Read Speed  (per worker avg): ~{int(write_ops / avg_read_time):,} ops/sec")
    else:
        print("Read Speed: N/A")

if __name__ == "__main__":
    main()