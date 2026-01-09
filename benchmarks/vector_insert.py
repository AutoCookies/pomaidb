#!/usr/bin/env python3
"""
bench_insert.py -- High-performance bulk insert via VSET PWP.
Optimized with Persistent Connections and Multi-threading.
"""
import argparse
import socket
import struct
import time
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

PWP_MAGIC = ord('P')
OP_VSET = 10
OP_VMEM = 12

def read_exact(sock, n):
    """Helper to read exactly n bytes from socket"""
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket connection broken")
        buf += chunk
    return buf

def float_list_to_bytes(values):
    return b''.join(struct.pack('f', float(x)) for x in values)

def insert_worker(thread_id, host, port, start_idx, count, dim, batch_size, timeout):
    """Worker thread: opens ONE connection and inserts 'count' vectors."""
    errors = 0
    t_start = time.time()
    
    try:
        # 1. Open persistent connection
        with socket.create_connection((host, port), timeout=timeout) as s:
            # Disable Nagle's algorithm for lower latency
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            for i in range(count):
                global_idx = start_idx + i
                key = f"vec_{global_idx}".encode('utf-8')
                
                # Generate random vector (simulate float32)
                # Optimization: Generate bytes directly if possible, but struct pack is fast enough
                vec = [random.random() for _ in range(dim)]
                vec_bytes = float_list_to_bytes(vec)

                # 2. Send header + body
                hdr = struct.pack("!BBHIII", PWP_MAGIC, OP_VSET, 0, len(key), len(vec_bytes), 0)
                s.sendall(hdr + key + vec_bytes)

                # 3. Read response (Synchronous but persistent)
                resp = read_exact(s, 16)
                magic, op, status, rklen, rvlen, reserved = struct.unpack("!BBHIII", resp)
                
                if magic != PWP_MAGIC:
                    raise RuntimeError("bad magic")
                if status != 0:
                    raise RuntimeError(f"server status {status}")

                # Progress reporting (per thread)
                if (i + 1) % batch_size == 0:
                    # Optional: print progress per thread or stay silent to reduce I/O lock
                    pass
                    
    except Exception as e:
        errors += 1
        print(f"[Thread-{thread_id}] Error: {e}", file=sys.stderr)
        # In a real stress test, we might want to reconnect and continue, 
        # but for now we exit the thread on connection error.
    
    duration = time.time() - t_start
    return count, errors, duration

def request_memusage(host, port, timeout=5):
    """Single request to get memory usage"""
    hdr = struct.pack("!BBHIII", PWP_MAGIC, OP_VMEM, 0, 0, 0, 0)
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(hdr)
        resp = read_exact(s, 16)
        magic, op, status, rklen, rvlen, reserved = struct.unpack("!BBHIII", resp)
        if status != 0:
            raise RuntimeError(f"server status {status}")
        
        body = b''
        if rvlen > 0:
            body = read_exact(s, rvlen)
        return body.decode('ascii', errors='replace')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1000, help="reporting batch size")
    parser.add_argument("--threads", type=int, default=4, help="number of concurrent threads")
    parser.add_argument("--timeout", type=int, default=30, help="socket timeout")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Starting benchmark: {args.count} vectors, {args.threads} threads, target {args.host}:{args.port}")

    # Calculate split
    vecs_per_thread = args.count // args.threads
    remainder = args.count % args.threads

    futures = []
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        curr_start = args.start
        for t_id in range(args.threads):
            # Distribute remainder to first few threads
            count = vecs_per_thread + (1 if t_id < remainder else 0)
            
            futures.append(
                executor.submit(insert_worker, t_id, args.host, args.port, curr_start, count, args.dim, args.batch, args.timeout)
            )
            curr_start += count
        
        # Wait for all threads
        total_inserted = 0
        total_errors = 0
        for f in futures:
            c, e, _ = f.result()
            total_inserted += c
            total_errors += e

    elapsed = time.time() - t0
    rate = total_inserted / elapsed if elapsed > 0 else 0

    print(f"\n=== RESULT ===")
    print(f"Total Inserted: {total_inserted}")
    print(f"Total Errors:   {total_errors}")
    print(f"Time Elapsed:   {elapsed:.2f}s")
    print(f"Throughput:     {rate:.2f} vectors/s")

    # Request memory usage from server
    try:
        print("\nRequesting Server Memory Usage...")
        time.sleep(1) # Give server a moment to settle
        s = request_memusage(args.host, args.port)
        print("Server memoryUsage (payload,index_overhead,total):")
        print(s)
    except Exception as e:
        print(f"[WARN] memoryUsage request failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()