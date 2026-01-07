#!/usr/bin/env python3
"""
bench_insert.py -- bulk insert random float vectors via VSET PWP.

Usage:
  ./benchmarks/bench_insert.py --host 127.0.0.1 --port 6379 --count 10000 --dim 64 --start 0 --batch 100

This will send count vectors named key_{i} with random floats in [0,1].
It prints progress and a summary (total time, insert/sec).
"""
import argparse
import socket
import struct
import time
import random
import sys

PWP_MAGIC = ord('P')
OP_VSET = 10

def send_one(host, port, key_bytes, vec_bytes, timeout=5):
    hdr = struct.pack("!BBHIII", PWP_MAGIC, OP_VSET, 0, len(key_bytes), len(vec_bytes), 0)
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(hdr + key_bytes + vec_bytes)
        # read response header (16 bytes)
        resp = s.recv(16)
        if len(resp) < 16:
            raise RuntimeError("short response")
        magic, op, status, rklen, rvlen, reserved = struct.unpack("!BBHIII", resp)
        if magic != PWP_MAGIC:
            raise RuntimeError("bad mag")
        if status != 0:
            raise RuntimeError(f"server status {status}")

def float_list_to_bytes(values):
    return b''.join(struct.pack('f', float(x)) for x in values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch", type=int, default=100, help="reporting batch size")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    total = args.count
    start = args.start
    dim = args.dim
    host = args.host
    port = args.port

    t0 = time.time()
    errors = 0
    for i in range(start, start + total):
        key = f"vec_{i}".encode('utf-8')
        vec = [random.random() for _ in range(dim)]
        vec_bytes = float_list_to_bytes(vec)
        try:
            send_one(host, port, key, vec_bytes)
        except Exception as e:
            errors += 1
            print(f"[ERROR] insert {i} -> {e}", file=sys.stderr)
        if (i - start + 1) % args.batch == 0:
            elapsed = time.time() - t0
            rate = (i - start + 1) / (elapsed if elapsed > 0 else 1e-9)
            print(f"Inserted {i - start + 1}/{total} (errs={errors}) rate={rate:.1f}/s", flush=True)

    elapsed = time.time() - t0
    print(f"Done inserts: total={total} errors={errors} time={elapsed:.2f}s rate={total/elapsed:.2f}/s")

if __name__ == "__main__":
    main()