#!/usr/bin/env python3
"""
benchmark_client.py - DIAGNOSTIC VERSION
High-Resolution Logging & Error Tracing for PomaiDB Binary Protocol.

Usage:
  python3 benchmarks/benchmark_client.py --mode insert --batch-size 10 --total 100 --conns 1 --verbose
"""

import argparse
import asyncio
import struct
import random
import time
import statistics
import logging
import numpy as np
import binascii
from typing import List

# ---- Protocol Constants ----
OP_INSERT       = 0x01
OP_SEARCH       = 0x02
OP_INSERT_BATCH = 0x04
RESP_OK         = 0x01
RESP_FAIL       = 0x00

class BinaryStats:
    def __init__(self):
        self.latencies = []
        self.connect_times = []
        self.send_times = []
        self.recv_times = []
        self.success = 0
        self.fail = 0
        self.bytes_sent = 0

    def add(self, t_conn, t_send, t_recv, total, ok, size):
        self.connect_times.append(t_conn)
        self.send_times.append(t_send)
        self.recv_times.append(t_recv)
        self.latencies.append(total)
        if ok: self.success += 1
        else: self.fail += 1
        self.bytes_sent += size

    def merge(self, other):
        self.latencies.extend(other.latencies)
        self.connect_times.extend(other.connect_times)
        self.send_times.extend(other.send_times)
        self.recv_times.extend(other.recv_times)
        self.success += other.success
        self.fail += other.fail
        self.bytes_sent += other.bytes_sent

def gen_batch_vectors(count: int, dim: int) -> bytes:
    arr = np.random.rand(count, dim).astype(np.float32)
    return arr.tobytes()

async def worker_insert(
    worker_id: int, host: str, port: int,
    total: int, batch_size: int, dim: int,
    label_start: int, stats: BinaryStats, verbose: bool
):
    # 1. Measure Connect Time
    t0 = time.perf_counter()
    try:
        reader, writer = await asyncio.open_connection(host, port)
        # Tweak socket for low latency
        sock = writer.get_extra_info('socket')
        if sock:
            import socket
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        logging.error(f"[Worker {worker_id}] Connection Failed: {e}")
        return
    t_conn = time.perf_counter() - t0

    if verbose:
        logging.debug(f"[Worker {worker_id}] Connected in {t_conn*1000:.3f}ms")

    head_fmt = struct.Struct("<BII") 
    lbl_fmt = struct.Struct("<Q")

    remaining = total
    curr_label = label_start
    
    while remaining > 0:
        count = min(batch_size, remaining)
        
        # Prepare Data
        packet = bytearray()
        packet.extend(head_fmt.pack(OP_INSERT_BATCH, count, dim))
        vec_bytes = gen_batch_vectors(count, dim)
        vec_size = dim * 4
        
        for i in range(count):
            packet.extend(lbl_fmt.pack(curr_label + i))
            start = i * vec_size
            packet.extend(vec_bytes[start : start + vec_size])
            
        if verbose and len(packet) < 200:
            logging.debug(f"[Worker {worker_id}] Sending {len(packet)} bytes: {binascii.hexlify(packet)}")
        elif verbose:
            logging.debug(f"[Worker {worker_id}] Sending {len(packet)} bytes (Header: {binascii.hexlify(packet[:9])}...)")

        # 2. Measure Send Time
        t_start_send = time.perf_counter()
        try:
            writer.write(packet)
            await writer.drain()
        except Exception as e:
            logging.error(f"[Worker {worker_id}] Send Failed: {e}")
            break
        t_sent = time.perf_counter()

        # 3. Measure Receive Time (Wait for ACK)
        try:
            resp = await reader.readexactly(1)
        except asyncio.IncompleteReadError:
            logging.error(f"[Worker {worker_id}] Server closed connection unexpectedly!")
            break
        except Exception as e:
            logging.error(f"[Worker {worker_id}] Recv Error: {e}")
            break
        
        t_end = time.perf_counter()
        
        t_send_dur = t_sent - t_start_send
        t_recv_dur = t_end - t_sent
        t_total = t_end - t_start_send
        
        ok = (resp[0] == RESP_OK)
        stats.add(t_conn, t_send_dur, t_recv_dur, t_total, ok, len(packet))
        
        if not ok:
            logging.warning(f"[Worker {worker_id}] Server sent FAIL (0x00). Membrance missing?")
        elif verbose:
            logging.debug(f"[Worker {worker_id}] OK. Send={t_send_dur*1000:.3f}ms, ServerWait={t_recv_dur*1000:.3f}ms")
            
        remaining -= count
        curr_label += count

    writer.close()
    await writer.wait_closed()

async def worker_search(
    worker_id: int, host: str, port: int,
    requests: int, dim: int, topk: int,
    stats: BinaryStats, verbose: bool
):
    t0 = time.perf_counter()
    try:
        reader, writer = await asyncio.open_connection(host, port)
    except Exception as e:
        logging.error(f"[Worker {worker_id}] Connect Failed: {e}")
        return
    t_conn = time.perf_counter() - t0

    head_fmt = struct.Struct("<BII")
    
    for _ in range(requests):
        vec_bytes = gen_batch_vectors(1, dim)
        packet = head_fmt.pack(OP_SEARCH, topk, dim) + vec_bytes
        
        t_start_send = time.perf_counter()
        try:
            writer.write(packet)
            await writer.drain()
            
            # Read Count
            raw_cnt = await reader.readexactly(4)
            (count,) = struct.unpack("<I", raw_cnt)
            
            # Read Hits
            if count > 0:
                await reader.readexactly(count * 12) # 8+4
            
            t_end = time.perf_counter()
            
            t_send_dur = 0 # approximated
            t_recv_dur = t_end - t_start_send
            
            stats.add(t_conn, t_send_dur, t_recv_dur, t_recv_dur, True, len(packet))
            
        except Exception as e:
            logging.error(f"[Worker {worker_id}] Search IO Error: {e}")
            break

    writer.close()
    await writer.wait_closed()

async def run(args):
    print(f"=== POMAI DIAGNOSTIC BENCHMARK ===")
    print(f"Mode: {args.mode.upper()} | Workers: {args.conns} | Batch: {args.batch_size}")
    
    stats_list = [BinaryStats() for _ in range(args.conns)]
    tasks = []
    
    t_start = time.perf_counter()
    
    for i in range(args.conns):
        if args.mode == "insert":
            per_worker = args.total // args.conns
            label_base = 1 + i * per_worker
            tasks.append(worker_insert(
                i, args.host, args.port, per_worker, args.batch_size, 
                args.dim, label_base, stats_list[i], args.verbose
            ))
        else:
            per_worker = args.requests // args.conns
            tasks.append(worker_search(
                i, args.host, args.port, per_worker, 
                args.dim, args.topk, stats_list[i], args.verbose
            ))
            
    await asyncio.gather(*tasks)
    dur = time.perf_counter() - t_start
    
    # Aggregation
    agg = BinaryStats()
    for s in stats_list: agg.merge(s)
    
    print("\n" + "="*30)
    print(f"  Execution Time:   {dur:.4f} s")
    print(f"  Total Requests:   {len(agg.latencies)}")
    print(f"  Success / Fail:   \033[92m{agg.success}\033[0m / \033[91m{agg.fail}\033[0m")
    
    if len(agg.latencies) > 0:
        avg_conn = statistics.mean(agg.connect_times) * 1000
        avg_send = statistics.mean(agg.send_times) * 1000
        avg_wait = statistics.mean(agg.recv_times) * 1000
        
        print("\n  [TIMING BREAKDOWN (avg)]")
        print(f"  1. Connect:       {avg_conn:.3f} ms")
        print(f"  2. Data Send:     {avg_send:.3f} ms  (Client CPU/Net)")
        print(f"  3. Server Wait:   {avg_wait:.3f} ms  (Server Processing)")
        
        print("\n  [LATENCY PERCENTILES]")
        lats = sorted(agg.latencies)
        print(f"  P50:              {lats[int(len(lats)*0.5)]*1000:.3f} ms")
        print(f"  P99:              {lats[int(len(lats)*0.99)]*1000:.3f} ms")
        
        throughput = len(agg.latencies) / dur
        print(f"\n  [THROUGHPUT]")
        print(f"  Ops/sec:          {throughput:.2f}")
        if args.mode == "insert":
            vec_rate = (agg.success * args.batch_size) / dur
            print(f"  Vectors/sec:      {vec_rate:.2f}")
            
    else:
        print("\n  [ERROR] No requests completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--mode", default="insert", choices=["insert", "search"])
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--conns", type=int, default=1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Log detailed packet info")
    
    try:
        asyncio.run(run(parser.parse_args()))
    except KeyboardInterrupt:
        pass