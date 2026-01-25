#!/usr/bin/env python3
"""
benchmark_client.py

Asynchronous TCP benchmark client for Pomai server (text SQL protocol).

Modes:
 - insert : sends "INSERT INTO <membr> VALUES (<label>, [v1, v2, ...])"
 - search : sends "SEARCH <membr> QUERY ([v1,v2,...]) TOP <k>"

Notes:
 - Keeps one persistent connection per worker.
 - Measures per-request latency and prints p50/p95/p99 and throughput.
 - Response reading: tries to read until "<END>" marker (server may append),
   otherwise uses a short timeout.
 - Verbose mode prints full server responses as they arrive.
"""

import argparse
import asyncio
import random
import time
import statistics
import logging
from typing import List, Optional, Tuple

DEFAULT_PORT = 7777

# Response read timeout (sec) when server does not provide explicit marker.
RESP_TIMEOUT = 2.0

# When the server doesn't emit <END>, allow this many extra seconds before giving up
MAX_RESP_FALLBACK = 2.0

PROGRESS_INTERVAL_S = 2.0


def gen_vector(dim: int, rng: random.Random) -> str:
    # Generate compact representation e.g. [0.123, -0.23, ...]
    vals = (f"{rng.uniform(-1.0, 1.0):.6f}" for _ in range(dim))
    return "[" + ",".join(vals) + "]"


async def read_response(
    reader: asyncio.StreamReader, require_end_marker: bool = False
) -> str:
    """
    Read server response. Prefer stopping when "<END>" line observed.
    Fallback: accumulate until RESP_TIMEOUT expires (short).
    Returns the full textual response.
    """
    buf_lines: List[str] = []
    end_marker = "<END>"
    start = time.monotonic()
    while True:
        # compute remaining timeout for this loop
        elapsed = time.monotonic() - start
        timeout = max(0.001, RESP_TIMEOUT - elapsed)
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            # If we've read anything, return it; otherwise allow a longer fallback
            if buf_lines:
                return "".join(buf_lines)
            if not require_end_marker:
                # give a small additional window
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=MAX_RESP_FALLBACK)
                except asyncio.TimeoutError:
                    return "".join(buf_lines)
                if not line:
                    return "".join(buf_lines)
            else:
                return "".join(buf_lines)

        if not line:
            return "".join(buf_lines)

        try:
            s = line.decode("utf-8", errors="replace")
        except Exception:
            s = line.decode("latin-1", errors="replace")
        buf_lines.append(s)

        # strip trailing newline/CR
        if s.rstrip("\r\n") == end_marker:
            return "".join(buf_lines)

        # Heuristic: if first line starts with "OK " or "ERR" and there's no marker,
        # return after first line so interactive flows don't stall forever.
        if len(buf_lines) == 1 and (s.startswith("OK") or s.startswith("ERR")) and not require_end_marker:
            # give a short chance for multiline replies, otherwise return
            try:
                peek = await asyncio.wait_for(reader.readline(), timeout=0.05)
                if not peek:
                    return "".join(buf_lines)
                # we got more; push into buffer and continue loop
                try:
                    ps = peek.decode("utf-8", errors="replace")
                except Exception:
                    ps = peek.decode("latin-1", errors="replace")
                buf_lines.append(ps)
                # if that line was the end marker, return
                if ps.rstrip("\r\n") == end_marker:
                    return "".join(buf_lines)
                # else continue reading until timeout / marker
            except asyncio.TimeoutError:
                return "".join(buf_lines)


async def worker_task(
    worker_id: int,
    host: str,
    port: int,
    mode: str,
    membr: str,
    dim: int,
    requests_per_worker: int,
    label_start: int,
    topk: int,
    rng_seed: int,
    latencies: List[float],
    success_counter: List[int],
    verbose: bool,
    require_end_marker: bool,
):
    rng = random.Random(rng_seed + worker_id)
    try:
        reader, writer = await asyncio.open_connection(host, port)
    except Exception as e:
        logging.error("[worker %d] connection failed: %s", worker_id, e)
        return

    peername = writer.get_extra_info("peername")
    logging.info("[worker %d] connected to %s", worker_id, peername)

    for i in range(requests_per_worker):
        label = label_start + i
        if mode == "insert":
            vec = gen_vector(dim, rng)
            cmd = f"INSERT INTO {membr} VALUES ({label}, {vec})\n"
        else:  # search
            qvec = gen_vector(dim, rng)
            cmd = f"SEARCH {membr} QUERY {qvec} TOP {topk}\n"

        start = time.perf_counter()
        try:
            writer.write(cmd.encode("utf-8"))
            await writer.drain()
        except Exception as e:
            logging.error("[worker %d] write error: %s", worker_id, e)
            break

        try:
            resp = await read_response(reader, require_end_marker=require_end_marker)
            dur = time.perf_counter() - start
            latencies.append(dur)
            success_counter[0] += 1
            if verbose:
                logging.debug("[worker %d] response: %s", worker_id, resp.replace("\n", "\\n"))
        except Exception as e:
            dur = time.perf_counter() - start
            latencies.append(dur)
            logging.warning("[worker %d] read error: %s", worker_id, e)
            # Try to continue; small backoff to avoid hammering broken connection
            await asyncio.sleep(0.001)

    try:
        writer.close()
        await writer.wait_closed()
    except Exception:
        pass
    logging.info("[worker %d] finished", worker_id)


def humanize_rate(count: int, elapsed: float) -> str:
    if elapsed <= 0:
        return f"{count} reqs"
    r = count / elapsed
    if r >= 1e6:
        return f"{r/1e6:.2f} Mreq/s"
    if r >= 1e3:
        return f"{r/1e3:.2f} Kreq/s"
    return f"{r:.2f} req/s"


def summarize(latencies: List[float], total_sent: int, elapsed: float):
    if not latencies:
        print("No recorded latencies.")
        return
    l50 = statistics.median(latencies)
    quantiles = statistics.quantiles(latencies, n=100)
    l95 = quantiles[94]
    l99 = quantiles[98]
    print("=== Results ===")
    print(f"Total requests configured: {total_sent}")
    print(f"Total successes recorded: {len(latencies)}")
    print(f"Elapsed time: {elapsed:.3f} s")
    print(f"Throughput: {humanize_rate(len(latencies), elapsed)}")
    print(f"Latency p50: {l50*1000:.3f} ms")
    print(f"Latency p95: {l95*1000:.3f} ms")
    print(f"Latency p99: {l99*1000:.3f} ms")
    print(f"Mean latency: {statistics.mean(latencies)*1000:.3f} ms")
    if len(latencies) > 1:
        print(f"Stddev latency: {statistics.stdev(latencies)*1000:.3f} ms")


async def run_benchmark(
    host: str,
    port: int,
    mode: str,
    membr: str,
    dim: int,
    conns: int,
    requests: int,
    label_base: int,
    topk: int,
    seed: int,
    verbose: bool,
    require_end_marker: bool,
):
    total = requests
    if conns <= 0:
        conns = 1
    per_worker = total // conns
    extra = total % conns

    latencies: List[float] = []
    success_counter = [0]  # simple mutable counter container

    tasks = []
    label_base = label_base
    t0 = time.perf_counter()
    for w in range(conns):
        nreq = per_worker + (1 if w < extra else 0)
        if nreq == 0:
            continue
        label_start = label_base + w * (per_worker) + min(w, extra)
        task = asyncio.create_task(
            worker_task(
                w,
                host,
                port,
                mode,
                membr,
                dim,
                nreq,
                label_start,
                topk,
                seed,
                latencies,
                success_counter,
                verbose,
                require_end_marker,
            )
        )
        tasks.append(task)

    # progress reporter
    async def progress_loop():
        last_count = 0
        last_time = t0
        while not all(t.done() for t in tasks):
            await asyncio.sleep(PROGRESS_INTERVAL_S)
            now = time.perf_counter()
            done = len(latencies)
            delta = done - last_count
            dt = now - last_time
            last_count = done
            last_time = now
            logging.info("Progress: %.1f%% (%d/%d) | recent throughput: %s", (done/total*100 if total>0 else 0), done, total, humanize_rate(delta, dt) if dt>0 else "0 req/s")
        return

    if tasks:
        reporter = asyncio.create_task(progress_loop())
        await asyncio.gather(*tasks)
        # ensure reporter finishes
        await reporter

    elapsed = time.perf_counter() - t0
    summarize(latencies, total, elapsed)


def parse_args():
    p = argparse.ArgumentParser(description="Async benchmark client for Pomai TCP server")
    p.add_argument("--host", type=str, default="127.0.0.1", help="server host")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="server port")
    p.add_argument("--mode", choices=("insert", "search"), default="insert", help="benchmark mode")
    p.add_argument("--membr", type=str, default="default", help="membrance name")
    p.add_argument("--dim", type=int, default=128, help="vector dimension")
    p.add_argument("--conns", type=int, default=4, help="number of concurrent connections/workers")
    p.add_argument("--requests", type=int, default=10000, help="total requests to send")
    p.add_argument("--label-base", type=int, default=1, help="starting label id for inserts")
    p.add_argument("--topk", type=int, default=10, help="k for SEARCH")
    p.add_argument("--seed", type=int, default=12345, help="random seed")
    p.add_argument("--verbose", action="store_true", help="print debug responses")
    p.add_argument("--require-end-marker", action="store_true", help="require <END> marker in responses (safer for multi-line)")
    return p.parse_args()


def main():
    args = parse_args()
    # basic validation
    if args.mode == "insert" and args.dim <= 0:
        print("dim must be > 0 for insert")
        return
    if args.requests <= 0:
        print("requests must be > 0")
        return

    # configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

    try:
        asyncio.run(
            run_benchmark(
                args.host,
                args.port,
                args.mode,
                args.membr,
                args.dim,
                args.conns,
                args.requests,
                args.label_base,
                args.topk,
                args.seed,
                args.verbose,
                args.require_end_marker,
            )
        )
    except KeyboardInterrupt:
        logging.info("Interrupted by user")


if __name__ == "__main__":
    main()