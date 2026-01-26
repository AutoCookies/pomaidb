#!/usr/bin/env python3
import argparse
import socket
import struct
import time
import random
import statistics
from array import array

# ---- Binary protocol ----
# Frame: u32 len (LE) + payload[len]
# payload: u8 op + fields
OP_PING = 1
OP_CREATE_COLLECTION = 2
OP_UPSERT_BATCH = 3
OP_SEARCH = 4

METRIC_L2 = 0
METRIC_COSINE = 1


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def frame(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("server closed connection")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> bytes:
    ln = struct.unpack("<I", recv_exact(sock, 4))[0]
    if ln == 0:
        return b""
    return recv_exact(sock, ln)


def put_string_u16(s: str) -> bytes:
    b = s.encode("utf-8")
    if len(b) > 65535:
        raise ValueError("string too long")
    return struct.pack("<H", len(b)) + b


def req_ping(sock) -> None:
    sock.sendall(frame(struct.pack("<B", OP_PING)))
    resp = recv_frame(sock)
    if not resp or resp[0] != 1:
        raise RuntimeError(f"PING failed resp={resp!r}")


def req_create_collection(sock, name: str, dim: int, metric: str, shards: int, cap: int) -> int:
    m = METRIC_COSINE if metric == "cosine" else METRIC_L2
    payload = bytearray()
    payload += struct.pack("<B", OP_CREATE_COLLECTION)
    payload += put_string_u16(name)
    payload += struct.pack("<H", dim)
    payload += struct.pack("<B", m)
    # optional overrides (server may ignore or use)
    payload += struct.pack("<I", shards)
    payload += struct.pack("<I", cap)

    sock.sendall(frame(payload))
    resp = recv_frame(sock)
    if not resp:
        raise RuntimeError("CREATE_COLLECTION empty resp")
    ok = resp[0]
    if ok != 1:
        # error is u16-len string in our C++ error response
        # but some servers may just return ok=0
        raise RuntimeError(f"CREATE_COLLECTION failed resp={resp!r}")

    # resp: [u8 ok][u32 col_id]
    if len(resp) < 1 + 4:
        raise RuntimeError(f"CREATE_COLLECTION bad resp len={len(resp)}")
    col_id = struct.unpack_from("<I", resp, 1)[0]
    return col_id


def req_upsert_batch(sock, col_id: int, dim: int, ids, vec_f32: array) -> int:
    n = len(ids)
    if len(vec_f32) != n * dim:
        raise ValueError("vec_f32 length mismatch")

    payload = bytearray()
    payload += struct.pack("<B", OP_UPSERT_BATCH)
    payload += struct.pack("<I", col_id)
    payload += struct.pack("<I", n)
    payload += struct.pack("<H", dim)
    payload += struct.pack("<" + "Q" * n, *ids)
    payload += vec_f32.tobytes()

    sock.sendall(frame(payload))
    resp = recv_frame(sock)
    if not resp:
        raise RuntimeError("UPSERT_BATCH empty resp")
    ok = resp[0]
    if ok != 1:
        raise RuntimeError(f"UPSERT_BATCH failed resp={resp!r}")

    # resp: [u8 ok][u64 lsn]
    if len(resp) < 1 + 8:
        raise RuntimeError(f"UPSERT_BATCH bad resp len={len(resp)}")
    lsn = struct.unpack_from("<Q", resp, 1)[0]
    return lsn


def req_search(sock, col_id: int, dim: int, topk: int, q_f32: array):
    if len(q_f32) != dim:
        raise ValueError("query dim mismatch")

    payload = bytearray()
    payload += struct.pack("<B", OP_SEARCH)
    payload += struct.pack("<I", col_id)
    payload += struct.pack("<I", topk)
    payload += struct.pack("<H", dim)
    payload += q_f32.tobytes()

    sock.sendall(frame(payload))
    resp = recv_frame(sock)
    if not resp:
        raise RuntimeError("SEARCH empty resp")
    ok = resp[0]
    if ok != 1:
        raise RuntimeError(f"SEARCH failed resp={resp!r}")

    # resp: [u8 ok][u32 count][u64 ids[count]][f32 scores[count]]
    if len(resp) < 1 + 4:
        raise RuntimeError("SEARCH bad resp header")
    count = struct.unpack_from("<I", resp, 1)[0]
    off = 1 + 4
    need = off + 8 * count + 4 * count
    if len(resp) < need:
        raise RuntimeError(f"SEARCH truncated resp len={len(resp)} need={need}")

    ids = list(struct.unpack_from("<" + "Q" * count, resp, off))
    off += 8 * count
    scores = list(struct.unpack_from("<" + "f" * count, resp, off))
    return list(zip(ids, scores))


def main():
    ap = argparse.ArgumentParser(description="Pomai binary protocol benchmark (localhost, single-user)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7744)

    ap.add_argument("--collection", default="bench")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--cap", type=int, default=4096)

    ap.add_argument("--ping", type=int, default=1000)
    ap.add_argument("--vectors", type=int, default=20000, help="total vectors to upsert")
    ap.add_argument("--batch", type=int, default=512, help="vectors per upsert_batch")
    ap.add_argument("--search", type=int, default=5000, help="number of search ops")
    ap.add_argument("--topk", type=int, default=10)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--id-base", type=int, default=0)

    args = ap.parse_args()

    # Local daemon style: one TCP connection, keepalive
    sock = socket.create_connection((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    rng = random.Random(args.seed)

    # --- PING ---
    ping_lat = []
    t0 = time.perf_counter()
    for _ in range(args.ping):
        a = time.perf_counter()
        req_ping(sock)
        b = time.perf_counter()
        ping_lat.append((b - a) * 1000.0)
    wall = time.perf_counter() - t0
    thr = args.ping / wall
    xs = sorted(ping_lat)
    print("PING:")
    print(f"  ops: {args.ping}")
    print(f"  wall: {wall:.3f}s")
    print(f"  throughput: {thr:,.1f} ops/s")
    print(
        f"  latency ms: avg={statistics.mean(xs):.3f}  "
        f"p50={percentile(xs,50):.3f}  p95={percentile(xs,95):.3f}  "
        f"p99={percentile(xs,99):.3f}  max={xs[-1]:.3f}"
    )
    print()

    # --- CREATE_COLLECTION (once) ---
    col_id = req_create_collection(sock, args.collection, args.dim, args.metric, args.shards, args.cap)

    # --- UPSERT_BATCH ---
    total = args.vectors
    bs = args.batch
    batches = (total + bs - 1) // bs

    up_lat = []
    start_id = args.id_base

    t0 = time.perf_counter()
    for b in range(batches):
        n = min(bs, total - b * bs)
        ids = [start_id + i for i in range(n)]
        start_id += n

        vec = array("f")
        # contiguous float32 block
        vec.extend((rng.uniform(-1.0, 1.0) for _ in range(n * args.dim)))

        a = time.perf_counter()
        req_upsert_batch(sock, col_id, args.dim, ids, vec)
        c = time.perf_counter()
        up_lat.append((c - a) * 1000.0)

    wall = time.perf_counter() - t0
    thr = total / wall
    xs = sorted(up_lat)
    print(f"UPSERT_BATCH:")
    print(f"  vectors: {total} (batch={bs})")
    print(f"  wall: {wall:.3f}s")
    print(f"  throughput: {thr:,.1f} vec/s")
    print(
        f"  batch latency ms: avg={statistics.mean(xs):.3f}  "
        f"p50={percentile(xs,50):.3f}  p95={percentile(xs,95):.3f}  "
        f"p99={percentile(xs,99):.3f}  max={xs[-1]:.3f}"
    )
    print()

    # --- SEARCH ---
    s_lat = []
    t0 = time.perf_counter()
    for _ in range(args.search):
        q = array("f", (rng.uniform(-1.0, 1.0) for _ in range(args.dim)))
        a = time.perf_counter()
        req_search(sock, col_id, args.dim, args.topk, q)
        c = time.perf_counter()
        s_lat.append((c - a) * 1000.0)

    wall = time.perf_counter() - t0
    thr = args.search / wall
    xs = sorted(s_lat)
    print("SEARCH:")
    print(f"  ops: {args.search} (topk={args.topk})")
    print(f"  wall: {wall:.3f}s")
    print(f"  throughput: {thr:,.1f} ops/s")
    print(
        f"  latency ms: avg={statistics.mean(xs):.3f}  "
        f"p50={percentile(xs,50):.3f}  p95={percentile(xs,95):.3f}  "
        f"p99={percentile(xs,99):.3f}  max={xs[-1]:.3f}"
    )

    sock.close()


if __name__ == "__main__":
    main()
