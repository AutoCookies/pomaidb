#!/usr/bin/env python3
"""
vsearch.py - query Pomai server for ANN (VSEARCH).

Usage:
  # random query
  ./vsearch.py --host 127.0.0.1 --port 12345 --topk 5 --dim 128 --random

  # explicit values
  ./vsearch.py --host 127.0.0.1 --port 12345 --topk 10 --values "0.1,0.2,0.3"

Response parsing:
  Server responds with body = repeated entries: [4B keylen][key bytes][4B score_bits(network order)]
"""
import argparse
import socket
import struct
import random
import sys

PWP_MAGIC = ord('P')
OP_VSEARCH = 11

def read_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket closed")
        buf += chunk
    return buf

def build_vector_bytes_from_values(values):
    return b''.join(struct.pack('f', float(x)) for x in values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--topk", type=int, required=True)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--random", action="store_true")
    grp.add_argument("--values", help="comma-separated float values")
    grp.add_argument("--fromfile", help="file containing whitespace-separated floats")
    parser.add_argument("--dim", type=int, default=128, help="vector dimension (used with --random)")
    args = parser.parse_args()

    if args.random:
        values = [random.random() for _ in range(args.dim)]
    elif args.values:
        values = [v for v in args.values.split(',') if v != '']
    else:
        with open(args.fromfile, 'r') as f:
            data = f.read().strip().split()
            if not data:
                print("Empty vector file", file=sys.stderr); sys.exit(2)
            values = data

    vec_bytes = build_vector_bytes_from_values(values)
    # build body = [4B topk(net)] + vec_bytes
    body = struct.pack("!I", args.topk) + vec_bytes
    klen = 0
    vlen = len(body)
    hdr = struct.pack("!BBHIII", PWP_MAGIC, OP_VSEARCH, 0, klen, vlen, 0)

    with socket.create_connection((args.host, args.port), timeout=5) as s:
        s.sendall(hdr + body)
        resp_hdr = read_exact(s, 16)
        magic, op, status, rklen, rvlen, reserved = struct.unpack("!BBHIII", resp_hdr)
        if magic != PWP_MAGIC:
            print("Invalid response magic", file=sys.stderr); sys.exit(3)
        status = status
        if status != 0:
            print("Server returned status", status, file=sys.stderr); sys.exit(4)
        # read body of length rvlen
        body = b''
        if rvlen:
            body = read_exact(s, rvlen)
        # parse repeated entries: [4B keylen][key bytes][4B score(net float)]
        pos = 0
        results = []
        while pos + 4 <= len(body):
            (klen_net,) = struct.unpack_from("!I", body, pos)
            pos += 4
            klen = klen_net
            if pos + klen + 4 > len(body):
                print("Malformed response body", file=sys.stderr); break
            key_bytes = body[pos:pos + klen]
            pos += klen
            score_bytes = body[pos:pos + 4]
            pos += 4
            # float in network (big-endian) order
            (score,) = struct.unpack("!f", score_bytes)
            key = key_bytes.decode('utf-8', errors='replace')
            results.append((key, score))
        # print results
        for i, (k, sscore) in enumerate(results):
            print(f"{i+1:2d}: key='{k}' score={sscore}")
        if not results:
            print("No matches")
        sys.exit(0)

if __name__ == "__main__":
    main()