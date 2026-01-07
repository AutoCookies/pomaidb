#!/usr/bin/env python3
"""
vset.py - send a vector (VSET) to Pomai server.

Usage examples:
  # random vector
  ./vset.py --host 127.0.0.1 --port 12345 --key myvec --dim 128 --random

  # explicit values (comma separated)
  ./vset.py --host 127.0.0.1 --port 12345 --key myvec --values "0.1,0.2,0.3"

  # reads vector values from a whitespace-separated file
  ./vset.py --host 127.0.0.1 --port 12345 --key myvec --fromfile vec.txt
"""
import argparse
import socket
import struct
import random
import sys

PWP_MAGIC = ord('P')
OP_VSET = 10

def read_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket closed")
        buf += chunk
    return buf

def build_vector_bytes_from_values(values):
    # pack floats in native float32 representation
    b = b''.join(struct.pack('f', float(x)) for x in values)
    return b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--key", required=True)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--random", action="store_true", help="use random vector")
    grp.add_argument("--values", help="comma-separated float values")
    grp.add_argument("--fromfile", help="file containing whitespace-separated floats")
    parser.add_argument("--dim", type=int, default=128, help="vector dimension (used with --random)")
    args = parser.parse_args()

    if args.random:
        values = [random.random() for _ in range(args.dim)]
    elif args.values:
        values = [v for v in args.values.split(',') if v != '']
    else:
        # from file
        with open(args.fromfile, 'r') as f:
            data = f.read().strip().split()
            if not data:
                print("Empty vector file", file=sys.stderr); sys.exit(2)
            values = data

    vec_bytes = build_vector_bytes_from_values(values)
    key_bytes = args.key.encode('utf-8')

    # build PWP header: !BBHIII
    klen = len(key_bytes)
    vlen = len(vec_bytes)
    hdr = struct.pack("!BBHIII", PWP_MAGIC, OP_VSET, 0, klen, vlen, 0)

    body = key_bytes + vec_bytes

    with socket.create_connection((args.host, args.port), timeout=5) as s:
        s.sendall(hdr + body)
        # read response header (16 bytes)
        resp_hdr = read_exact(s, 16)
        magic, op, status, rklen, rvlen, reserved = struct.unpack("!BBHIII", resp_hdr)
        if magic != PWP_MAGIC:
            print("Invalid response magic", file=sys.stderr); sys.exit(3)
        status = status  # network->host already by unpack '!H'
        if status == 0:
            print("VSET OK")
            sys.exit(0)
        elif status == 2:
            print("VSET failed: server FULL", file=sys.stderr)
            sys.exit(4)
        else:
            print("VSET failed: status", status, file=sys.stderr)
            sys.exit(5)

if __name__ == "__main__":
    main()