#!/usr/bin/env python3
"""
Python wrapper to build and run the C++ SimHash microbenchmark and print results.
Usage:
  python3 benchmarks/run_simhash_bench.py --dim 512 --bits 512 --reps 20000 --batch 8
"""

import argparse
import subprocess
import sys
import shutil

def build(binaries):
    # compile simhash_bench
    cmd = [
        "g++", "-std=c++20", "-O3", "-march=native",
        "src/tools/simhash_bench.cc", "src/ai/simhash.cc",
        "-I.", "-pthread", "-o", binaries
    ]
    print("Building:", " ".join(cmd))
    r = subprocess.run(cmd)
    return r.returncode == 0

def run(binaries, dim, bits, reps, batch, seed):
    cmd = [binaries, "--dim", str(dim), "--bits", str(bits), "--reps", str(reps), "--batch", str(batch), "--seed", str(seed)]
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    return proc.returncode

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--bits", type=int, default=512)
    p.add_argument("--reps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    binpath = "build/simhash_bench"
    if not shutil.which("g++"):
        print("g++ not found in PATH")
        sys.exit(1)
    ok = build(binpath)
    if not ok:
        print("Build failed")
        sys.exit(2)
    rc = run(binpath, args.dim, args.bits, args.reps, args.batch, args.seed)
    sys.exit(rc)

if __name__ == "__main__":
    main()