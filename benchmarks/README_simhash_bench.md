```text
SimHash micro-benchmark
=======================

This benchmark measures the throughput of SimHash::compute() and
SimHash::compute_words() on your machine.

Files:
 - src/tools/simhash_bench.cc   (C++ microbenchmark)
 - benchmarks/run_simhash_bench.py  (small Python wrapper to run the binary and parse results)

Build:
  mkdir -p build
  g++ -std=c++20 -O3 -march=native \
      src/tools/simhash_bench.cc src/ai/simhash.cc \
      -I. -pthread -o build/simhash_bench

Run:
  # Example: dim=512 bits=512 reps=10000
  ./build/simhash_bench --dim 512 --bits 512 --reps 10000 --batch 8

The Python wrapper simply runs the binary and prints the results in a tidy table.