#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv-benchmark
source .venv-benchmark/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib faiss-cpu hnswlib

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

python benchmarks/cross_engine/run_benchmark.py \
  --output-dir benchmarks/cross_engine/output \
  --libpomai build/libpomai_c.so

echo "Benchmark complete. See benchmarks/cross_engine/output/results.md and results.json"
