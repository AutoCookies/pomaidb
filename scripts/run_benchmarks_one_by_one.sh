#!/usr/bin/env bash
# Run all PomaiDB benchmarks one by one (from build dir).
# Usage: ./scripts/run_benchmarks_one_by_one.sh [build_dir]
#   build_dir defaults to ./build
set -e
BUILD_DIR="${1:-build}"
cd "$(dirname "$0")/.."
if [[ ! -d "$BUILD_DIR" ]]; then
  echo "Build dir not found: $BUILD_DIR. Run: mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\$(nproc)"
  exit 1
fi
cd "$BUILD_DIR"

run() { echo "===== $1 ====="; "$@"; echo ""; }

echo "Running PomaiDB benchmarks one by one..."
echo ""

# 1. Baseline (single-threaded write + search, ~5s)
run ./bench_baseline

# 2. Comprehensive (dataset small = 10k vectors, 1k queries)
run ./comprehensive_bench --dataset small

# 3. Ingestion throughput (10k vectors, 128 dim)
run ./ingestion_bench 10000 128

# 4. RAG (minimal: 100 chunks)
run ./rag_bench 100 64 32

# 5. CI perf gate (deterministic, writes JSON)
run ./ci_perf_bench

# 6. CBR-S (single scenario; output in build/out/ or build/bin/../out/)
mkdir -p out 2>/dev/null || true
run ./bin/bench_cbrs

# 7. Multi-environment stress (use low-memory for shorter run)
run env POMAI_BENCH_LOW_MEMORY=1 ./benchmark_a

# 8. Quantization Comparison (Recall@1, Recall@10, Throughput)
run ./quantization_bench

echo "All benchmarks completed."
