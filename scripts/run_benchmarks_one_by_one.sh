#!/usr/bin/env bash
# Run all PomaiDB benchmark binaries with bounded workloads (no ctest default timeout surprises).
# Usage: ./scripts/run_benchmarks_one_by_one.sh [build_dir]
#   build_dir defaults to ./build
set -euo pipefail
BUILD_DIR="${1:-build}"
cd "$(dirname "$0")/.."
if [[ ! -d "$BUILD_DIR" ]]; then
  echo "Build dir not found: $BUILD_DIR. Run: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\"\$(nproc)\""
  exit 1
fi
cd "$BUILD_DIR"

run() { echo "===== $* ====="; "$@"; echo ""; }

echo "Running PomaiDB benchmarks (bounded args, POMAI_BENCH_LOW_MEMORY for benchmark_a)..."
echo ""

# Core / ingest / RAG / CI gate
run ./comprehensive_bench --dataset small
run ./ingestion_bench 10000 128
run ./rag_bench 100 64 32
run ./ci_perf_bench

# Graph + quantization
run ./graph_bench 2000 1000 4000
run ./quantization_bench

# CBR-S (single scenario; full matrix: ./bin/bench_cbrs --matrix full)
mkdir -p out 2>/dev/null || true
run ./bin/bench_cbrs --n 8000 --queries 400 --topk 10

# Multi-environment stress (smaller when POMAI_BENCH_LOW_MEMORY set)
run env POMAI_BENCH_LOW_MEMORY=1 ./benchmark_a

# Remaining harnesses
run ./encryption_perf_bench
run ./hybrid_orchestrator_bench
run ./low_ram_profile_bench
run ./new_membrane_bench
run ./simd_new_membranes_bench
run ./edge_ai_core_bench
run ./mesh_lod_bench
run ./vulkan_transfer_bench

echo "All benchmarks completed."
echo "Tip: run the same set under CTest with: ctest -L bench --output-on-failure"
