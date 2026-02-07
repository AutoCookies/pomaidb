#!/bin/bash
# PomaiDB Performance Gate
#
# Runs benchmarks and fails if performance degrades below thresholds.
# Usage: ./tools/perf_gate.sh [--dataset=small|medium|large]

set -e

# Default configuration
DATASET="small"
BUILD_DIR="build"
BENCH_EXEC="./build/comprehensive_bench"
RESULTS_FILE="/tmp/pomai_perf_results.json"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dataset=*)
            DATASET="${arg#*=}"
            shift
            ;;
        --build-dir=*)
            BUILD_DIR="${arg#*=}"
            BENCH_EXEC="./${arg#*=}/comprehensive_bench"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset=SIZE     Dataset size: small, medium, large (default: small)"
            echo "  --build-dir=PATH   Build directory (default: build)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Performance thresholds (vary by dataset size)
# These are baseline thresholds - adjust based on your environment
case $DATASET in
    small)
        MAX_P99_US=10000         # 10ms max P99 latency
        MIN_QPS=300              # Minimum 300 QPS
        MIN_RECALL=0.40          # 40% recall minimum (baseline)
        MAX_BUILD_SEC=10         # 10 second max build time
        ;;
    medium)
        MAX_P99_US=20000         # 20ms max P99 latency
        MIN_QPS=200              # Minimum 200 QPS
        MIN_RECALL=0.35          # 35% recall minimum
        MAX_BUILD_SEC=60         # 60 second max build time
        ;;
    large)
        MAX_P99_US=50000         # 50ms max P99 latency
        MIN_QPS=100              # Minimum 100 QPS
        MIN_RECALL=0.30          # 30% recall minimum
        MAX_BUILD_SEC=300        # 5 minute max build time
        ;;
    *)
        echo "Error: Invalid dataset '$DATASET'. Must be small, medium, or large."
        exit 1
        ;;
esac

echo "=========================================="
echo " PomaiDB Performance Gate"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Thresholds:"
echo "  - P99 Latency  < ${MAX_P99_US} µs"
echo "  - Throughput   > ${MIN_QPS} QPS"
echo "  - Recall@10    > ${MIN_RECALL}"
echo "  - Build Time   < ${MAX_BUILD_SEC} sec"
echo "=========================================="
echo ""

# Check if benchmark executable exists
if [ ! -f "$BENCH_EXEC" ]; then
    echo "Error: Benchmark executable not found at $BENCH_EXEC"
    echo "Please build it first: cmake --build $BUILD_DIR --target comprehensive_bench"
    exit 1
fi

# Run benchmark
echo "Running benchmark..."
$BENCH_EXEC --dataset $DATASET --output $RESULTS_FILE --threads 1

# Check if results file was created
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Results file not created"
    exit 1
fi

# Parse results using Python (fallback to awk if Python not available)
if command -v python3 &> /dev/null; then
    # Use Python for robust JSON parsing
    python3 - <<EOF
import json
import sys

with open("$RESULTS_FILE") as f:
    results = json.load(f)

# Extract metrics
p99_us = results["search_latency_us"]["p99"]
qps = results["throughput"]["qps"]
recall = results["accuracy"]["recall_at_k"]
build_sec = results["build"]["time_sec"]

# Check thresholds
failed = False

print("\nResults:")
print(f"  P99 Latency:  {p99_us:.2f} µs")
if p99_us > $MAX_P99_US:
    print(f"    ❌ FAILED: Exceeds threshold of $MAX_P99_US µs")
    failed = True
else:
    print(f"    ✅ PASSED")

print(f"  Throughput:   {qps:.2f} QPS")
if qps < $MIN_QPS:
    print(f"    ❌ FAILED: Below threshold of $MIN_QPS QPS")
    failed = True
else:
    print(f"    ✅ PASSED")

print(f"  Recall@10:    {recall:.4f}")
if recall < $MIN_RECALL:
    print(f"    ❌ FAILED: Below threshold of $MIN_RECALL")
    failed = True
else:
    print(f"    ✅ PASSED")

print(f"  Build Time:   {build_sec:.2f} sec")
if build_sec > $MAX_BUILD_SEC:
    print(f"    ❌ FAILED: Exceeds threshold of $MAX_BUILD_SEC sec")
    failed = True
else:
    print(f"    ✅ PASSED")

print("")
if failed:
    print("========================================")
    print(" ❌ PERFORMANCE GATE FAILED")
    print("========================================")
    sys.exit(1)
else:
    print("========================================")
    print(" ✅ PERFORMANCE GATE PASSED")
    print("========================================")
    sys.exit(0)
EOF
else
    # Fallback to basic grep/awk parsing (less robust)
    echo "Warning: Python3 not found, using basic parsing"
    
    # Extract with grep/awk (assumes specific JSON format)
    P99=$(grep '"p99"' "$RESULTS_FILE" | awk -F: '{print $2}' | tr -d ' ,')
    QPS=$(grep '"qps"' "$RESULTS_FILE" | awk -F: '{print $2}' | tr -d ' ,')
    RECALL=$(grep '"recall_at_k"' "$RESULTS_FILE" | awk -F: '{print $2}' | tr -d ' ,}')
    BUILD=$(grep '"time_sec"' "$RESULTS_FILE" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
    
    echo "Results:"
    echo "  P99: $P99 µs, QPS: $QPS, Recall: $RECALL, Build: $BUILD sec"
    echo ""
    echo "Note: Install python3 for detailed threshold checking"
    exit 0
fi
