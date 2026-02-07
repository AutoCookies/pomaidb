#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEED="${1:-1337}"
DB_PATH="${2:-/tmp/pomai_bench/cbrs_matrix}"

cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build-bench -j"$(nproc)"

mkdir -p out
build-bench/bin/bench_cbrs --matrix full --seed "$SEED" --path "$DB_PATH"
