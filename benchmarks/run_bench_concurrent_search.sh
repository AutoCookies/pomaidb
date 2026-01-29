#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build
cmake --build build --target bench_concurrent_search
./build/bench_concurrent_search
