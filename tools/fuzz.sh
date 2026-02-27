#!/bin/bash
set -e

# Build with Fuzzing enabled
rm -rf build_fuzz/*
mkdir -p build_fuzz
cd build_fuzz
CC=clang CXX=clang++ cmake .. -DPOMAI_BUILD_TESTS=ON -DENABLE_FUZZING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc) membrane_fuzzer storage_fuzzer

echo "Successfully built fuzzers."

# Run a fuzzer (example)
# ./tests/fuzz/membrane_fuzzer -max_total_time=600
