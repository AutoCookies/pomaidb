# PomaiDB CI/CD Quick Reference

Complete guide for running all quality checks locally before pushing code.

## Quick Start - Run Everything

```bash
# Full CI pipeline (takes ~5-10 minutes)
./tools/ci_local.sh
```

Or run individual stages:

```bash
# 1. Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 2. Unit & Integration Tests
ctest --test-dir build --output-on-failure -j$(nproc)

# 3. Performance Gates
./tools/perf_gate.sh --dataset=small

# 4. Fuzzing (optional, 10K random operations)
./tools/fuzz.sh
```

---

## Test Categories

### 1. Core Functionality Tests

```bash
# Basic DB operations
./build/db_basic_test

# Consistency guarantees (RYW semantics)
./build/db_consistency_test

# Multi-shard error handling
./build/db_partial_search_test

# Membrane persistence
./build/membrane_persistence_test
```

**Expected**: All tests pass ✓

### 2. Snapshot Iterator Tests

```bash
# Full-scan iteration with tombstone filtering
./build/iterator_test
```

**Expected**: All 5 test cases pass (ID iteration, tombstones, deduplication, snapshot isolation, ordering)

### 3. Crash Recovery & Safety

```bash
# WAL corruption, incomplete flush, concurrent consistency
./build/recovery_test

# Multi-round crash simulation (requires ~30 seconds)
POMAI_ENABLE_CRASH_TESTS=1 ./build/pomai_crash_test
```

**Expected**: No data loss, graceful corruption handling

### 4. Performance Benchmarks

```bash
# Small dataset (10K vectors, ~30 seconds)
./build/comprehensive_bench --dataset small --threads 1

# Medium dataset (100K vectors, ~2-3 minutes)
./build/comprehensive_bench --dataset medium --threads 4 --output results.json

# Check performance gates
./tools/perf_gate.sh --dataset=small
```

**Expected**:
- Small: P99 < 10ms, QPS > 300, Recall > 40%
- JSON output saved for tracking

### 5. Fuzzing (Random Operations)

```bash
# 10K random Put/Delete/Get/Search operations
./tools/fuzz.sh
```

**Expected**: No crashes, all operations complete

---

## Complete Test Suite

Run all tests with CTest:

```bash
# All tests (parallel execution)
ctest --test-dir build --output-on-failure -j8

# Specific labels
ctest --test-dir build -L integ    # Integration tests only
ctest --test-dir build -L crash    # Crash recovery tests
ctest --test-dir build -L bench    # Benchmarks
```

---

## CI Pipeline Stages

### Stage 1: Build & Compile

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-Wall -Werror"
cmake --build build -j$(nproc)
```

**Gate**: Build must succeed with zero warnings/errors

### Stage 2: Unit & Integration Tests

```bash
ctest --test-dir build --output-on-failure -j$(nproc) -L integ
```

**Gate**: All tests must pass

### Stage 3: Performance Validation

```bash
./tools/perf_gate.sh --dataset=small
```

**Gate**: Must meet baseline thresholds (P99, QPS, Recall)

### Stage 4: Safety & Robustness

```bash
./build/recovery_test
```

**Gate**: Crash recovery tests pass

### Stage 5: Export & Tools

```bash
# Verify pomai_inspect works
./build/pomai_inspect scan /tmp/test_db --format=json > /dev/null
```

**Gate**: Tools execute without errors

---

## Local Pre-Commit Checklist

Before pushing code, verify:

- [ ] `cmake --build build` succeeds
- [ ] `ctest --test-dir build -L integ` passes
- [ ] `./tools/perf_gate.sh --dataset=small` passes
- [ ] Code formatted (if applicable)

**Optional but recommended**:
- [ ] `./build/recovery_test` passes
- [ ] `./tools/fuzz.sh` completes successfully

---

## Automated CI Configuration

### GitHub Actions Example

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build
        run: |
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build -j$(nproc)
      
      - name: Test
        run: ctest --test-dir build --output-on-failure -j$(nproc)
      
      - name: Performance Gate
        run: ./tools/perf_gate.sh --dataset=small
```

### GitLab CI Example

```yaml
stages:
  - build
  - test
  - benchmark

build:
  stage: build
  script:
    - cmake -B build -DCMAKE_BUILD_TYPE=Release
    - cmake --build build -j$(nproc)

test:
  stage: test
  script:
    - ctest --test-dir build --output-on-failure -j$(nproc)

benchmark:
  stage: benchmark
  script:
    - ./tools/perf_gate.sh --dataset=small
  allow_failure: true  # Performance gate optional
```

---

## Troubleshooting

### Tests Fail

```bash
# Run specific test with verbose output
./build/db_basic_test --verbose

# Check test logs
cat build/Testing/Temporary/LastTest.log
```

### Performance Gate Fails

```bash
# Get detailed benchmark results
./build/comprehensive_bench --dataset small --output results.json
cat results.json

# Adjust thresholds in tools/perf_gate.sh if needed
```

### Build Errors

```bash
# Clean rebuild
rm -rf build
cmake -B build
cmake --build build
```

---

## Performance Tracking

Save benchmark results over time:

```bash
# Run and save with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
./build/comprehensive_bench --dataset small \
  --output "results_${TIMESTAMP}.json"

# Compare with baseline
python3 scripts/compare_benchmarks.py results_baseline.json results_${TIMESTAMP}.json
```

---

## Quick Commands

```bash
# Fast iteration during development
alias pomai-test='cmake --build build && ctest --test-dir build -L integ'
alias pomai-bench='./build/comprehensive_bench --dataset small'
alias pomai-gate='./tools/perf_gate.sh --dataset=small'

# Full pre-commit check
alias pomai-ci='cmake --build build && ctest --test-dir build && ./tools/perf_gate.sh'
```

---

## References

- [Benchmarking Guide](BENCHMARKING.md)
- [SOT (Single Source of Truth)](SOT.md)
- Test source: `tests/integ/`, `tests/crash/`
- Tools: `tools/perf_gate.sh`, `tools/fuzz.sh`
