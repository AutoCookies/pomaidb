# Contributing to PomaiDB

PomaiDB is a local-first, single-machine vector database. Contributions are welcome if they reinforce correctness, deterministic recovery, and explicit contracts.

## Build matrix & toolchain
- **OS:** Linux (primary), WSL accepted.
- **Compilers:** GCC 10+ or Clang 11+ (C++20).
- **CMake:** 3.10+.

## Code style rules
- **Naming:**
  - Types: `PascalCase` (e.g., `ShardState`).
  - Functions/variables: `snake_case`.
  - Private members: trailing underscore (`mutex_`).
- **Includes:** keep minimal and sorted. No try/catch around imports.
- **Headers:** use `#pragma once` (existing convention).
- **RAII:** use RAII for resource ownership; avoid raw ownership.

## Error handling rules (standard)
- Public API must return `Status`/`Result<T>`.
- Exceptions must not cross API boundaries.
- No silent failures: log or return a `Status` with a clear error message.
- Quality mode must not return success on insufficient results.

## Thread safety rules
- **No locks on the hot read path.** Readers operate on immutable snapshots (`ShardState`).
- **Immutable view rules:** only publish new views via atomic swap; never mutate a published view.

## Commit conventions
- Small, focused commits.
- Format: `<area>: <intent>` (e.g., `core: normalize filter contract`).

## How to run

### Build
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Tests
```bash
ctest --test-dir build --output-on-failure
./tests/test.sh
```

### Sanitizers
```bash
cmake -S . -B build-asan -DPOMAI_SANITIZE=ASAN
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

### Formatting / lint
- Not implemented in this repository. Follow existing style and keep diffs small.

### Bench sanity checks
```bash
./build/bench_embedded --queries=100 --topk=10
./build/bench_wal --threads 2 --duration 5
```

## PR checklist
- [ ] Updated docs if behavior or contracts changed.
- [ ] Added/updated tests.
- [ ] No perf regressions (include benchmark output for hot-path changes).
- [ ] Error paths return `Status`/`Result<T>` and are not silent.

## Issue templates (guidance)
For bug reports, include:
- Exact build command and compiler version.
- `ctest --output-on-failure` output.
- Benchmark output if the issue is performance-related.
- Steps to reproduce and minimal dataset if possible.
