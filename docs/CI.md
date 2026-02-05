# PomaiDB CI/CD Quick Reference

This guide reflects the **current** CI setup in `.github/workflows/ci.yml`.

## CI scope (current)

PomaiDB CI runs on **Linux only** (`ubuntu-latest`).

Jobs:
- `build-test-linux`: configure + build + ctest
- `tsan-linux`: TSAN-instrumented build and `-L tsan` tests
- `python-ffi-smoke`: build `pomai_c` shared library and run ctypes smoke test
- `perf-gate`: build performance harness and enforce baseline thresholds

## Run equivalent checks locally (Ubuntu)

```bash
# 1) Build + tests
cmake -S . -B build -DPOMAI_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# 2) TSAN
CC=clang CXX=clang++ cmake -S . -B build-tsan -DPOMAI_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS='-fsanitize=thread -fno-omit-frame-pointer' \
  -DCMAKE_CXX_FLAGS='-fsanitize=thread -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=thread' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=thread'
cmake --build build-tsan --parallel
ctest --test-dir build-tsan --output-on-failure -L tsan

# 3) Python FFI smoke
cmake -S . -B build -DPOMAI_BUILD_TESTS=ON
cmake --build build --target pomai_c --parallel
python3 tests/ffi/python_ctypes_smoke.py

# 4) Performance gate
cmake -S . -B build -DPOMAI_BUILD_TESTS=ON
cmake --build build --target ci_perf_bench --parallel
python3 tools/perf_gate.py --build-dir build --baseline benchmarks/perf_baseline.json --threshold 0.10
```

## Docker-assisted CI-like run

```bash
docker build -t pomaidb/dev:local .
docker run --rm pomaidb/dev:local
# or

docker compose up --build pomaidb-dev
```

## Notes

- The repository does not ship a standalone production DB server binary; CI validates the embedded library and C ABI artifacts.
- If you need macOS/Windows verification for a downstream integration, run it in that downstream project pipeline.
