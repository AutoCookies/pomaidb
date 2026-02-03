# PomaiDB Operations Runbook

## 1. Build Instructions

### Prerequisites
- CMake 3.20+
- C++20 compliant compiler (GCC 11+, Clang 12+, MSVC 19.29+)
- Linux (recommended) or macOS

### Debug Build (Development)
Enable tests, ASAN, and debug symbols.
```bash
cmake -S . -B build_debug \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPOMAI_BUILD_TESTS=ON \
    -DPOMAI_ENABLE_CRASH_TESTS=ON \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build_debug -j
```

### Release Build (Production)
Optimized, no tests, no sanitizers.
```bash
cmake -S . -B build_release \
    -DCMAKE_BUILD_TYPE=Release \
    -DPOMAI_BUILD_TESTS=OFF
cmake --build build_release -j
```

## 2. Testing

### Run All Unit & Integration Tests
```bash
cd build_debug
ctest --output-on-failure -L "unit|integ"
```

### Run Crash Safety Suite (Long Running)
This suite verifies data consistency across 50 simulated crashes.
```bash
cd build_debug
ctest --output-on-failure -R pomai_crash_replay
```
**Success Criteria**: All 50 rounds pass with "Verifying consistency..." logs showing no data loss.
**Failure Analysis**:
- If `Data Loss detected!`: The WAL replay failed or fsync was not honored.
- If `Named membrane ... failed to open`: Persistence recovery logic is broken.

### Run TSAN (Thread Sanitizer)
Requires separate build configuration.
```bash
cmake -S . -B build_tsan \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPOMAI_BUILD_TESTS=ON \
    -DCMAKE_CXX_FLAGS="-fsanitize=thread"
cmake --build build_tsan -j
ctest --test-dir build_tsan --output-on-failure -L tsan
```
*Note: crash tests are disabled in TSAN mode.*

## 3. Inspection & Debugging

### Using `pomai_inspect`
The inspection tool helps diagnose corrupt files or verify proper writing.

**1. Checksum Verification**
Verify CRC32C integrity of any file (WAL, Segment, Manifest).
```bash
./build_debug/pomai_inspect checksum /path/to/db/membranes/default/wal_0_0.log
```

**2. Dump Manifest**
View the structure of the binary manifest file.
```bash
./build_debug/pomai_inspect dump-manifest /path/to/db/MANIFEST
```

## 4. Release Procedures

See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
