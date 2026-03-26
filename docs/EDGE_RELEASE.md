# Edge release 1 — criteria, membrane capabilities, bindings

This document is the **release bar** for shipping PomaiDB as an **embedded multi-modal database on edge** (single-process, constrained RAM/flash). It complements operational guidance in [EDGE_DEPLOYMENT.md](EDGE_DEPLOYMENT.md).

---

## 1. Membrane capability matrix

Capabilities are also exposed in code: `pomai::GetMembraneKindCapabilities` ([membrane_capabilities.h](../include/pomai/membrane_capabilities.h)), and the C API `pomai_membrane_kind_capabilities` / `pomai_membrane_capabilities_init` ([c_api.h](../include/pomai/c_api.h)).

| Kind | API name | Read path | Write path | Unified scan | Snapshot-isolated scan¹ | Stability² |
|------|----------|-----------|------------|--------------|-------------------------|------------|
| `kVector` | vector | Yes | Yes | Yes | **Yes** | Stable |
| `kRag` | rag | Yes | Yes | Yes | No | Stable |
| `kGraph` | graph | Yes | Yes | Yes | No | Stable |
| `kText` | text | Yes | Yes | Yes | No | Stable |
| `kTimeSeries` | timeseries | Yes | Yes | Yes | No | Stable |
| `kKeyValue` | keyvalue | Yes | Yes | Yes | No | Stable |
| `kSketch` | sketch | Yes | Yes | Yes | No | Stable |
| `kBlob` | blob | Yes | Yes | Yes | No | Stable |
| `kMeta` | meta | Yes | Yes | Yes | No | Stable |
| `kSpatial` | spatial | Yes | Yes | Yes | No | Experimental |
| `kMesh` | mesh | Yes | Yes | Yes | No | Experimental |
| `kSparse` | sparse | Yes | Yes | Yes | No | Experimental |
| `kBitset` | bitset | Yes | Yes | Yes | No | Experimental |

¹ **Snapshot-isolated scan**: only **kVector** uses the same point-in-time snapshot iterator as `pomai_scan` / `NewIterator`. All other kinds read **live engine state** during `pomai_membrane_scan` / `NewMembraneRecordIterator` — suitable for export/ops, not for strict cross-membrane transactional isolation. See [membrane_iterator.h](../include/pomai/membrane_iterator.h).

² **Stable** = API and semantics intended for production; breaking changes require semver/ABI notes. **Experimental** = SIMD-heavy or rapidly evolving surfaces; still covered by tests, but callers should pin versions and expect tighter iteration.

---

## 2. Success metrics (release gate)

Targets are **qualitative gates** unless a CI job enforces a number. Baseline sizes must be **recorded** per release tag (see `scripts/edge_release_print_sizes.sh`).

### 2.1 Binary footprint

| Artifact | Gate | Notes |
|----------|------|--------|
| `libpomai_c.so` (shared) | **Record** size per release tag | Run `scripts/edge_release_print_sizes.sh`. Investigate **> ~10% growth** without an intentional feature (regression guard). Typical Release builds are **single-digit MiB** for the C shim + core; full tree with optional ML stacks can be much larger — always compare against the **previous tag** on the same toolchain. |
| `libpomai.a` (static) | Baseline only | Link only what you need; edge firmware often static-links a subset. |

Use `-DPOMAI_EDGE_BUILD=ON` for size-oriented builds. Optional components (e.g. heavy ML paths) may justify larger binaries if called out in release notes.

### 2.2 Latency (embedded vector path, reference class hardware)

Measured on a **single-threaded** workload, default HNSW settings, warm cache after open. Values are **targets** for regression tracking, not guarantees on every device.

| Operation | p99 target (edge class³) |
|-----------|---------------------------|
| `Put` (single vector, 128-d, SQ8 path) | ≤ **5 ms** |
| `Search` (top-10, single query, 100k–1M vectors shard) | ≤ **50 ms** |
| `Freeze` (typical memtable → segment) | ≤ **500 ms** (device-dependent) |

³ e.g. Raspberry Pi 4 class vs laptop-class edge: scale targets in internal benchmarks; record device in benchmark output.

### 2.3 Reliability / crash acceptance

The following **must pass** in CI for a release candidate (when `POMAI_BUILD_TESTS=ON`, `POMAI_ENABLE_CRASH_TESTS=ON`; crash tests are off under TSAN):

| Test | Purpose |
|------|---------|
| `pomai_crash_replay` | Randomized crash mid-write; reopen and validate consistency (`tests/crash/crash_replay_test.cc`). |
| `power_loss_stress` | Simulated power-loss rounds (`tests/crash/power_loss_stress_test.cc`). |
| `recovery_test` | WAL replay and recovery (`tests/crash/recovery_test.cc`). |
| `manifest_corruption_test`, `sd_fault_injection_test`, `new_membrane_fault_test` | Fault tolerance; must not abort the process on bad tails. |

Optional soak: `db_soak_stress_test` for long-run stability on release branches.

---

## 3. Foreign bindings roadmap (edge apps)

Priority reflects typical embedded stacks: **memory-safe systems languages first**, then Go (agents/cloud-edge bridges), then Python (tooling and notebooks — already shipped as `pomaidb` ctypes).

| Phase | Language | Deliverable | Status |
|-------|----------|-------------|--------|
| **1** | **Rust** | `-sys` crate linking `libpomai_c`, then `pomaidb` safe wrapper: open/put/search/membrane scan + capabilities | Planned |
| **2** | **Go** | `cgo` package `pomaidb` with the same core surface; static-friendly build tags | Planned |
| **3** | **Python** | Extend official `pomaidb` package beyond vectors (`membrane_kind_capabilities` and membrane scan already callable from ctypes when the library is new enough) | In progress |

**Rule**: C ABI (`include/pomai/c_api.h`, `c_types.h`) is the **stability boundary**; bindings must not rely on C++ symbols.

---

## 4. Related docs

- [EDGE_DEPLOYMENT.md](EDGE_DEPLOYMENT.md) — profiles, fsync, memory limits.
- [FAILURE_SEMANTICS.md](FAILURE_SEMANTICS.md) — WAL/manifest behavior.
- [VERSIONING.md](VERSIONING.md) — ABI struct `struct_size` pattern.
- [COMPAT_MATRIX.md](COMPAT_MATRIX.md) — OS/CPU/GPU compatibility summary.

---

## 5. Release artifact policy (operational)

Each tagged release (`v*`) publishes:
- `pomaidb-<os>-<arch>.tar.gz`
- `<asset>.sha256`
- `<asset>.sig` and `<asset>.pem` (keyless cosign signature/cert)

Bundle contents should include:
- `pomaidb_server`
- `pomaictl`
- `include/pomai/*`
- available runtime/static libraries for the target platform

CI (`run-all-tests`) additionally uploads a `ci-fast` Linux artifact for quick download and validation after every push/PR.
