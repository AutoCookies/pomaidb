## PomaiDB on Edge Devices: Recommended Settings & Failure Semantics

PomaiDB is designed first for embedded / edge workloads: single-process, local storage, constrained memory, and frequent power loss. This guide summarizes **recommended configuration presets** and **what happens on failure** so you can reason about behavior on devices like Raspberry Pi, Jetson, or custom ARM boards.

This document focuses on the **embedded `pomai::Database` API** (single-instance engine) and the **sharded `pomai::DB` API** where relevant.

---

### 1. Build profile and compiler settings

- **Edge build profile (size-optimized):**
  - Configure CMake with:
    - `-DPOMAI_EDGE_BUILD=ON` (enables `-Os -g0` and other size-focused flags)
    - `-DCMAKE_BUILD_TYPE=Release`
  - Recommended for production firmware images and containers where binary size and cold-start latency matter more than debug info.

- **Strict warnings for development and CI:**
  - Enable:
    - `-DPOMAI_STRICT=ON`
  - This turns most compiler warnings into errors for PomaiDB’s own code while keeping vendored dependencies (HNSW, SIMD kernels) lenient.
  - Safe to combine with `POMAI_EDGE_BUILD` once your toolchain is stable; it helps surface misconfigurations early.

---

### 2. Storage and durability settings

PomaiDB stores all data under a **single directory** on local storage (e.g., SD card, eMMC, SSD).

- **Filesystem & mount:**
  - Prefer **ext4** or another journaling filesystem with barriers enabled.
  - Avoid network filesystems for embedded use; PomaiDB assumes low-latency local I/O.

- **Durability via `FsyncPolicy`:**
  - For the sharded `pomai::DB` API (`pomai::DBOptions`):
    - `FsyncPolicy::kNever`:
      - Best for **cache-like or reconstructible** data.
      - Power loss may drop recent writes still in OS buffers, but on-disk data remains self-consistent.
    - `FsyncPolicy::kAlways`:
      - Recommended when **data must survive power loss** and write rates are modest.
      - Every WAL / manifest commit is fsynced; expect higher latency but strong durability.
  - For the embedded `pomai::Database` API (`pomai::EmbeddedOptions`):
    - Use `EmbeddedOptions::fsync` in the same way.
    - On intermittently powered devices, prefer `kAlways` for critical logs and `kNever` where data can be rebuilt.

- **Flush vs. Freeze:**
  - `Flush()` ensures the **WAL is pushed to disk** according to `FsyncPolicy`.
  - `Freeze()` moves the current memtable into an on-disk **segment** and updates manifests.
  - On edge devices, a common pattern from an event loop or watchdog is:
    - Periodically call `Flush()` and `Freeze()` on a timer (e.g., every N seconds) or after M ingests.
    - On clean shutdown, issue `Flush()` and `Freeze()` before `Close()`.

For the detailed atomic commit protocol and WAL / manifest guarantees, see `docs/FAILURE_SEMANTICS.md`.

---

### 3. Memory limits and backpressure (embedded `pomai::Database`)

`pomai::Database` exposes **explicit backpressure controls** in `EmbeddedOptions`:

- **Key fields:**
  - `max_memtable_mb`:
    - Hard cap for the memtable (in MiB). `0` = use environment or default:
      - Default is tuned for edge and may differ between low-memory and normal builds.
  - `pressure_threshold_percent`:
    - Soft threshold (percent of `max_memtable_mb`) where pressure handling kicks in. `0` = default (typically 80%).
  - `auto_freeze_on_pressure`:
    - If `true`, when the memtable exceeds the pressure threshold, the engine will **call `Freeze()` internally** rather than returning an error.
  - `memtable_flush_threshold_mb`:
    - Absolute size in MiB where `auto_freeze_on_pressure` triggers, overriding the percentage. `0` = derive from `pressure_threshold_percent`.

- **Recommended presets for edge:**
  - **Tiny devices (≤ 256 MiB RAM):**
    - `max_memtable_mb = 32`–`64`
    - `pressure_threshold_percent = 70`–`80`
    - `auto_freeze_on_pressure = true`
    - `memtable_flush_threshold_mb = 32` (optional override)
  - **Moderate devices (512 MiB – 1 GiB RAM):**
    - `max_memtable_mb = 128`–`256`
    - `pressure_threshold_percent = 80`
    - `auto_freeze_on_pressure = true` (recommended) or `false` if you want manual control via `TryFreezeIfPressured()`.

- **Environment overrides:**
  - The embedded engine also honors:
    - `POMAI_MAX_MEMTABLE_MB` – caps memtable size if `max_memtable_mb` is `0`.
    - `POMAI_MEMTABLE_PRESSURE_THRESHOLD` – overrides `pressure_threshold_percent` for defaults.
    - `POMAI_BENCH_LOW_MEMORY` – switches to lower default memtable sizes for benchmarks / tests.

- **Operational pattern:**
  - In a single-threaded event loop, the typical pattern is:
    - Call `AddVector()` / `AddVectorBatch()` for ingestion.
    - Periodically call `TryFreezeIfPressured()` to keep memory use bounded.
    - Inspect `GetMemTableBytesUsed()` for metrics / logging.

---

### 4. Index and quantization presets for low memory

PomaiDB’s `IndexParams` exposes presets tuned for edge workloads:

- **Use `IndexParams::ForEdge()` wherever possible:**
  - In `EmbeddedOptions`:
    - `opt.index_params = pomai::IndexParams::ForEdge();`
  - This preset reduces:
    - IVF list count (`nlist`), probes (`nprobe`),
    - HNSW degree / ef parameters,
    - and other memory-heavy knobs.
  - The goal is to keep index RAM usage predictable while still providing reasonable recall.

- **Distance metric:**
  - For most embedding-style workloads on edge devices:
    - Use `MetricType::kL2` (squared L2) with SQ8 or FP16 quantization for compact storage.
  - `MetricType::kInnerProduct` is also supported but may be more sensitive to quantization.

- **Quantization knobs (when applicable):**
  - Prefer SQ8 or FP16 quantization where your model tolerates some loss, especially for:
    - Large corpora on devices with ≤ 512 MiB RAM.
    - Scenarios where on-disk size is heavily constrained (e.g., SD cards with many tenants).

---

### 5. Failure semantics on edge devices

PomaiDB is built to **fail closed** rather than risking silent corruption. High-level behaviors (see `docs/FAILURE_SEMANTICS.md` for details):

- **On `Open()` (embedded `Database::Open` / sharded `DB::Open`):**
  - Invalid configuration (e.g., `dim == 0`, empty `path`) returns:
    - `Status::InvalidArgument`.
  - Filesystem errors (permissions, missing dirs that cannot be created) return:
    - `Status::IOError`.
  - WAL or manifest corruption:
    - The engine attempts to **replay or recover**.
    - If recovery is not possible, `Open()` returns a non-OK `Status` (e.g., `Corruption`, `Aborted`, or `Internal` depending on context) and **does not start** the engine.

- **During ingestion / search:**
  - **Backpressure (embedded engine):**
    - If the memtable exceeds `max_memtable_mb` and `auto_freeze_on_pressure` is `false`:
      - `AddVector` / `AddVectorBatch` will return `Status::ResourceExhausted` with a message instructing callers to `Freeze()` or `TryFreezeIfPressured()`.
    - If `auto_freeze_on_pressure` is `true`:
      - The engine attempts to `Freeze()` internally once pressure is detected.
      - If freeze fails (e.g., I/O error), the operation returns the corresponding failure `Status`.
  - **I/O failures (ENOSPC, EIO, etc.):**
    - Write failures on WAL / segments propagate as:
      - `Status::IOError` or `Status::Aborted` / `Status::Internal`, depending on the layer.
    - After a serious I/O error, affected shards / the embedded engine will refuse further operations until reopened, to avoid compounding corruption.

- **Crash and restart behavior:**
  - On restart, both APIs:
    - Re-open WALs and attempt **replay up to the last valid record**.
    - Validate manifests and segment files; fall back from `manifest.current` to `manifest.prev` if needed.
  - Tests such as `recovery_test`, `manifest_corruption_test`, and WAL corruption scenarios validate the following guarantees:
    - No silent acceptance of corrupted manifests or WAL segments.
    - Either **recover to a consistent state** (possibly losing a tail of recent writes) or **fail to open** with a non-OK `Status`.

---

### 6. Operational recommendations for real devices

- **Choose a failure policy per device class:**
  - For sensor nodes with upstream replicas:
    - Prefer `FsyncPolicy::kNever`, small `max_memtable_mb`, and `auto_freeze_on_pressure = true`.
    - Rely on upstream for long-term durability.
  - For gateway / aggregation devices:
    - Prefer `FsyncPolicy::kAlways` for critical data.
    - Use `IndexParams::ForEdge()` and conservative `max_memtable_mb` to bound RAM.

- **Integrate health checks:**
  - Treat **any non-OK `Status` from `Open()`** as a signal to:
    - Log and raise an alert.
    - Potentially rotate to a new storage path or device.
  - Monitor:
    - `GetMemTableBytesUsed()`
    - Open / search error codes (e.g., `ResourceExhausted`, `IOError`, `Corruption`).

- **Test on your actual target:**
  - Run the existing integration, TSAN, and crash tests on:
    - Your device type, filesystem, and kernel.
  - Perform your own chaos test:
    - Ingest + `Flush()` / `Freeze()` loop.
    - Physically cut power or kill the process.
    - Verify that:
      - `Open()` either succeeds with intact historical data or fails with a clear error code.

These guidelines are intentionally conservative: they aim to keep your edge deployments safe even under frequent power loss and tight memory budgets.

