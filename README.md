# PomaiDB

<p align="center">
  <img src="assets/logo.png" alt="PomaiDB Logo" width="200">
  <br>
  <i><b>The vector database for humans, not just machines.</b></i>
</p>

---

AI should not require expensive machines.  
**PomaiDB** makes vector search possible on hardware everyone already owns:  
old laptops, edge boxes, offline workstations, and humble VMs.

---

## Why PomaiDB Exists

Modern AI infrastructure assumes expensive hardware and abundant resources.

**PomaiDB challenges that assumption.**

**We believe:**
- AI should run on machines in real life, not just in datacenters.
- Infrastructure should adapt to resource limits, not crash.
- Systems must degrade gracefully, never fail catastrophically.
- Everyone—not just big tech—should have access to fast, usable vector search.

---

## Overview

PomaiDB exists because **most modern AI infrastructure assumes abundance**—RAM, CPU, connectivity.  
We know that **data lives everywhere**, including the edge, developing markets, offline offices, and underpowered laptops.

PomaiDB is a **high-performance, embedded-ready vector database** written in modern C++20.  
It is designed for **low-latency similarity search on commodity hardware**, using a custom "Pomegranate" architecture that combines multi-schema support, lock-free indexing, WAL, and SIMD-accelerated quantization—all with **zero external dependencies**.

> **PomaiDB treats the OS as a collaborator, not an enemy:** > It leverages the page cache, mmap, and atomic file semantics, refusing to "fight" Linux.

## Who PomaiDB Is For

PomaiDB is built for:
- Developers running AI on old laptops or small VMs
- Teams deploying vector search on edge or offline systems
- Engineers who care about stability more than benchmarks
- Anyone who believes AI infrastructure should be humane

## What Makes PomaiDB Different

PomaiDB does not aim to win raw benchmarks.
It aims to survive.

- If resources drop, PomaiDB degrades gracefully.
- If load spikes, PomaiDB protects the system.
- If hardware is weak, PomaiDB adapts — not crashes.

## Core Architecture and Algorithms

### 1. **Multi-Membrance Storage**
Not just multi-index, but **multi-membrance**: PomaiDB divides vectorspaces into fully isolated logical areas ("membrances"). Each membrance can have its own dimensionality, RAM limits, and persistence.

Why?  
**To support real concurrent workloads, multi-tenancy, and test/dev isolation—without "locking the world” for every schema change.**

---

### 2. **Pomegranate Indexing (IVF + Adaptive Routing)**
Instead of dogmatic HNSW/IVF, PomaiDB fuses strong ideas:

- **Centroid Initialization:** via K-Means++ for consistent clustering.
- **Dynamic Bucketing:** Vectors packed by closest centroid, stored via lock-free allocators.
- **Routing Graph:** Centroids are connected similar to HNSW, allowing fast search without brute-forcing all clusters.

Why?  
**To combine the practical strengths of both "global cluster first" and "navigate like a human"—without copying untuned academic code.**

---

### 3. **SynapseCodec (4-bit Compression, Delta Quantization)**
Not all vectors merit 32-bit floats in RAM.  
PomaiDB quantizes deltas between a vector and its centroid using **4-bit nibbles** (~8x RAM savings).  
Distance is approximated using SIMD LUTs; exact refinement is possible on-demand.

Why?  
**Because the trade-off between RAM, speed, and recall must be tunable—especially on small machines.**

---

### 4. **SimHash Prefiltering**
Before running L2/Dot-Product on candidates, PomaiDB computes 512-bit SimHash fingerprints.  
Candidates with a high Hamming distance to the query are _immediately_ discarded using POPCNT.

Why?  
**Because sometimes approximate recall is enough, and your CPU cycles are precious.**

---

### 5. **ShardArena Allocator**
Custom bump-pointer allocator with...

- **mmap, Huge Pages:** for large object allocation and minimal fragmentation.
- **Zero-Copy Persistence:** “Freezing” a bucket to disk is just a remapping—no serialization ceremony.
- **Atomic Offsets, Lock-Free Readers:** Massive throughput, even under light contention.

**Design Philosophy:** PomaiDB relies on the OS, trusting page cache and atomic renames, not reinventing "mini-filesystems" inside a user process.

---

### 6. **Write-Ahead Log (WAL) and Crash Recovery**
Not “just for show.” WAL is implemented with:

- **Simple, CRC32-checksummed logs.** - **Atomic file operations** (rename patterns).
- **Crash resilience:** DB is consistent after power loss (we replay the WAL until successful).

Why?  
**Because reliability is not optional, even on the edge.**

---

### 7. **WhisperGrain: Energy Operating System**

PomaiDB is not another “dumb” search engine hard-coded to eat RAM.  
The **WhisperGrain** controller transforms PomaiDB into a living system:

- Every search and vector op is translated to "ops" (operation units).
- A dynamic budget—based on real-time latency, CPU, and system health—decides how hard to try, when to degrade, when to do exact refine.
- The result?  
  - **PomaiDB simply does not crash under load:** Quality and recall are traded _gracefully_ for health.
  - **No wild tail-latency spikes:** Your device is safe, no matter what abuse you give it.

Why?  
**Because AI should fade gracefully on bad days, not ruin your system.**

---

### 8. **Dataset Engine & Split Strategies (New in V3)**
PomaiDB is now a full-fledged **AI Data Engine**, capable of understanding your data's semantics to prepare training sets automatically. It supports 4 powerful splitting strategies:

- **Random Split:** Standard shuffle for general purpose data.
- **Stratified Split:** Balances class distribution (e.g., maintain 80/20 ratio for rare classes).
- **Cluster Split (Spatial):** Splits data by geometric clusters (e.g., Train on Day/Night, Test on Rain) to test model generalization.
- **Temporal Split:** Splits data by time (e.g., Train on Jan-Oct, Test on Nov-Dec) to prevent data leakage in time-series tasks.

Why?
**Because a Vector DB should help you build better models, not just store numbers.**

---

### 9. **AI Contract & Streaming (New in V3)**
- **AI Contract:** `GET MEMBRANCE INFO` now returns a semantic contract (dimension, metric, split sizes) so ML frameworks (PyTorch/TensorFlow) can auto-configure without manual hard-coding.
- **Binary Streaming:** The `ITERATE` command pumps training data directly from disk to GPU via a high-performance binary protocol, bypassing slow Python loops.

Why?
**Because MLOps should be automated and standardized.**

---

## Performance Benchmarks

Tested on Dell Latitude E5440 (2 Cores, 8GB RAM):

- **Dataset:** 1,000,000 vectors (512-dim)
- **Search Latency (P50):** ~0.46ms
- **P99.9:** < 0.60ms

> **Note:** These figures are averages under controlled load, with recall and nprobe adaptively traded for latency. See WhisperGrain for details.


## Building from Source

**Prerequisites:** - GCC/Clang with C++20
- Linux/macOS (Windows via WSL)
- CMake ≥ 3.10, Make

```bash
chmod +x ./build.sh
./build.sh
```

Binaries:

* `pomai_server` — main server
* `pomai_cli` — SQL client
---

## Running the Server

```bash
chmod +x ./run.sh
./run.sh
```

**Env vars:**

* `POMAI_DB_DIR` = data root (`./data/pomai_db` default)
* `POMAI_PORT`   = override listen port (default: 7777)

---

## Using the CLI

```bash
./pomai_cli -h 127.0.0.1 -p 7777
```

---

## PomaiSQL Protocol

Custom, SQL-inspired. Commands end with `;`.

**Create Schema:**

```sql
CREATE MEMBRANCE name DIM N RAM MB;
SHOW MEMBRANCES;
```

**Insert/Select Context:**

```sql
USE myspace;
INSERT VALUES (photo_001, [0.12, 0.45, ...]);
-- Batch Insert with Tags
INSERT INTO myspace VALUES (p1, [...]), (p2, [...]) TAGS (class:dog, date:2024-01-01);
SEARCH QUERY ([0.12, 0.45, ...]) TOP 5;
```

**Retrieve or Delete:**

```sql
GET LABEL photo_001;
DELETE LABEL photo_001;
```

**Dataset Management (New):**

```sql
-- View AI Contract
GET MEMBRANCE INFO myspace; 

-- Split Dataset
EXEC SPLIT myspace 0.8 0.1 0.1;               -- Random
EXEC SPLIT myspace 0.8 0.1 0.1 STRATIFIED class; -- Balanced by class
EXEC SPLIT myspace 0.8 0.1 0.1 TEMPORAL date;    -- Time-ordered
```

**Bulk Ingest:**

```sql
LOAD BINARY '/path/vectors.bin' INTO myspace;
```

## Why Use PomaiDB?

* Because your data is *not* always in the cloud.
* Because not everyone has a 128GB server.
* Because crash-only design is unacceptable for real users.
* Because you want a database that fights for you—not the other way around.

## Benchmarks

We run this benchmarks on Github Codespace, with 4 cores CPU, and 16Gb RAM. By inserting, searching, filtering, delete 512 dimensions vectors.

<p align="center">
<img src="assets/baseline_cdf_last.png" alt="PomaiDB Logo" width="600">





</p>

<p align="center">
<img src="assets/latency_percentiles_vs_N.png" alt="PomaiDB Logo" width="600">





</p>

<p align="center">
<img src="assets/recall_vs_latency.png" alt="PomaiDB Logo" width="600">





</p>

<p align="center">
<img src="assets/throughput_vs_N.png" alt="PomaiDB Logo" width="600">





</p>

<p align="center">
<img src="assets/sys_mem_time.png" alt="PomaiDB Memory through time" width="600">





</p>

<p align="center">
<img src="assets/sys_cpu_time.png" alt="PomaiDB Memory through time" width="600">





</p>

## License

PomaiDB is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

See `LICENSE` for details.

> We deeply encourage both academic and practical use,
> but ask you to **credit the original author (Quan Van)** and **contribute improvements** for the benefit of the community.

---

## Citation

If you use PomaiDB in academic or research projects:

```
@misc{pomai,
  author={Quan Van},
  title={PomaiDB: Vector Search for Every Machine},
  url={[https://github.com/AutoCookies/pomai](https://github.com/AutoCookies/pomai)},
  year={2026}
}

```

## A Final Word

**PomaiDB is not just code.** It is a manifesto:

* That *state-of-the-art* belongs to everyone.
* That *robustness* is possible, even on weak hardware.
* That a database can be proud of what it doesn’t demand.

Welcome to the new edge of AI.


## How to run sync policy test
Run this block at a time
```
# 1) confirm binary exists
ls -l build/pomai_db_writer || ls -l build/pomai_db_writer*

# 2) prepare tmp dir and run writer (5 batches x 10 vectors)
rm -rf /tmp/pomai_dbg
mkdir -p /tmp/pomai_dbg
./build/pomai_db_writer /tmp/pomai_dbg 1 5 10 8 | tee /tmp/pomai_dbg/writer.stdout

# 3) show captured outputs and files
echo "---- writer.stdout ----"
sed -n '1,200p' /tmp/pomai_dbg/writer.stdout || true

echo "---- writer.status ----"
ls -l /tmp/pomai_dbg/writer.status || true
cat /tmp/pomai_dbg/writer.status || true

echo "---- ls -la /tmp/pomai_dbg ----"
ls -la /tmp/pomai_dbg || true

echo "---- hexdump shard-0.wal (first 128 bytes) ----"
hexdump -C -n 128 /tmp/pomai_dbg/shard-0.wal || true

echo "---- replay inspect ----"
/bin/true && ./build/wal_replay_inspect /tmp/pomai_dbg 8 || true
```

Then run ```./test/test_sync_policy.sh```