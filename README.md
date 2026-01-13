Here is the comprehensive `README.md` for PomaiDB. It is written in professional, technical English, focusing on architectural details and algorithmic implementations as requested, without the use of icons or emojis.

---

# PomaiDB

PomaiDB is a high-performance, embedded-ready vector database written in modern C++20. It is designed for low-latency similarity search on commodity hardware, utilizing a custom "Pomegranate" architecture that combines multi-schema support (Membrances), lock-free indexing, and SIMD-accelerated quantization.

## Overview

PomaiDB distinguishes itself by avoiding heavy external dependencies. It implements its own storage engine, memory allocator, and write-ahead logging (WAL) system. The system is optimized for "Edge AI" scenarios where RAM efficiency and CPU cycle conservation are critical.

## Core Architecture and Algorithms

### 1. Multi-Membrance Storage

Unlike traditional single-index vector stores, PomaiDB utilizes a **Multi-Membrance** architecture. A "Membrance" is an isolated vector space with its own dimensionality, RAM allocation, and storage path.

* **Isolation:** Operations on one membrance (e.g., `INSERT`, `SEARCH`) do not lock or affect others.
* **Management:** Membrances are managed via a persistent `manifest` file and protected by a `std::shared_mutex` at the database level, allowing concurrent reads while ensuring safety during `CREATE` or `DROP` operations.

### 2. The "Pomegranate" Indexing (IVF + HNSW-like Routing)

The core search engine (`PomaiOrbit`) uses an Inverted File (IVF) structure with several optimizations:

* **Centroid Initialization:** Uses **K-Means++** to initialize centroids, ensuring a statistically distributed starting point for clusters.
* **Dynamic Bucketing:** Vectors are assigned to buckets based on the nearest centroid. Buckets are allocated via the `ShardArena` allocator.
* **Routing Graph:** Centroids are connected in a graph structure similar to HNSW (Hierarchical Navigable Small World), allowing the search algorithm to quickly locate the nearest centroids (`nprobe`) without scanning the entire centroid list.

### 3. SynapseCodec (4-bit Quantization)

To maximize memory efficiency, PomaiDB does not always store full 32-bit floating-point vectors in the hot path.

* **Delta Compression:** It calculates the delta between the inserted vector and its assigned centroid.
* **4-bit Packing:** These deltas are compressed into 4-bit nibbles. This reduces memory footprint by approximately 8x compared to raw `float32` storage.
* **SIMD Lookup:** During search, a Look-Up Table (LUT) is precomputed using the query vector. Distances are then approximated using AVX2 SIMD instructions on the packed 4-bit data.

### 4. SimHash Prefiltering

Before performing expensive distance calculations (L2 or Dot Product), candidates are filtered using **SimHash** (Sign-bit Projections).

* **Fingerprinting:** A 512-bit fingerprint is generated for every vector using random hyperplanes.
* **Hamming Distance:** During search, the Hamming distance between the query fingerprint and candidate fingerprints is calculated using `POPCNT` instructions.
* **Early Rejection:** Candidates exceeding a Hamming distance threshold (default 140) are immediately discarded, significantly reducing CPU load.

### 5. ShardArena Memory Allocator

PomaiDB uses a custom bump-pointer allocator named `ShardArena`.

* **mmap & Huge Pages:** It allocates large blocks of memory using `mmap` (attempting `MAP_HUGETLB` for performance).
* **Zero-Copy Persistence:** Data can be "frozen" (demoted) to disk. The system uses memory mapping to read these files, relying on the OS page cache for lazy loading without explicit serialization overhead.
* **Lock-Free Reads:** Readers can traverse atomic offsets without acquiring locks, ensuring high read throughput even during write operations.

### 6. Write-Ahead Log (WAL)

Data integrity is ensured via a custom WAL implementation.

* **Structure:** Operations (Create/Drop Membrance) are appended to a log file with CRC32 checksums.
* **Crash Recovery:** On startup, the WAL is replayed to restore the database state.
* **Atomicity:** Critical file operations use atomic rename patterns to prevent corruption.

## Building from Source

### Prerequisites

* **Compiler:** GCC or Clang with C++20 support.
* **OS:** Linux (Recommended) or macOS. Windows support is experimental via WSL.
* **Tools:** CMake (3.10+), Make.

### Build Steps

```bash
# 1. Create a build directory
mkdir build
cd build

# 2. Configure the project
cmake ..

# 3. Compile (use -j for parallel build)
make -j$(nproc)

```

This will generate two binaries:

* `pomai-server`: The database server instance.
* `pomai-cli`: The interactive SQL client.

## Running the Server

Start the server by running the binary. It listens on port 7777 by default.

```bash
./pomai-server

```

**Environment Variables:**

* `POMAI_DB_DIR`: Set the root directory for data storage (default: `./data/pomai_db`).
* `POMAI_PORT`: Override the default listening port.

## Using the CLI

Connect to the server using the client tool:

```bash
./pomai-cli -h 127.0.0.1 -p 7777

```

## PomaiSQL Protocol Specification

PomaiDB uses a custom text-based protocol inspired by SQL. All commands must end with a semicolon (`;`).

### 1. Data Definition (DDL)

**Create a new Membrance (Schema):**

```sql
CREATE MEMBRANCE <name> DIM <dimension> RAM <size_in_mb>;
-- Example:
CREATE MEMBRANCE images DIM 512 RAM 1024;

```

**Drop a Membrance:**

```sql
DROP MEMBRANCE <name>;

```

**List all Membrances:**

```sql
SHOW MEMBRANCES;

```

### 2. Context Management

**Select a Membrance for subsequent operations:**

```sql
USE <name>;
-- Example:
USE images;

```

### 3. Data Manipulation (DML)

**Insert a Vector:**

```sql
-- Full syntax:
INSERT INTO <name> VALUES (<label>, [<v1>, <v2>, ...]);

-- Short syntax (after USE):
INSERT VALUES (<label>, [<v1>, <v2>, ...]);

-- Example:
INSERT VALUES (photo_001, [0.12, 0.45, 0.99, ...]);

```

**Search for Nearest Neighbors:**

```sql
-- Full syntax:
SEARCH <name> QUERY ([<v1>, <v2>, ...]) TOP <k>;

-- Short syntax (after USE):
SEARCH QUERY ([<v1>, <v2>, ...]) TOP <k>;

-- Example:
SEARCH QUERY ([0.12, 0.45, 0.99, ...]) TOP 5;

```

**Retrieve a Vector by Label:**

```sql
GET <name> LABEL <label>;
-- Short syntax:
GET LABEL <label>;

```

**Delete a Vector (Soft Delete):**

```sql
DELETE <name> LABEL <label>;

```

### 4. Bulk Operations

**Load Binary Data (Fast Ingestion):**
Loads a flat binary file directly into memory. The file format must be: Header [Magic|Count|Dim] followed by packed [Vector|Label] records.

```sql
LOAD BINARY '<path_to_file>' INTO <name>;

```

## Performance Benchmarks

Tested on commodity hardware (Dell Latitude E5440, 2 Cores, 8GB RAM):

* **Dataset:** 1,000,000 Vectors (512-dim).
* **Search Latency (P50):** ~0.46ms.
* **Search Latency (P99.9):** < 0.60ms.
* **Throughput:** Scalable O(1) complexity relative to dataset size due to IVF clustering.

## License

[Insert License Here]