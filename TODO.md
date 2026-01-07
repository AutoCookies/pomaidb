Tóm tắt ngắn (1 câu)
- Bắt đầu bằng: (1) cơ chế lưu vector (blob indirect trong Arena), (2) API put_vector/get_vector (binary‑safe), (3) brute‑force KNN (per‑shard) để có tính năng ngay; sau đó (4) bổ sung index ANN (HNSW/IVF/PQ), compaction/free‑blob, replication & persistence, metrics/monitoring, và query routing/sharding.

1) Thiết kế dữ liệu: lưu Vector/Embedding
- Representation:
  - dtype: float32 (Mặc định). Có thể mở hỗ trợ float16/int8 sau.
  - store vector bytes contiguously in Arena as blob: [uint64_t len_bytes][payload bytes...] — we already have alloc_blob that does this.
  - Seed payload stores the key and a pointer to blob (indirect flag). Seed.header klen holds key length; seed header vlen==0 for indirect.
  - In addition, for vectors we must store metadata: dimension (uint32), flags (dtype), maybe layout version. That lives in blob header (preceding the payload): e.g. Blob layout:
    - uint64_t payload_len (existing)
    - uint32_t vec_dim
    - uint8_t dtype (0=float32,1=float16,etc)
    - padding
    - raw vector bytes
- Why: cheap to implement, uses existing alloc_blob, works for arbitrary-size vectors.

2) API surface (C++ functions to add to PomaiMap)
- put_vector(const char* key, size_t klen, const float* vec, uint32_t dim)
- const float* get_vector(const char* key, size_t klen, uint32_t *out_dim) — returns pointer into blob
- knn_query_bruteforce(const float* q, uint32_t dim, size_t k, std::vector<pair<key,score>>& out, float (*metric)(...))
- (optional) put_vector_with_ttl, update_vector_inplace if same dim and inline/indirect can be rewritten.

3) Brute‑force knn (initial, per-shard)
- Implementation: scan all Seeds in the hash table (or iterate arena allocated seeds), for each seed with type==VECTOR and same dim, compute distance (dot/cosine or L2) between q and candidate, keep a size‑k max‑heap (or partial sort).
- Performance: O(N * dim) — good for small N or for sharded subsets; use SIMD/BLAS later.
- Deterministic & simple; works while you implement ANN.

4) ANN index plan (next stage)
- Options:
  - HNSW: high recall, fast queries, dynamic inserts. Implementation non-trivial but doable without deps (there are reference HNSW implementations). Good when you need low‑latency, high‑recall.
  - IVF+PQ (Inverted File + Product Quantization): more compact, good for large corpora, needs offline training.
  - Graph-based HNSW is recommended for initial production path.
- Approach:
  - Implement HNSW per‑collection (per logical vector set). Keep index metadata in arena (nodes store vector id and links).
  - For sharding: build per‑shard HNSW over shard's vectors, route queries to all shards (fan-out) or use a lightweight router that maps keys to specific shard, or use two‑stage search (coarse quantizer then shard).
- For zero‑dependency you can port a minimalist HNSW variant. Alternatively, allow optional linking to hnswlib if user opts in.

5) Memory discipline & freebies (you asked earlier)
- We already added free_seeds; ensure:
  - On erase() of vector keys, call arena->free_seed(s).
  - For blob space: implement blob free list or slab allocator for blobs (later). Bump allocator for blobs is OK at first but must be improved for long‑running workloads with churn — implement free_blob(ptr) that pushes offset into free_blob_buckets keyed by size class.
- Compaction: background compaction can coalesce free blobs by copying survivors (complex). Better: pre-segment arenas by lifetime or use slab sizes.

6) Eviction policy adaptations for vectors
- Vectors are large: harvest should be size-aware — when sampling victims, prefer cold seeds with small cost to evict? Actually you want to free more bytes: victim selection should incorporate payload size (indirect blob size) and entropy: select victim with smallest score = entropy / size or simply minimal entropy × weight(size).
- Also track total bytes used by arena and per‑seed blob size (put into metrics).

7) Protocol & binary safety (you asked to remove strtok)
- We already made server use a length‑prefixed binary protocol for SET/GET/DEL/INFO. For vector API we can reuse:
  - Add op codes:
    - 'V' + subop: e.g., 'A' = add vector (PUT_VECTOR), 'Q' = query (KNN)
  - For PUT_VECTOR frame: [1B 'V'][1B sub='A'][4B keylen][key][4B dim][4B bytes_of_vector? or dim*4][vector bytes]
  - For KNN query frame: [1B 'V'][1B sub='Q'][4B k][4B dim][vector bytes]
  - Server returns binary list of (keylen,key,score) with length prefixes.
- This is binary-safe and simple. We should design client helpers to make it easy to test.

8) Persistence, WAL & snapshots (later)
- Snapshot (RDB style): periodic dump of arena content (serialize Seeds + blob payloads).
- WAL: append-only log of operations (SET/DEL/PUT_VECTOR) so on restart you can replay. WAL must record raw bytes; keep small sync frequency and rotate.
- For initial version, snapshot-only may be OK. WAL helps durability.

9) Sharding & routing for vector queries
- Approaches:
  - Hash partitioning by key: vector data pinned to shard by key; queries must broadcast to all shards unless you have a routing mechanism.
  - Routing by coarse quantizer: client or front-end uses coarse index to decide which shards to query.
- Start with per‑key sharding (simple). For KNN queries broadcast to all shards and aggregate top‑k — works up to moderate shard count.

10) Tests & Benchmarks
- Unit tests:
  - put/get vector with various dims and binary contents
  - indirect path test already exists
  - knn_bruteforce correctness test (small dataset)
  - deterministic harvest test with seed_rng
- Benchmarks:
  - SIFT 128‑dim, Deep1B style: measure throughput & latency for KNN brute force vs HNSW.
  - Measure memory usage and eviction behavior under churn.

11) Metrics & ops
- Add metrics per‑shard: vector_count, total_vector_bytes, knn_queries, knn_latency histogram.
- Add INFO output to include these fields.

12) Security & safety
- Validate incoming lengths and reject big allocations to prevent OOM attacks.
- Limit per‑connection / per‑client quotas.

13) Concrete next steps (code tasks I can implement for you now)
Pick one (or ask me to do multiple in series). I can implement immediately:

A) Implement vector storage + API + brute‑force KNN in code:
   - Add functions to PomaiMap: put_vector/get_vector/knn_bruteforce
   - Add server protocol frames for vector PUT/QUERY (binary ops)
   - Add no‑gtest tests (binary) to exercise vector put/get and knn_bruteforce
   - ETA: ~6–12 hours (depending on desired optimization)

B) Implement blob free list (free_blob + free list buckets) so blob memory can be reclaimed on delete — ETA ~4–8 hours.

C) Implement HNSW prototype (simple, minimal) for ANN — ETA larger (2–5 days), I recommend after A+B.

D) Add WAL + snapshot persistence — ETA few days.

Which do you want me to implement next? My recommended immediate order:
1) A (vector storage + brute force KNN + protocol), then
2) B (blob free list), then
3) add HNSW when you need scale.

If you choose A now I will:
- Add functions in core/map.h (put_vector/get_vector/knn_bruteforce) with code that uses existing alloc_blob and metrics.
- Update server.h to accept new 'V' op with subops for PUT_VECTOR and KNN_QUERY and reply with length-prefixed results.
- Add tests/tests_vector_no_gtest.cc that:
  - puts a few vectors,
  - queries knn (k=2) and asserts results (brute force).
- Update tests/run_tests.sh to compile the new test.

Tell me: “Implement A” (I’ll produce the code changes), or pick another option. If “Implement A”, confirm:
- Default metric/distance: L2 squared or cosine? (I’ll implement L2 by default and allow cosine via parameter).
- Default dtype: float32? (I’ll use float32).