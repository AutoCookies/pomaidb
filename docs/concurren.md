```markdown
# Pomai Concurrency Policy (single-thread-per-shard)

Design decision:
- Pomai uses a shard-per-core model: each shard (Shard) is intended to be owned and accessed by exactly one worker thread.
- Under this model:
  - PomaiArena::alloc_seed(), PomaiArena::alloc_blob(), PomaiMap::put/get/erase are expected to be called from the shard owner thread.
  - This minimizes locking and maximizes throughput (no per-operation mutex).
  - The code includes atomic operations for Seed::header to prevent torn reads and to provide a safe publish point for writers (header.store with release, readers use header.load with acquire). This is sufficient for single-writer/multiple-reader patterns if readers are on the same thread or synchronized externally.

If you want multi-writer access to the same shard:
- Add a per-shard mutex (std::mutex or std::shared_mutex).
- Or redesign the allocator / index to be lock-free (non-trivial).
- Important fields that may require attention:
  - Seed::header is atomic; other fields (type/flags/payload) are non-atomic and rely on header acquire/release ordering.
  - PomaiArena's RNG is member-based; either guard it or use thread_local RNG when multi-threaded.

Recommendation:
- Keep single-thread-per-shard for now for simplicity and performance.
- Document in user-facing docs and enforce in startup (e.g., pin threads to cores via ShardManager).
```