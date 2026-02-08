# PomaiDB Benchmark Trust Model

PomaiDB benchmarks exist to earn **trust**, not to market speed. This document is the single source of truth (SOT) for what PomaiDB **guarantees**, what it **does not guarantee**, and which benchmarks validate each guarantee.

## Guarantees vs Benchmarks

| Guarantee | What it means | Benchmark | What is enforced |
| --- | --- | --- | --- |
| Correctness (ANN recall) | Approximate search results match a brute-force oracle with high recall. | Recall Correctness Benchmark (`pomai-bench recall`) | Recall@1/10/100 must be ≥ 0.94 for each dataset case. |
| Tail latency stability | Latency under mixed ingest + search load is visible and does not hide p999. | Mixed Load Tail Benchmark (`pomai-bench mixed-load`) | p50/p95/p99/p999 and QPS are reported. |
| Crash safety | Data is not lost after SIGKILL during ingest. | Crash Recovery Benchmark (`pomai-bench crash-recovery`) | Recovered count is validated; recall after recovery is measured. |
| Low-end viability | PomaiDB remains usable on 2–4 core / 8GB RAM class machines. | Low-End Laptop Benchmark (`pomai-bench low-end`) | Brute-force vs PomaiDB p99 latency + Recall@10 reported. |
| Explainability | Search routing visibility is exposed to operators. | Explain / Search Plan Benchmark (`pomai-bench explain`) | Plan fields are printed for a representative query. |

## What PomaiDB Guarantees

1. **Recall correctness against an oracle** for the tested datasets and dimensions, validated via brute-force dot-product top-k. The recall gate is hard: any case below 0.94 fails the benchmark and exits non-zero.
2. **Tail latency transparency**: p50/p95/p99/p999 are always reported for mixed ingest + search workloads.
3. **Crash recovery safety**: data count is validated after SIGKILL and recall is re-measured to detect corruption or missing vectors.
4. **Low-end viability**: the benchmark is runnable on low-end laptops and reports brute-force vs PomaiDB p99 latency.
5. **Determinism**: all datasets use fixed random seeds and deterministic generators, enabling reproducible runs.

## What PomaiDB Does NOT Guarantee (and how we surface it)

- **No hidden tail latency**: p999 is always shown; if it spikes, it remains visible in the output.
- **No cloud-only assumptions**: benchmarks are designed to run on laptop-class hardware and do not assume GPU or large RAM.
- **No opaque search plans**: the C API currently exposes shard routing via result shard IDs only. The explain benchmark reports the fields available and explicitly marks `Exact rerank: not_exposed` until the C API surfaces more details.
- **No best-case-only runs**: mixed-load includes simultaneous ingest + search; recall uses clustered datasets and multiple sizes/dimensions.

## Benchmark-to-Guarantee Mapping

| Benchmark | Validates |
| --- | --- |
| Recall Correctness | Recall gates for ANN correctness. |
| Mixed Load Tail | Tail latency stability with concurrent ingest and search. |
| Low-End Laptop | Usability on constrained hardware (p99 + recall). |
| Crash Recovery | WAL replay + recall stability after SIGKILL. |
| Explain Plan | Search plan visibility (best-effort with exposed API). |

## Reproducibility Requirements

- **Fixed random seeds** (default seed = 42; each benchmark uses documented seeds).
- **Deterministic generators** (clustered, normalized vectors).
- **Documented commands** for every benchmark (see `scripts/pomai-bench` and `scripts/benchmark_trust.sh`).
- **Machine-readable output** is emitted as JSON alongside human-readable text.

## Known Limitations (Transparent by Design)

- **Search plan details**: The C API does not expose rerank counts or shard-level timings; the explain benchmark explicitly flags this as `not_exposed` rather than inventing data.
- **Metric configurability**: The C API does not currently expose metric selection; benchmarks assume dot-product behavior as implemented in the core search path.

## Canonical Commands

```
./scripts/pomai-bench recall
./scripts/pomai-bench mixed-load
./scripts/pomai-bench low-end --machine "i5-8250U, 4c/8t, 8GB RAM"
./scripts/pomai-bench crash-recovery
./scripts/pomai-bench explain
```

## CI Gates

CI runs `scripts/benchmark_trust.sh`, which:

1. Executes the full recall benchmark and enforces recall gates.
2. Executes a mixed-load smoke test and reports p999.

Failures are treated as regressions and fail the CI job.
