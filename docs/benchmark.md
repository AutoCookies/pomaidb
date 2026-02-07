# CBR-S Benchmark Suite (`bench_cbrs`)

## Build

```bash
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build-bench -j
```

Binary path:

```bash
build-bench/bin/bench_cbrs
```

## Run

Single scenario:

```bash
build-bench/bin/bench_cbrs \
  --path /tmp/pomai_bench/single \
  --seed 1337 \
  --shards 4 --dim 256 --n 100000 --queries 2000 --topk 10 \
  --dataset clustered --clusters 8 \
  --routing cbrs --probe 2 --k_global 8 \
  --fsync never --threads 1
```

Full matrix (recommended):

```bash
build-bench/bin/bench_cbrs --matrix full --seed 1337 --path /tmp/pomai_bench/full
```

Quick matrix (faster, smaller):

```bash
build-bench/bin/bench_cbrs --matrix quick --seed 1337 --path /tmp/pomai_bench/quick
```

Python runner (recommended):

```bash
python3 tools/bench/run_bench.py --quick
python3 tools/bench/run_bench.py --full --baseline-csv out/bench_runs/bench_cbrs_full_<prev>.csv
```

The runner emits a Markdown report under `out/bench_runs/` with a summary table and PASS/WARN/FAIL verdicts.

Outputs:

- `out/bench_cbrs_<timestamp>.json`
- `out/bench_cbrs_<timestamp>.csv`

## Interpreting Results

Each scenario reports:

- ingest throughput (`ingest_qps`)
- query latency (`p50/p90/p95/p99/p999/p9999` in µs; p9999 is present when queries >= 10k)
- query throughput (`query_qps`)
- warmup timing (`warmup_sec`, `warmup_qps`)
- CPU time (`user_cpu_sec`, `sys_cpu_sec`)
- memory (`rss_open_kb`, `rss_ingest_kb`, `rss_query_kb`, `peak_rss_kb`)
- quality (`recall@1`, `recall@10`, `recall@100`)
- routing behavior (`routed_shards_avg/p95`, `routing_probe_avg/p95`, `routed_buckets_avg/p95`). `routed_buckets` is the per-query candidate scan count when bucket-level routing is unavailable.

Dataset modes:

- `uniform`: random unit vectors
- `clustered`: well-separated clusters
- `overlap`: overlapping clusters
- `overlap_hard`: clusters with very small centroid margins
- `skew`: heavy centroid skew
- `skew_hard`: 99% hot centroid skew
- `epoch_drift_hard`: routing-epoch drift with owner remap

Verdict rules:

- **PASS**: `recall@10 >= 0.94` and either p99 improves vs fanout baseline by >=5%, or routed shards avg <= half of fanout shard count.
- **WARN**: `recall@10 >= 0.94` but no clear p99/routing win.
- **FAIL**: recall below target or scenario error.

For epoch drift, compare `epoch_drift_hard_*` with `routing=cbrs` vs `routing=cbrs_no_dual`; dual-probe is expected to preserve recall under routing epoch transitions and the delta is printed in the bench output. The `cbrs_no_dual` scenario intentionally suppresses prior-epoch hits when scoring base-epoch queries to surface the expected recall gap.

## Example Output

```
=== Scenario: overlap_hard_cbrs ===
ingest_qps=58231.4 query_qps=8932.1 p99=412.3us recall@10=0.9723 routed_shards_avg=1.25
epoch_drift_hard recall delta (dual_on - dual_off)=0.0634
```
