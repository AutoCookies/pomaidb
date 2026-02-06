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

Outputs:

- `out/bench_cbrs_<timestamp>.json`
- `out/bench_cbrs_<timestamp>.csv`

## Interpreting Results

Each scenario reports:

- ingest throughput (`ingest_qps`)
- query latency (`p50/p90/p95/p99/p999` in µs)
- query throughput (`query_qps`)
- CPU time (`user_cpu_sec`, `sys_cpu_sec`)
- memory (`rss_open_kb`, `rss_ingest_kb`, `rss_query_kb`, `peak_rss_kb`)
- quality (`recall@1`, `recall@10`, `recall@100`)
- routing behavior (`routed_shards_avg/p95`, `routing_probe_avg/p95`)

Verdict rules:

- **PASS**: `recall@10 >= 0.94` and either p99 improves vs fanout baseline by >=5%, or routed shards avg <= half of fanout shard count.
- **WARN**: `recall@10 >= 0.94` but no clear p99/routing win.
- **FAIL**: recall below target or scenario error.

For epoch drift, compare `epoch_drift_dual_on` vs `epoch_drift_dual_off`; dual-probe is expected to preserve recall under routing epoch transitions.
