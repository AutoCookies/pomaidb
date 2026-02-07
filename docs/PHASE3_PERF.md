# Phase 3 Perf Gate परिणाम (ingest + search)

## Command

```
python3 tools/perf_gate.py --build-dir build --baseline benchmarks/perf_baseline.json --threshold 0.10
```

## Results

```
ingest_qps baseline=18000.00 current=32809.80 min_allowed=16200.00
p95_us baseline=2300.00 current=431.16 max_allowed=2530.00
p999_us baseline=2600.00 current=726.73 max_allowed=2860.00
PASS: perf guardrail satisfied
```

### Notes

* The perf gate runs `ci_perf_bench`, which exercises a combined ingest + search workload and reports p50/p95/p99/p999 search latency.
* The p999 latency remained below the 10% regression guardrail compared to the baseline.
