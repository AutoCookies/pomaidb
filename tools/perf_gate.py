#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_json(p: Path):
    with p.open() as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-dir', default='build')
    ap.add_argument('--baseline', default='benchmarks/perf_baseline.json')
    ap.add_argument('--threshold', type=float, default=0.10,
                    help='allowed regression ratio (default 0.10 == 10%%)')
    ap.add_argument('--write-baseline', action='store_true')
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    bench = build_dir / 'ci_perf_bench'
    if not bench.exists():
        print(f'missing benchmark executable: {bench}', file=sys.stderr)
        return 2

    out = build_dir / 'ci_perf_results.json'
    subprocess.run([str(bench), '--output', str(out)], check=True)

    current = load_json(out)
    if args.write_baseline:
        Path(args.baseline).write_text(json.dumps(current, indent=2) + '\n')
        print(f'wrote new baseline to {args.baseline}')
        return 0

    baseline = load_json(Path(args.baseline))

    b_ingest = baseline['metrics']['ingest_qps']
    c_ingest = current['metrics']['ingest_qps']
    b_p95 = baseline['metrics']['search_latency_us']['p95']
    c_p95 = current['metrics']['search_latency_us']['p95']

    min_ingest = b_ingest * (1.0 - args.threshold)
    max_p95 = b_p95 * (1.0 + args.threshold)

    failed = False
    print(f'ingest_qps baseline={b_ingest:.2f} current={c_ingest:.2f} min_allowed={min_ingest:.2f}')
    if c_ingest < min_ingest:
        print('FAIL: ingest throughput regression beyond threshold')
        failed = True

    print(f'p95_us baseline={b_p95:.2f} current={c_p95:.2f} max_allowed={max_p95:.2f}')
    if c_p95 > max_p95:
        print('FAIL: p95 latency regression beyond threshold')
        failed = True

    if failed:
        return 1

    print('PASS: perf guardrail satisfied')
    return 0


if __name__ == '__main__':
    sys.exit(main())
