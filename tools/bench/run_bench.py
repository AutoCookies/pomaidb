#!/usr/bin/env python3
"""PomaiDB benchmark runner (C++ bench + reporting).

C++ only for now. Python binding hook is stubbed for future integration.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RowKey:
    dataset: str
    dim: int
    n: int
    queries: int
    topk: int
    shards: int


def _float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _int(val: str, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def parse_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            rows.append(row)
    return rows


def build_key(row: Dict[str, Any]) -> RowKey:
    return RowKey(
        dataset=row.get("dataset", ""),
        dim=_int(row.get("dim", "0")),
        n=_int(row.get("n", "0")),
        queries=_int(row.get("queries", "0")),
        topk=_int(row.get("topk", "0")),
        shards=_int(row.get("shards", "0")),
    )


def group_rows(rows: Iterable[Dict[str, Any]]) -> Dict[RowKey, List[Dict[str, Any]]]:
    grouped: Dict[RowKey, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(build_key(row), []).append(row)
    return grouped


def run_cxx_bench(args: argparse.Namespace, out_dir: Path) -> Tuple[Path, Path]:
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_csv = out_dir / f"bench_cbrs_{args.matrix}_{stamp}.csv"
    report_json = out_dir / f"bench_cbrs_{args.matrix}_{stamp}.json"
    db_path = out_dir / f"db_{args.matrix}_{stamp}"
    cmd = [
        str(args.bench_bin),
        "--matrix",
        args.matrix,
        "--seed",
        str(args.seed),
        "--path",
        str(db_path),
        "--report_csv",
        str(report_csv),
        "--report_json",
        str(report_json),
    ]
    if args.extra_args:
        cmd.extend(args.extra_args)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return report_csv, report_json


def build_summary(grouped: Dict[RowKey, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        fanout = next((r for r in rows if r.get("routing") == "fanout"), None)
        cbrs = next((r for r in rows if r.get("routing") == "cbrs"), None)
        nodual = next((r for r in rows if r.get("routing") == "cbrs_no_dual"), None)
        if not fanout or not cbrs or not nodual:
            continue
        p99_gain = (_float(fanout["p99_us"]) - _float(cbrs["p99_us"])) / max(
            1e-9, _float(fanout["p99_us"])
        )
        qps_gain = (_float(cbrs["query_qps"]) - _float(fanout["query_qps"])) / max(
            1e-9, _float(fanout["query_qps"])
        )
        summary.append(
            {
                "key": key,
                "fanout": fanout,
                "cbrs": cbrs,
                "nodual": nodual,
                "p99_gain": p99_gain,
                "qps_gain": qps_gain,
            }
        )
    return summary


def build_verdict(rows: Iterable[Dict[str, Any]]) -> str:
    verdicts = [row.get("verdict", "").upper() for row in rows]
    if any(v == "FAIL" for v in verdicts):
        return "FAIL"
    if any(v == "WARN" for v in verdicts):
        return "WARN"
    return "PASS"


def compare_baseline(current: List[Dict[str, Any]], baseline: List[Dict[str, Any]]) -> List[str]:
    base_index: Dict[Tuple[str, str, int, int, int, int, int], Dict[str, Any]] = {}
    for row in baseline:
        key = (
            row.get("scenario", ""),
            row.get("routing", ""),
            _int(row.get("dim", "0")),
            _int(row.get("n", "0")),
            _int(row.get("queries", "0")),
            _int(row.get("topk", "0")),
            _int(row.get("shards", "0")),
        )
        base_index[key] = row

    regressions: List[str] = []
    for row in current:
        key = (
            row.get("scenario", ""),
            row.get("routing", ""),
            _int(row.get("dim", "0")),
            _int(row.get("n", "0")),
            _int(row.get("queries", "0")),
            _int(row.get("topk", "0")),
            _int(row.get("shards", "0")),
        )
        base = base_index.get(key)
        if not base:
            continue
        p99_now = _float(row.get("p99_us", "0"))
        p99_base = _float(base.get("p99_us", "0"))
        r10_now = _float(row.get("recall10", "0"))
        r10_base = _float(base.get("recall10", "0"))
        if p99_base > 0 and p99_now > p99_base * 1.10:
            regressions.append(
                f"{row.get('scenario')}[{row.get('routing')}] p99 {p99_now:.1f}us > {p99_base:.1f}us"
            )
        if r10_now + 0.01 < r10_base:
            regressions.append(
                f"{row.get('scenario')}[{row.get('routing')}] recall10 {r10_now:.3f} < {r10_base:.3f}"
            )
    return regressions


def write_markdown(report_path: Path, rows: List[Dict[str, Any]], summary: List[Dict[str, Any]], verdict: str, regressions: List[str]) -> None:
    lines: List[str] = []
    lines.append("# PomaiDB CBR-S Benchmark Report")
    lines.append("")
    lines.append(f"Overall verdict: **{verdict}**")
    lines.append("")
    if regressions:
        lines.append("## Regressions vs baseline")
        for item in regressions:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Scenario Summary")
    lines.append("| Dataset | dim | n | q | topk | shards | p99_gain | qps_gain | fanout_p99 | cbrs_p99 | fanout_qps | cbrs_qps | recall10_cbrs | routed_shards_cbrs | routed_buckets_cbrs |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in summary:
        key: RowKey = item["key"]
        fanout = item["fanout"]
        cbrs = item["cbrs"]
        lines.append(
            "| {dataset} | {dim} | {n} | {queries} | {topk} | {shards} | {p99_gain:.2%} | {qps_gain:.2%} | {fanout_p99:.1f} | {cbrs_p99:.1f} | {fanout_qps:.1f} | {cbrs_qps:.1f} | {recall10:.3f} | {routed_shards:.2f} | {routed_buckets:.1f} |".format(
                dataset=key.dataset,
                dim=key.dim,
                n=key.n,
                queries=key.queries,
                topk=key.topk,
                shards=key.shards,
                p99_gain=item["p99_gain"],
                qps_gain=item["qps_gain"],
                fanout_p99=_float(fanout.get("p99_us", "0")),
                cbrs_p99=_float(cbrs.get("p99_us", "0")),
                fanout_qps=_float(fanout.get("query_qps", "0")),
                cbrs_qps=_float(cbrs.get("query_qps", "0")),
                recall10=_float(cbrs.get("recall10", "0")),
                routed_shards=_float(cbrs.get("routed_shards_avg", "0")),
                routed_buckets=_float(cbrs.get("routed_buckets_avg", "0")),
            )
        )
    lines.append("")
    lines.append("## Raw Rows")
    lines.append("```json")
    lines.append(json.dumps(rows, indent=2))
    lines.append("```")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="PomaiDB benchmark runner")
    parser.add_argument("--bench-bin", type=Path, default=Path("build-bench/bin/bench_cbrs"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=Path, default=Path("out/bench_runs"))
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--backend", choices=["cxx", "python"], default="cxx")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Extra args passed to bench_cbrs")
    args = parser.parse_args()

    if args.quick and args.full:
        parser.error("Choose only one of --quick or --full")
    args.matrix = "full" if args.full else "quick"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "python":
        print("Python backend not wired yet. TODO: integrate pomai Python bindings.")
        return 2

    report_csv, report_json = run_cxx_bench(args, out_dir)
    rows = parse_csv(report_csv)
    grouped = group_rows(rows)
    summary = build_summary(grouped)
    verdict = build_verdict(rows)

    regressions: List[str] = []
    if args.baseline_csv:
        regressions = compare_baseline(rows, parse_csv(args.baseline_csv))

    report_path = out_dir / f"report_{args.matrix}_{report_csv.stem}.md"
    write_markdown(report_path, rows, summary, verdict, regressions)

    print(f"CSV: {report_csv}")
    print(f"JSON: {report_json}")
    print(f"Report: {report_path}")
    if regressions:
        print("Regressions detected:")
        for r in regressions:
            print(f" - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
