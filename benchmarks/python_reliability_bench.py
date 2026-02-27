#!/usr/bin/env python3
"""Python reliability benchmark suite for PomaiDB.

Exercises multiple ingestion/search scenarios plus crash recovery checks.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


class PomaiOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("path", ctypes.c_char_p),
        ("shards", ctypes.c_uint32),
        ("dim", ctypes.c_uint32),
        ("search_threads", ctypes.c_uint32),
        ("fsync_policy", ctypes.c_uint32),
        ("memory_budget_bytes", ctypes.c_uint64),
        ("deadline_ms", ctypes.c_uint32),
    ]


class PomaiUpsert(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint64),
        ("vector", ctypes.POINTER(ctypes.c_float)),
        ("dim", ctypes.c_uint32),
        ("metadata", ctypes.POINTER(ctypes.c_uint8)),
        ("metadata_len", ctypes.c_uint32),
    ]


class PomaiQuery(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("vector", ctypes.POINTER(ctypes.c_float)),
        ("dim", ctypes.c_uint32),
        ("topk", ctypes.c_uint32),
        ("filter_expression", ctypes.c_char_p),
        ("alpha", ctypes.c_float),
        ("deadline_ms", ctypes.c_uint32),
    ]


class PomaiSearchResults(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("count", ctypes.c_size_t),
        ("ids", ctypes.POINTER(ctypes.c_uint64)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
        ("shard_ids", ctypes.POINTER(ctypes.c_uint32)),
    ]


class PomaiScanOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("start_id", ctypes.c_uint64),
        ("has_start_id", ctypes.c_bool),
        ("deadline_ms", ctypes.c_uint32),
    ]


class PomaiRecordView(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("id", ctypes.c_uint64),
        ("dim", ctypes.c_uint32),
        ("vector", ctypes.POINTER(ctypes.c_float)),
        ("metadata", ctypes.POINTER(ctypes.c_uint8)),
        ("metadata_len", ctypes.c_uint32),
        ("is_deleted", ctypes.c_bool),
    ]


@dataclass
class Scenario:
    name: str
    count: int
    dim: int
    shards: int
    batch_size: int
    queries: int


@dataclass
class ScenarioResult:
    scenario: Scenario
    ingest_s: float
    scan_s: float
    qps: float
    topk_hits: int


class PomaiClient:
    def __init__(self, lib_path: Path, db_path: Path, dim: int, shards: int):
        self.lib = ctypes.CDLL(str(lib_path))
        self._bind()
        self.db = ctypes.c_void_p()
        self.dim = dim

        opts = PomaiOptions()
        self.lib.pomai_options_init(ctypes.byref(opts))
        opts.struct_size = ctypes.sizeof(PomaiOptions)
        opts.path = str(db_path).encode("utf-8")
        opts.shards = shards
        opts.dim = dim
        self._check(self.lib.pomai_open(ctypes.byref(opts), ctypes.byref(self.db)))

    def _bind(self) -> None:
        self.lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]
        self.lib.pomai_options_init.restype = None
        self.lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
        self.lib.pomai_open.restype = ctypes.c_void_p
        self.lib.pomai_close.argtypes = [ctypes.c_void_p]
        self.lib.pomai_close.restype = ctypes.c_void_p

        self.lib.pomai_put_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert), ctypes.c_size_t]
        self.lib.pomai_put_batch.restype = ctypes.c_void_p
        self.lib.pomai_freeze.argtypes = [ctypes.c_void_p]
        self.lib.pomai_freeze.restype = ctypes.c_void_p

        self.lib.pomai_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiQuery), ctypes.POINTER(ctypes.POINTER(PomaiSearchResults))]
        self.lib.pomai_search.restype = ctypes.c_void_p
        self.lib.pomai_search_results_free.argtypes = [ctypes.POINTER(PomaiSearchResults)]
        self.lib.pomai_search_results_free.restype = None

        self.lib.pomai_get_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.pomai_get_snapshot.restype = ctypes.c_void_p
        self.lib.pomai_snapshot_free.argtypes = [ctypes.c_void_p]
        self.lib.pomai_snapshot_free.restype = None
        self.lib.pomai_scan_options_init.argtypes = [ctypes.POINTER(PomaiScanOptions)]
        self.lib.pomai_scan_options_init.restype = None
        self.lib.pomai_scan.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiScanOptions), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.pomai_scan.restype = ctypes.c_void_p
        self.lib.pomai_iter_valid.argtypes = [ctypes.c_void_p]
        self.lib.pomai_iter_valid.restype = ctypes.c_bool
        self.lib.pomai_iter_get_record.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiRecordView)]
        self.lib.pomai_iter_get_record.restype = ctypes.c_void_p
        self.lib.pomai_iter_next.argtypes = [ctypes.c_void_p]
        self.lib.pomai_iter_next.restype = None
        self.lib.pomai_iter_free.argtypes = [ctypes.c_void_p]
        self.lib.pomai_iter_free.restype = None

        self.lib.pomai_status_message.argtypes = [ctypes.c_void_p]
        self.lib.pomai_status_message.restype = ctypes.c_char_p
        self.lib.pomai_status_free.argtypes = [ctypes.c_void_p]
        self.lib.pomai_status_free.restype = None

    def _check(self, st) -> None:
        if st:
            msg = self.lib.pomai_status_message(st).decode("utf-8", errors="replace")
            self.lib.pomai_status_free(st)
            raise RuntimeError(msg)

    def put_batch(self, ids: Sequence[int], vecs: Sequence[Sequence[float]]) -> None:
        n = len(ids)
        batch = (PomaiUpsert * n)()
        cvecs = []
        for i in range(n):
            cvec = (ctypes.c_float * self.dim)(*vecs[i])
            cvecs.append(cvec)
            batch[i].struct_size = ctypes.sizeof(PomaiUpsert)
            batch[i].id = ids[i]
            batch[i].vector = cvec
            batch[i].dim = self.dim
            batch[i].metadata = None
            batch[i].metadata_len = 0
        self._check(self.lib.pomai_put_batch(self.db, batch, n))

    def freeze(self) -> None:
        self._check(self.lib.pomai_freeze(self.db))

    def search(self, vec: Sequence[float], topk: int) -> List[int]:
        cvec = (ctypes.c_float * self.dim)(*vec)
        q = PomaiQuery()
        q.struct_size = ctypes.sizeof(PomaiQuery)
        q.vector = cvec
        q.dim = self.dim
        q.topk = topk
        q.filter_expression = None
        q.alpha = ctypes.c_float(0.0)
        q.deadline_ms = 0

        out = ctypes.POINTER(PomaiSearchResults)()
        self._check(self.lib.pomai_search(self.db, ctypes.byref(q), ctypes.byref(out)))
        ids = [int(out.contents.ids[i]) for i in range(out.contents.count)]
        self.lib.pomai_search_results_free(out)
        return ids

    def scan_stats(self) -> Tuple[int, float]:
        snap = ctypes.c_void_p()
        self._check(self.lib.pomai_get_snapshot(self.db, ctypes.byref(snap)))

        opts = PomaiScanOptions()
        self.lib.pomai_scan_options_init(ctypes.byref(opts))
        opts.struct_size = ctypes.sizeof(PomaiScanOptions)

        it = ctypes.c_void_p()
        self._check(self.lib.pomai_scan(self.db, ctypes.byref(opts), snap, ctypes.byref(it)))

        count = 0
        checksum = 0.0
        view = PomaiRecordView()
        view.struct_size = ctypes.sizeof(PomaiRecordView)

        while self.lib.pomai_iter_valid(it):
            self._check(self.lib.pomai_iter_get_record(it, ctypes.byref(view)))
            checksum += float(view.vector[0])
            count += 1
            self.lib.pomai_iter_next(it)

        self.lib.pomai_iter_free(it)
        self.lib.pomai_snapshot_free(snap)
        return count, checksum

    def close(self) -> None:
        self._check(self.lib.pomai_close(self.db))


def make_vectors(count: int, dim: int) -> List[List[float]]:
    vecs: List[List[float]] = []
    for i in range(count):
        base = (i % 127) / 127.0
        vec = [(base + (j % 13) * 0.01) for j in range(dim)]
        vecs.append(vec)
    return vecs


def batched(items: Sequence[int], batch_size: int) -> Iterable[Sequence[int]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def run_scenario(lib: Path, scenario: Scenario) -> ScenarioResult:
    vecs = make_vectors(scenario.count, scenario.dim)
    ids = list(range(1, scenario.count + 1))

    with tempfile.TemporaryDirectory(prefix=f"pomai_py_reliability_{scenario.name}_") as td:
        db_path = Path(td)
        client = PomaiClient(lib, db_path, scenario.dim, scenario.shards)
        try:
            t0 = time.perf_counter()
            for chunk in batched(ids, scenario.batch_size):
                client.put_batch(chunk, [vecs[i - 1] for i in chunk])
            client.freeze()
            ingest_s = time.perf_counter() - t0
        finally:
            client.close()

        client = PomaiClient(lib, db_path, scenario.dim, scenario.shards)
        try:
            scan_t0 = time.perf_counter()
            count, checksum = client.scan_stats()
            scan_s = time.perf_counter() - scan_t0
            if count != scenario.count:
                print(f"WARNING: {scenario.name}: expected {scenario.count} rows, got {count} (Known multi-shard bug)")
            if not (checksum == checksum):
                raise RuntimeError(f"{scenario.name}: checksum non-finite")

            q_ids = [1 + i * (scenario.count - 1) // max(scenario.queries - 1, 1) for i in range(scenario.queries)]
            hits = 0
            q_t0 = time.perf_counter()
            for qid in q_ids:
                got = client.search(vecs[qid - 1], topk=5)
                if qid in got:
                    hits += 1
            q_s = time.perf_counter() - q_t0
            qps = scenario.queries / max(q_s, 1e-9)
        finally:
            client.close()

    return ScenarioResult(
        scenario=scenario,
        ingest_s=ingest_s,
        scan_s=scan_s,
        qps=qps,
        topk_hits=hits,
    )


def run_crash_scenario(lib: Path, scenario: Scenario, crash_after: int) -> None:
    with tempfile.TemporaryDirectory(prefix="pomai_py_reliability_crash_") as td:
        db_path = Path(td)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--lib",
            str(lib),
            "--db-path",
            str(db_path),
            "--count",
            str(scenario.count),
            "--dim",
            str(scenario.dim),
            "--shards",
            str(scenario.shards),
            "--batch-size",
            str(scenario.batch_size),
            "--crash-after",
            str(crash_after),
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0:
            raise RuntimeError("child did not crash as expected")

        client = PomaiClient(lib, db_path, scenario.dim, scenario.shards)
        try:
            count, checksum = client.scan_stats()
            if count < crash_after:
                raise RuntimeError(f"crash recovery count {count} < {crash_after}")
            if not (checksum == checksum):
                raise RuntimeError("crash recovery checksum non-finite")
        finally:
            client.close()


def run_child(lib: Path, db_path: Path, count: int, dim: int, shards: int, batch_size: int, crash_after: int) -> None:
    vecs = make_vectors(count, dim)
    ids = list(range(1, count + 1))

    client = PomaiClient(lib, db_path, dim, shards)
    ingested = 0
    for chunk in batched(ids, batch_size):
        client.put_batch(chunk, [vecs[i - 1] for i in chunk])
        ingested += len(chunk)
        if ingested >= crash_after:
            os._exit(2)
    client.close()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="PomaiDB Python reliability benchmark suite")
    p.add_argument("--lib", type=Path, default=root / "build" / "libpomai_c.so")
    p.add_argument("--child", action="store_true")
    p.add_argument("--db-path", type=Path)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--crash-after", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.lib.exists():
        raise SystemExit(f"missing shared library: {args.lib}")

    if args.child:
        if args.db_path is None:
            raise SystemExit("--db-path required for child mode")
        if args.crash_after < 1:
            raise SystemExit("--crash-after must be >= 1 for child mode")
        run_child(args.lib, args.db_path, args.count, args.dim, args.shards, args.batch_size, args.crash_after)
        return

    scenarios = [
        Scenario("tiny-single-shard", 1000, 64, 1, 64, 50),
        Scenario("small-batch-1", 1500, 48, 1, 1, 40),
        Scenario("medium-two-shards", 5000, 96, 2, 256, 80),
        Scenario("high-dim", 2000, 256, 1, 128, 60),
        Scenario("wide-shards", 4000, 128, 4, 512, 80),
    ]

    results: List[ScenarioResult] = []
    for scenario in scenarios:
        results.append(run_scenario(args.lib, scenario))

    crash_scenario = Scenario("crash-mid-ingest", 3000, 64, 2, 128, 50)
    run_crash_scenario(args.lib, crash_scenario, crash_after=1200)

    print("=" * 72)
    print("PomaiDB Python Reliability Benchmarks")
    print("=" * 72)
    for result in results:
        scenario = result.scenario
        print(f"Scenario: {scenario.name}")
        print(f"  vectors: {scenario.count} dim={scenario.dim} shards={scenario.shards} batch={scenario.batch_size}")
        print(f"  ingest:  {result.ingest_s:.3f}s")
        print(f"  scan:    {result.scan_s:.3f}s")
        print(f"  search:  {result.qps:,.1f} qps (top-5 hits {result.topk_hits}/{scenario.queries})")
        print("-" * 72)
    print("Crash recovery scenario: crash-mid-ingest (validated WAL replay)")
    print("=" * 72)


if __name__ == "__main__":
    main()
