#!/usr/bin/env python3
"""Canonical PomaiDB benchmark suite.

Trust-focused benchmarks:
- Recall correctness vs brute-force oracle.
- Mixed ingest/search tail latency.
- Low-end viability with brute-force comparison.
- Crash recovery verification.
- Explain / search-plan visibility.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None
    HAS_NUMPY = False


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
class RecallCase:
    dim: int
    count: int
    queries: int
    topk: int
    seed: int


@dataclass
class RecallMetrics:
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float


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
            vec = vecs[i]
            if HAS_NUMPY and np is not None and hasattr(vec, "astype"):
                vec = vec.astype(np.float32, copy=False)
            cvec = (ctypes.c_float * self.dim)(*vec)
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

    def search(self, vec: Sequence[float], topk: int) -> Tuple[List[int], List[int]]:
        if HAS_NUMPY and np is not None and hasattr(vec, "astype"):
            vec = vec.astype(np.float32, copy=False)
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
        shard_ids = [int(out.contents.shard_ids[i]) for i in range(out.contents.count)]
        self.lib.pomai_search_results_free(out)
        return ids, shard_ids

    def scan_count(self) -> int:
        snap = ctypes.c_void_p()
        self._check(self.lib.pomai_get_snapshot(self.db, ctypes.byref(snap)))

        opts = PomaiScanOptions()
        self.lib.pomai_scan_options_init(ctypes.byref(opts))
        opts.struct_size = ctypes.sizeof(PomaiScanOptions)

        it = ctypes.c_void_p()
        self._check(self.lib.pomai_scan(self.db, ctypes.byref(opts), snap, ctypes.byref(it)))
        count = 0
        view = PomaiRecordView()
        view.struct_size = ctypes.sizeof(PomaiRecordView)

        while self.lib.pomai_iter_valid(it):
            self._check(self.lib.pomai_iter_get_record(it, ctypes.byref(view)))
            count += 1
            self.lib.pomai_iter_next(it)

        self.lib.pomai_iter_free(it)
        self.lib.pomai_snapshot_free(snap)
        return count

    def close(self) -> None:
        self._check(self.lib.pomai_close(self.db))


def batched_ids(count: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        yield start, end


def _normalize(vec: List[float]) -> List[float]:
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 1e-12:
        return [v / norm for v in vec]
    return vec


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def make_queries_from_base(base: Sequence[Sequence[float]], count: int, seed: int, noise: float = 0.001) -> Sequence[Sequence[float]]:
    if HAS_NUMPY and np is not None:
        rng = np.random.default_rng(seed)
        base_arr = np.asarray(base, dtype=np.float32)
        idxs = rng.integers(0, base_arr.shape[0], size=count)
        queries = base_arr[idxs].copy()
        if noise > 0:
            queries += rng.normal(0.0, noise, size=queries.shape).astype(np.float32)
            norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
            queries = queries / norms
        return queries.astype(np.float32)

    rng = random.Random(seed)
    queries: List[List[float]] = []
    for _ in range(count):
        vec = list(base[rng.randrange(0, len(base))])
        if noise > 0:
            vec = [v + rng.gauss(0.0, noise) for v in vec]
            vec = _normalize(vec)
        queries.append(vec)
    return queries


def make_clustered_vectors(count: int, dim: int, seed: int) -> Sequence[Sequence[float]]:
    clusters = max(8, dim // 32)
    if HAS_NUMPY and np is not None:
        rng = np.random.default_rng(seed)
        centers = rng.normal(0.0, 1.0, size=(clusters, dim)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
        assignments = rng.integers(0, clusters, size=count)
        noise = rng.normal(0.0, 0.05, size=(count, dim)).astype(np.float32)
        vectors = centers[assignments] + noise
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors.astype(np.float32)

    rng = random.Random(seed)
    centers: List[List[float]] = []
    for _ in range(clusters):
        center = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        centers.append(_normalize(center))
    vectors: List[List[float]] = []
    for _ in range(count):
        cid = rng.randrange(0, clusters)
        base = centers[cid]
        vec = [base[i] + rng.gauss(0.0, 0.05) for i in range(dim)]
        vectors.append(_normalize(vec))
    return vectors


def brute_force_topk(base: Sequence[Sequence[float]], queries: Sequence[Sequence[float]], k: int, block: int = 50000) -> List[List[int]]:
    if HAS_NUMPY and np is not None:
        base_arr = np.asarray(base, dtype=np.float32)
        query_arr = np.asarray(queries, dtype=np.float32)
        q = query_arr.shape[0]
        best_scores = None
        best_ids = None

        for start in range(0, base_arr.shape[0], block):
            end = min(base_arr.shape[0], start + block)
            block_vecs = base_arr[start:end]
            scores = query_arr @ block_vecs.T
            block_ids = np.arange(start, end, dtype=np.int64)
            if best_scores is None:
                best_scores = np.empty((q, k), dtype=np.float32)
                best_ids = np.empty((q, k), dtype=np.int64)
                for qi in range(q):
                    idx = np.argpartition(scores[qi], -k)[-k:]
                    best_scores[qi] = scores[qi][idx]
                    best_ids[qi] = block_ids[idx]
            else:
                for qi in range(q):
                    combined_scores = np.concatenate([best_scores[qi], scores[qi]])
                    combined_ids = np.concatenate([best_ids[qi], block_ids])
                    idx = np.argpartition(combined_scores, -k)[-k:]
                    best_scores[qi] = combined_scores[idx]
                    best_ids[qi] = combined_ids[idx]

        order = np.argsort(-best_scores, axis=1)
        sorted_ids = np.take_along_axis(best_ids, order, axis=1)
        return sorted_ids.tolist()

    results: List[List[int]] = []
    for qvec in queries:
        scores = []
        for idx, vec in enumerate(base):
            score = sum(a * b for a, b in zip(qvec, vec))
            scores.append((score, idx))
        scores.sort(key=lambda pair: pair[0], reverse=True)
        results.append([idx for _, idx in scores[:k]])
    return results


def recall_at_k(oracle: Sequence[Sequence[int]], approx: List[List[int]], k: int) -> float:
    total = len(approx)
    if total == 0:
        return 0.0
    acc = 0.0
    for qi in range(total):
        oracle_ids = oracle[qi][:k]
        if HAS_NUMPY and np is not None:
            approx_ids = np.array(approx[qi], dtype=np.int64)[:k]
            approx_list = approx_ids.tolist()
        else:
            approx_list = list(approx[qi])[:k]
        if not approx_list:
            continue
        acc += len(set(approx_list) & set(oracle_ids)) / float(k)
    return acc / total


def recall_metrics(oracle: Sequence[Sequence[int]], approx: List[List[int]]) -> RecallMetrics:
    r1 = recall_at_k(oracle, approx, 1)
    r10 = recall_at_k(oracle, approx, 10)
    r100 = recall_at_k(oracle, approx, 100)
    return RecallMetrics(r1, r10, r100)


def ensure_recall_gates(metrics: RecallMetrics) -> None:
    if metrics.recall_at_1 < 0.94 or metrics.recall_at_10 < 0.94 or metrics.recall_at_100 < 0.94:
        raise SystemExit(
            "Recall gate failed: "
            f"R@1={metrics.recall_at_1:.3f} "
            f"R@10={metrics.recall_at_10:.3f} "
            f"R@100={metrics.recall_at_100:.3f}"
        )


def run_recall_case(lib: Path, case: RecallCase, shards: int, batch_size: int) -> Tuple[RecallMetrics, float]:
    vectors = make_clustered_vectors(case.count, case.dim, case.seed)
    queries = make_queries_from_base(vectors, case.queries, case.seed + 1)
    oracle_ids = brute_force_topk(vectors, queries, 100)

    with tempfile.TemporaryDirectory(prefix="pomai_bench_recall_") as td:
        client = PomaiClient(lib, Path(td), case.dim, shards)
        try:
            t0 = time.perf_counter()
            for start, end in batched_ids(case.count, batch_size):
                ids = list(range(start + 1, end + 1))
                batch_vecs = [vectors[i] for i in range(start, end)]
                client.put_batch(ids, batch_vecs)
            client.freeze()
            ingest_s = time.perf_counter() - t0

            approx_ids: List[List[int]] = []
            for qvec in queries:
                got, _ = client.search(qvec, topk=100)
                approx_ids.append([gid - 1 for gid in got])
        finally:
            client.close()

    metrics = recall_metrics(oracle_ids, approx_ids)
    return metrics, ingest_s


def print_recall_report(case: RecallCase, metrics: RecallMetrics) -> None:
    name = f"clustered_{case.dim}d_seed{case.seed}"
    print("Recall Benchmark")
    print("----------------")
    print(f"Dataset: {name}")
    print(f"N={case.count}, Q={case.queries}, topk={case.topk}")
    print()
    print(f"Recall@1   = {metrics.recall_at_1:.3f}")
    print(f"Recall@10  = {metrics.recall_at_10:.3f}")
    print(f"Recall@100 = {metrics.recall_at_100:.3f}")
    print()


def percentile_ms(latencies: Sequence[float], p: float) -> float:
    if not latencies:
        return 0.0
    if HAS_NUMPY and np is not None:
        arr = np.array(latencies)
        return float(np.percentile(arr, p))
    sorted_vals = sorted(latencies)
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return float(sorted_vals[min(max(idx, 0), len(sorted_vals) - 1)])


def run_mixed_load(lib: Path, dim: int, count: int, shards: int, batch_size: int, queries: int) -> Tuple[dict, dict]:
    vectors = make_clustered_vectors(count, dim, 42)
    query_vecs = make_clustered_vectors(queries, dim, 43)
    latencies: List[float] = []
    search_duration = 0.0

    with tempfile.TemporaryDirectory(prefix="pomai_bench_mixed_") as td:
        client = PomaiClient(lib, Path(td), dim, shards)
        ingested = 0
        lock = threading.Lock()
        stop = threading.Event()

        def ingest_worker() -> None:
            nonlocal ingested
            for start, end in batched_ids(count, batch_size):
                ids = list(range(start + 1, end + 1))
                batch_vecs = [vectors[i] for i in range(start, end)]
                client.put_batch(ids, batch_vecs)
                with lock:
                    ingested = end
                if end % (batch_size * 4) == 0:
                    client.freeze()
            stop.set()

        def search_worker() -> None:
            nonlocal search_duration
            idx = 0
            start = time.perf_counter()
            while not stop.is_set() or idx < queries:
                with lock:
                    available = max(1, ingested)
                qvec = query_vecs[idx % queries]
                if idx >= available:
                    time.sleep(0.001)
                    continue
                t0 = time.perf_counter()
                client.search(qvec, topk=10)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                idx += 1
                if idx >= queries:
                    break
            search_duration = time.perf_counter() - start

        t_ingest = threading.Thread(target=ingest_worker, daemon=True)
        t_search = threading.Thread(target=search_worker, daemon=True)
        t_ingest.start()
        t_search.start()
        t_ingest.join()
        t_search.join()
        oracle_vectors = vectors
        oracle_queries = query_vecs
        if not HAS_NUMPY:
            oracle_vectors = vectors[:min(len(vectors), 20000)]
            oracle_queries = query_vecs[:min(len(query_vecs), 200)]
        oracle_ids = brute_force_topk(oracle_vectors, oracle_queries, 10)
        approx_ids: List[List[int]] = []
        for qvec in oracle_queries:
            got, _ = client.search(qvec, topk=10)
            approx_ids.append([gid - 1 for gid in got])
        recall_10 = recall_at_k(oracle_ids, approx_ids, 10)
        client.close()

    stats = {
        "p50_ms": percentile_ms(latencies, 50),
        "p95_ms": percentile_ms(latencies, 95),
        "p99_ms": percentile_ms(latencies, 99),
        "p999_ms": percentile_ms(latencies, 99.9),
        "qps": len(latencies) / max(search_duration, 1e-9),
        "recall_at_10": recall_10,
    }
    config = {"dim": dim, "N": count, "shards": shards}
    return config, stats


def run_low_end(lib: Path, dim: int, count: int, queries: int, shards: int, machine: str) -> Tuple[dict, dict]:
    vectors = make_clustered_vectors(count, dim, 7)
    query_vecs = make_clustered_vectors(queries, dim, 8)

    oracle_ids = brute_force_topk(vectors, query_vecs, 100)

    brute_lat = []
    for qi in range(queries):
        t0 = time.perf_counter()
        qvec = query_vecs[qi]
        if HAS_NUMPY and np is not None:
            scores = np.asarray(vectors) @ np.asarray(qvec)
            _ = np.argpartition(scores, -10)[-10:]
        else:
            scores = [dot(vec, qvec) for vec in vectors]
            _ = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        brute_lat.append((time.perf_counter() - t0) * 1000.0)

    with tempfile.TemporaryDirectory(prefix="pomai_bench_lowend_") as td:
        client = PomaiClient(lib, Path(td), dim, shards)
        for start, end in batched_ids(count, 256):
            ids = list(range(start + 1, end + 1))
            batch_vecs = [vectors[i] for i in range(start, end)]
            client.put_batch(ids, batch_vecs)
        client.freeze()

        ann_lat = []
        approx_ids: List[List[int]] = []
        for qvec in query_vecs:
            t0 = time.perf_counter()
            got, _ = client.search(qvec, topk=10)
            ann_lat.append((time.perf_counter() - t0) * 1000.0)
            approx_ids.append([gid - 1 for gid in got])
        client.close()

    recall_10 = recall_at_k(oracle_ids, approx_ids, 10)
    stats = {
        "machine": machine,
        "bruteforce_p99_ms": percentile_ms(brute_lat, 99),
        "pomai_p99_ms": percentile_ms(ann_lat, 99),
        "recall_at_10": recall_10,
    }
    config = {"dim": dim, "N": count, "shards": shards}
    return config, stats


def run_crash_recovery(lib: Path, dim: int, count: int, shards: int, crash_after: int) -> Tuple[int, RecallMetrics]:
    vectors = make_clustered_vectors(count, dim, 99)
    query_vecs = make_clustered_vectors(200, dim, 100)
    oracle_ids = brute_force_topk(vectors, query_vecs, 100)

    with tempfile.TemporaryDirectory(prefix="pomai_bench_crash_") as td:
        db_path = Path(td)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "crash-child",
            "--lib",
            str(lib),
            "--db-path",
            str(db_path),
            "--count",
            str(count),
            "--dim",
            str(dim),
            "--shards",
            str(shards),
            "--crash-after",
            str(crash_after),
        ]
        proc = subprocess.Popen(cmd)
        proc.wait()
        if proc.returncode == 0:
            raise SystemExit("Crash recovery child did not crash as expected")

        client = PomaiClient(lib, db_path, dim, shards)
        try:
            recovered = client.scan_count()
            approx_ids: List[List[int]] = []
            for qvec in query_vecs:
                got, _ = client.search(qvec, topk=100)
                approx_ids.append([gid - 1 for gid in got])
        finally:
            client.close()

    metrics = recall_metrics(oracle_ids, approx_ids)
    if recovered != count:
        raise SystemExit(f"Crash recovery failed: recovered {recovered} / {count}")
    return recovered, metrics


def run_crash_child(args: argparse.Namespace) -> None:
    vectors = make_clustered_vectors(args.count, args.dim, 99)
    client = PomaiClient(args.lib, args.db_path, args.dim, args.shards)
    ingested = 0
    try:
        for start, end in batched_ids(args.count, 256):
            ids = list(range(start + 1, end + 1))
            batch_vecs = [vectors[i] for i in range(start, end)]
            client.put_batch(ids, batch_vecs)
            ingested = end
            if ingested >= args.crash_after:
                os.kill(os.getpid(), signal.SIGKILL)
    finally:
        client.close()


def run_explain(lib: Path, dim: int, count: int, shards: int) -> dict:
    vectors = make_clustered_vectors(count, dim, 11)
    query = make_clustered_vectors(1, dim, 12)[0]

    with tempfile.TemporaryDirectory(prefix="pomai_bench_explain_") as td:
        client = PomaiClient(lib, Path(td), dim, shards)
        for start, end in batched_ids(count, 256):
            ids = list(range(start + 1, end + 1))
            batch_vecs = [vectors[i] for i in range(start, end)]
            client.put_batch(ids, batch_vecs)
        client.freeze()
        ids, shard_ids = client.search(query, topk=10)
        client.close()

    shards_visited = len(set(shard_ids))
    return {
        "mode": "VectorSearch",
        "shards_visited": shards_visited,
        "candidates_gathered": len(ids),
        "exact_rerank": "not_exposed",
    }


def emit_machine_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="PomaiDB trust benchmarks")
    parser.add_argument("--lib", type=Path, default=root / "build" / "libpomai_c.so")

    sub = parser.add_subparsers(dest="cmd", required=True)

    recall = sub.add_parser("recall", help="Recall correctness benchmark")
    recall.add_argument("--shards", type=int, default=4)
    recall.add_argument("--batch-size", type=int, default=1024)
    recall.add_argument("--seed", type=int, default=42)
    recall.add_argument("--matrix", choices=["full", "ci"], default="full")

    mixed = sub.add_parser("mixed-load", help="Mixed ingest/search tail latency")
    mixed.add_argument("--dim", type=int, default=512)
    mixed.add_argument("--count", type=int, default=200000)
    mixed.add_argument("--shards", type=int, default=4)
    mixed.add_argument("--batch-size", type=int, default=1024)
    mixed.add_argument("--queries", type=int, default=2000)

    low_end = sub.add_parser("low-end", help="Low-end hardware benchmark")
    low_end.add_argument("--dim", type=int, default=256)
    low_end.add_argument("--count", type=int, default=50000)
    low_end.add_argument("--queries", type=int, default=1000)
    low_end.add_argument("--shards", type=int, default=2)
    low_end.add_argument("--machine", type=str, default="")

    crash = sub.add_parser("crash-recovery", help="Crash recovery benchmark")
    crash.add_argument("--dim", type=int, default=256)
    crash.add_argument("--count", type=int, default=200000)
    crash.add_argument("--shards", type=int, default=4)
    crash.add_argument("--crash-after", type=int, default=150000)

    explain = sub.add_parser("explain", help="Explain/search plan benchmark")
    explain.add_argument("--dim", type=int, default=256)
    explain.add_argument("--count", type=int, default=50000)
    explain.add_argument("--shards", type=int, default=4)

    child = sub.add_parser("crash-child", help=argparse.SUPPRESS)
    child.add_argument("--lib", type=Path, required=True)
    child.add_argument("--db-path", type=Path, required=True)
    child.add_argument("--count", type=int, required=True)
    child.add_argument("--dim", type=int, required=True)
    child.add_argument("--shards", type=int, required=True)
    child.add_argument("--crash-after", type=int, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.lib.exists():
        raise SystemExit(f"missing shared library: {args.lib}")

    if args.cmd == "recall":
        if args.matrix == "ci":
            cases = [
                RecallCase(128, 5000, 200, 10, args.seed),
                RecallCase(256, 5000, 200, 10, args.seed),
                RecallCase(512, 5000, 200, 10, args.seed),
            ]
        else:
            cases = [
                RecallCase(128, 50000, 1000, 10, args.seed),
                RecallCase(128, 100000, 1000, 10, args.seed),
                RecallCase(128, 200000, 1000, 10, args.seed),
                RecallCase(256, 50000, 1000, 10, args.seed),
                RecallCase(256, 100000, 1000, 10, args.seed),
                RecallCase(256, 200000, 1000, 10, args.seed),
                RecallCase(512, 50000, 1000, 10, args.seed),
                RecallCase(512, 100000, 1000, 10, args.seed),
                RecallCase(512, 200000, 1000, 10, args.seed),
            ]
        for case in cases:
            metrics, ingest_s = run_recall_case(args.lib, case, args.shards, args.batch_size)
            print_recall_report(case, metrics)
            ensure_recall_gates(metrics)
            emit_machine_json(
                {
                    "benchmark": "recall",
                    "dataset": f"clustered_{case.dim}d_seed{case.seed}",
                    "N": case.count,
                    "Q": case.queries,
                    "ingest_s": ingest_s,
                    "recall_at_1": metrics.recall_at_1,
                    "recall_at_10": metrics.recall_at_10,
                    "recall_at_100": metrics.recall_at_100,
                }
            )
        return

    if args.cmd == "mixed-load":
        config, stats = run_mixed_load(args.lib, args.dim, args.count, args.shards, args.batch_size, args.queries)
        print("Mixed Load Benchmark")
        print("-------------------")
        print(f"dim={config['dim']}, N={config['N']}, shards={config['shards']}")
        print()
        print(f"p50  = {stats['p50_ms']:.3f} ms")
        print(f"p95  = {stats['p95_ms']:.3f} ms")
        print(f"p99  = {stats['p99_ms']:.3f} ms")
        print(f"p999 = {stats['p999_ms']:.3f} ms")
        print(f"QPS  = {stats['qps']:.1f}")
        print(f"Recall@10 (post-load) = {stats['recall_at_10']:.3f}")
        emit_machine_json({"benchmark": "mixed-load", **config, **stats})
        return

    if args.cmd == "low-end":
        machine = args.machine or f"{platform.processor()} ({platform.system()} {platform.release()})"
        config, stats = run_low_end(args.lib, args.dim, args.count, args.queries, args.shards, machine)
        print("Low-End Laptop Benchmark")
        print("------------------------")
        print(f"Machine: {stats['machine']}")
        print()
        print("Bruteforce:")
        print(f"  p99 = {stats['bruteforce_p99_ms']:.1f} ms")
        print()
        print("PomaiDB:")
        print(f"  p99 = {stats['pomai_p99_ms']:.1f} ms")
        print(f"  Recall@10 = {stats['recall_at_10']:.3f}")
        emit_machine_json({"benchmark": "low-end", **config, **stats})
        return

    if args.cmd == "crash-recovery":
        recovered, metrics = run_crash_recovery(args.lib, args.dim, args.count, args.shards, args.crash_after)
        print("Crash Recovery Benchmark")
        print("------------------------")
        print(f"Crash at {args.crash_after / args.count:.0%} ingest")
        print()
        print(f"Recovered vectors: {recovered} / {args.count}")
        print(f"Recall@10 after recovery: {metrics.recall_at_10:.3f}")
        ensure_recall_gates(metrics)
        emit_machine_json(
            {
                "benchmark": "crash-recovery",
                "recovered": recovered,
                "total": args.count,
                "recall_at_10": metrics.recall_at_10,
            }
        )
        return

    if args.cmd == "explain":
        explain = run_explain(args.lib, args.dim, args.count, args.shards)
        print("SEARCH PLAN")
        print("-----------")
        print(f"Mode: {explain['mode']}")
        print(f"Shards visited: {explain['shards_visited']} / {args.shards}")
        print(f"Candidates gathered: {explain['candidates_gathered']}")
        print(f"Exact rerank: {explain['exact_rerank']}")
        emit_machine_json({"benchmark": "explain", **explain})
        return

    if args.cmd == "crash-child":
        run_crash_child(args)
        return

    raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
