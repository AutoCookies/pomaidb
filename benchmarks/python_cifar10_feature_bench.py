#!/usr/bin/env python3
"""End-to-end Python benchmark: CIFAR-10 feature extraction + PomaiDB workflows.

Designed to run in constrained/offline environments:
- Uses pure-Python CIFAR-10 loader (downloads dataset only when requested).
- Falls back to deterministic CIFAR-like synthetic samples if dataset is unavailable.
- Uses ctypes against libpomai_c.so for ingestion, search, and iterator scan.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import pickle
import random
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_DIRNAME = "cifar-10-batches-py"


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
class BenchResult:
    backend: str
    requested: int
    visible: int
    feat_time_s: float
    feat_rate: float
    ingest_time_s: float
    ingest_rate: float
    search_p50_ms: float
    search_p95_ms: float
    search_p99_ms: float
    search_qps: float
    knn_top1: float
    scan_time_s: float
    scan_rate: float


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
            cvec = (ctypes.c_float * self.dim)(*vec)
            cvecs.append(cvec)
            batch[i].struct_size = ctypes.sizeof(PomaiUpsert)
            batch[i].id = ids[i]
            batch[i].vector = cvec
            batch[i].dim = self.dim
            batch[i].metadata = None
            batch[i].metadata_len = 0
        self._check(self.lib.pomai_put_batch(self.db, batch, n))

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

    def full_scan_mean(self) -> Tuple[int, List[float]]:
        snap = ctypes.c_void_p()
        self._check(self.lib.pomai_get_snapshot(self.db, ctypes.byref(snap)))

        opts = PomaiScanOptions()
        self.lib.pomai_scan_options_init(ctypes.byref(opts))
        opts.struct_size = ctypes.sizeof(PomaiScanOptions)

        it = ctypes.c_void_p()
        self._check(self.lib.pomai_scan(self.db, ctypes.byref(opts), snap, ctypes.byref(it)))

        count = 0
        acc = [0.0] * self.dim
        view = PomaiRecordView()
        view.struct_size = ctypes.sizeof(PomaiRecordView)

        while self.lib.pomai_iter_valid(it):
            self._check(self.lib.pomai_iter_get_record(it, ctypes.byref(view)))
            for j in range(self.dim):
                acc[j] += float(view.vector[j])
            count += 1
            self.lib.pomai_iter_next(it)

        self.lib.pomai_iter_free(it)
        self.lib.pomai_snapshot_free(snap)

        if count > 0:
            acc = [v / count for v in acc]
        return count, acc

    def close(self) -> None:
        self._check(self.lib.pomai_close(self.db))


def maybe_download_cifar(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target_dir = root / CIFAR_DIRNAME
    if target_dir.exists():
        return

    archive = root / "cifar-10-python.tar.gz"
    urllib.request.urlretrieve(CIFAR_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=root)


def load_cifar_samples(root: Path, count: int) -> Tuple[str, List[List[int]], List[int]]:
    base = root / CIFAR_DIRNAME
    images: List[List[int]] = []
    labels: List[int] = []
    if not base.exists():
        raise FileNotFoundError(base)

    for batch_id in range(1, 6):
        batch_file = base / f"data_batch_{batch_id}"
        with batch_file.open("rb") as f:
            obj = pickle.load(f, encoding="bytes")
        data = obj[b"data"]
        batch_labels = obj[b"labels"]
        for i in range(len(batch_labels)):
            images.append(list(data[i]))
            labels.append(int(batch_labels[i]))
            if len(images) >= count:
                return "cifar10", images, labels
    return "cifar10", images, labels


def make_fake_cifar(count: int) -> Tuple[str, List[List[int]], List[int]]:
    rnd = random.Random(42)
    images: List[List[int]] = []
    labels: List[int] = []
    for i in range(count):
        label = i % 10
        base = label * 20
        img = [(base + rnd.randrange(0, 236)) % 256 for _ in range(3072)]
        images.append(img)
        labels.append(label)
    return "fake-cifar10-like", images, labels


def extract_feature(raw: Sequence[int]) -> List[float]:
    # raw: 3072 bytes, channel-major (1024 R, 1024 G, 1024 B)
    r = raw[0:1024]
    g = raw[1024:2048]
    b = raw[2048:3072]

    feat: List[float] = []
    for ch in (r, g, b):
        mean = sum(ch) / 1024.0
        var = sum((v - mean) * (v - mean) for v in ch) / 1024.0
        feat.extend([mean / 255.0, math.sqrt(var) / 255.0])

    # 4x4 pooled (per channel): 48 dims
    for ch in (r, g, b):
        for by in range(4):
            for bx in range(4):
                s = 0
                for y in range(by * 8, by * 8 + 8):
                    row = y * 32
                    for x in range(bx * 8, bx * 8 + 8):
                        s += ch[row + x]
                feat.append((s / 64.0) / 255.0)

    # 8x8 pooled (per channel): 192 dims
    for ch in (r, g, b):
        for by in range(8):
            for bx in range(8):
                s = 0
                for y in range(by * 4, by * 4 + 4):
                    row = y * 32
                    for x in range(bx * 4, bx * 4 + 4):
                        s += ch[row + x]
                feat.append((s / 16.0) / 255.0)

    # Simple edge magnitudes: 4 dims
    for ch in (r, g):
        gx = 0.0
        gy = 0.0
        for y in range(32):
            row = y * 32
            for x in range(31):
                gx += abs(ch[row + x + 1] - ch[row + x])
        for y in range(31):
            row = y * 32
            row2 = (y + 1) * 32
            for x in range(32):
                gy += abs(ch[row2 + x] - ch[row + x])
        feat.extend([gx / (32 * 31 * 255.0), gy / (31 * 32 * 255.0)])

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in feat))
    if norm > 0:
        feat = [v / norm for v in feat]
    return feat


def extract_features(images: Sequence[Sequence[int]]) -> Tuple[List[List[float]], float]:
    t0 = time.perf_counter()
    vecs = [extract_feature(img) for img in images]
    return vecs, time.perf_counter() - t0


def percentile(vals: Sequence[float], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    idx = int((len(arr) - 1) * p / 100.0)
    return arr[idx]


def run(args: argparse.Namespace) -> BenchResult:
    if not args.lib.exists():
        raise SystemExit(f"missing shared library: {args.lib}")

    backend = ""
    images: List[List[int]]
    labels: List[int]

    try:
        if args.download:
            maybe_download_cifar(args.dataset_root)
        backend, images, labels = load_cifar_samples(args.dataset_root, args.images)
    except Exception:
        if not args.allow_fake_fallback:
            raise
        backend, images, labels = make_fake_cifar(args.images)

    vecs, feat_time = extract_features(images)
    dim = len(vecs[0])
    ids = [i + 1 for i in range(len(vecs))]

    with tempfile.TemporaryDirectory(prefix="pomai_py_bench_") as td:
        client = PomaiClient(args.lib, Path(td), dim=dim, shards=args.shards)
        try:
            t0 = time.perf_counter()
            for i in range(0, len(ids), args.batch_size):
                client.put_batch(ids[i:i + args.batch_size], vecs[i:i + args.batch_size])
            ingest_time = time.perf_counter() - t0

            scan_t0 = time.perf_counter()
            visible, mean_vec = client.full_scan_mean()
            scan_time = time.perf_counter() - scan_t0
            if visible <= 0:
                raise RuntimeError("No visible rows from iterator scan")
            if any((not math.isfinite(v)) for v in mean_vec):
                raise RuntimeError("Non-finite values found in scan analytics")

            queries = min(args.queries, visible)
            idxs = [int(i * (visible - 1) / max(queries - 1, 1)) for i in range(queries)]
            lat = []
            t_search = time.perf_counter()
            hit = 0
            for qi in idxs:
                qvec = vecs[qi]
                q0 = time.perf_counter()
                got = client.search(qvec, args.topk)
                lat.append((time.perf_counter() - q0) * 1000.0)
                if got:
                    nn = got[0]
                    if 1 <= nn <= visible and labels[nn - 1] == labels[qi]:
                        hit += 1
            search_time = time.perf_counter() - t_search

            return BenchResult(
                backend=backend,
                requested=len(vecs),
                visible=visible,
                feat_time_s=feat_time,
                feat_rate=len(vecs) / max(feat_time, 1e-9),
                ingest_time_s=ingest_time,
                ingest_rate=len(vecs) / max(ingest_time, 1e-9),
                search_p50_ms=percentile(lat, 50),
                search_p95_ms=percentile(lat, 95),
                search_p99_ms=percentile(lat, 99),
                search_qps=queries / max(search_time, 1e-9),
                knn_top1=hit / max(queries, 1),
                scan_time_s=scan_time,
                scan_rate=visible / max(scan_time, 1e-9),
            )
        finally:
            client.close()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="PomaiDB Python CIFAR-10 benchmark")
    p.add_argument("--lib", type=Path, default=root / "build" / "libpomai_c.so")
    p.add_argument("--dataset-root", type=Path, default=root / "data")
    p.add_argument("--images", type=int, default=5200)
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--download", action="store_true")
    p.add_argument("--allow-fake-fallback", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.images < 5001:
        raise SystemExit("--images must be >= 5001 (soft freeze visibility)")
    if args.queries < 1 or args.topk < 1:
        raise SystemExit("--queries and --topk must be >= 1")

    r = run(args)
    print("=" * 68)
    print("PomaiDB Python CIFAR-10 Real-World Benchmark")
    print("=" * 68)
    print(f"Dataset backend:          {r.backend}")
    print(f"Requested images:         {r.requested}")
    print(f"Visible vectors:          {r.visible}")
    print("-" * 68)
    print(f"Feature extraction:       {r.feat_time_s:.3f}s  ({r.feat_rate:,.1f} img/s)")
    print(f"Ingestion throughput:     {r.ingest_time_s:.3f}s  ({r.ingest_rate:,.1f} vec/s)")
    print(f"Search P50/P95/P99:       {r.search_p50_ms:.3f}/{r.search_p95_ms:.3f}/{r.search_p99_ms:.3f} ms")
    print(f"Search QPS:               {r.search_qps:,.1f}")
    print(f"kNN top-1 agreement:      {100.0 * r.knn_top1:.2f}%")
    print(f"Iterator scan throughput: {r.scan_time_s:.3f}s  ({r.scan_rate:,.1f} vec/s)")
    print("=" * 68)


if __name__ == "__main__":
    main()
