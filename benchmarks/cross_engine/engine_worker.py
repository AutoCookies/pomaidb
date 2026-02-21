#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def load_f32bin(path: Path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        n, d = int(header[0]), int(header[1])
        arr = np.fromfile(f, dtype=np.float32, count=n * d).reshape(n, d)
    return arr


def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int = 10) -> float:
    hits = 0
    for i in range(pred_ids.shape[0]):
        hits += len(set(pred_ids[i, :k].tolist()).intersection(set(gt_ids[i, :k].tolist())))
    return hits / float(pred_ids.shape[0] * k)


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


def run_pomai(base, queries, gt, lib_path: Path, repeats: int):
    lib = ctypes.CDLL(str(lib_path))
    lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]
    lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
    lib.pomai_put_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert), ctypes.c_size_t]
    lib.pomai_freeze.argtypes = [ctypes.c_void_p]
    lib.pomai_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiQuery), ctypes.POINTER(ctypes.POINTER(PomaiSearchResults))]
    lib.pomai_search_results_free.argtypes = [ctypes.POINTER(PomaiSearchResults)]
    lib.pomai_status_message.argtypes = [ctypes.c_void_p]
    lib.pomai_status_message.restype = ctypes.c_char_p
    lib.pomai_status_free.argtypes = [ctypes.c_void_p]
    lib.pomai_close.argtypes = [ctypes.c_void_p]

    def check(st):
        if st:
            msg = lib.pomai_status_message(st).decode("utf-8", errors="replace")
            lib.pomai_status_free(st)
            raise RuntimeError(msg)

    tmpdir = Path(tempfile.mkdtemp(prefix="pomai_bench_"))
    db = ctypes.c_void_p()
    opts = PomaiOptions()
    lib.pomai_options_init(ctypes.byref(opts))
    opts.struct_size = ctypes.sizeof(PomaiOptions)
    opts.path = str(tmpdir / "db").encode()
    opts.shards = 4
    opts.dim = base.shape[1]
    check(lib.pomai_open(ctypes.byref(opts), ctypes.byref(db)))

    ids = np.arange(base.shape[0], dtype=np.uint64)
    ingest_start = time.perf_counter()
    bs = 1000
    holder = []
    for s in range(0, base.shape[0], bs):
        e = min(s + bs, base.shape[0])
        n = e - s
        batch = (PomaiUpsert * n)()
        for i in range(n):
            v = (ctypes.c_float * base.shape[1])(*base[s + i])
            holder.append(v)
            batch[i].struct_size = ctypes.sizeof(PomaiUpsert)
            batch[i].id = int(ids[s + i])
            batch[i].vector = v
            batch[i].dim = base.shape[1]
            batch[i].metadata = None
            batch[i].metadata_len = 0
        check(lib.pomai_put_batch(db, batch, n))
    ingestion_s = time.perf_counter() - ingest_start

    build_start = time.perf_counter()
    check(lib.pomai_freeze(db))
    build_s = time.perf_counter() - build_start

    all_lat = []
    qps = []
    pred = []
    for r in range(repeats):
        lat = []
        start = time.perf_counter()
        run_pred = []
        for qv in queries:
            cvec = (ctypes.c_float * base.shape[1])(*qv)
            q = PomaiQuery()
            q.struct_size = ctypes.sizeof(PomaiQuery)
            q.vector = cvec
            q.dim = base.shape[1]
            q.topk = 10
            q.filter_expression = None
            q.alpha = ctypes.c_float(0.0)
            q.deadline_ms = 0
            out = ctypes.POINTER(PomaiSearchResults)()
            t0 = time.perf_counter()
            check(lib.pomai_search(db, ctypes.byref(q), ctypes.byref(out)))
            lat.append((time.perf_counter() - t0) * 1000.0)
            run_pred.append([int(out.contents.ids[i]) for i in range(min(10, out.contents.count))])
            lib.pomai_search_results_free(out)
        elapsed = time.perf_counter() - start
        qps.append(len(queries) / elapsed)
        all_lat.append(float(np.mean(lat)))
        pred = run_pred

    lib.pomai_close(db)
    disk_bytes = sum(f.stat().st_size for f in tmpdir.rglob("*") if f.is_file())
    shutil.rmtree(tmpdir, ignore_errors=True)

    pred_ids = np.array(pred, dtype=np.int64)
    rec = recall_at_k(pred_ids, gt, 10)
    return {
        "engine": "PomaiDB",
        "params": {"shards": 4, "topk": 10, "durability": "default (WAL enabled by default)"},
        "ingestion_time_s": ingestion_s,
        "index_build_time_s": build_s,
        "query_throughput_qps": float(np.mean(qps)),
        "avg_latency_ms": float(np.mean(all_lat)),
        "disk_usage_bytes": int(disk_bytes),
        "recall_at_10": rec,
    }


def run_hnswlib(base, queries, gt, repeats: int):
    import hnswlib

    idx = hnswlib.Index(space="l2", dim=base.shape[1])
    t0 = time.perf_counter()
    idx.init_index(max_elements=base.shape[0], M=16, ef_construction=200)
    build_base = time.perf_counter() - t0
    t1 = time.perf_counter()
    idx.add_items(base, np.arange(base.shape[0]))
    ingest = time.perf_counter() - t1
    idx.set_ef(64)

    tmp = Path(tempfile.mkdtemp(prefix="hnswlib_bench_"))
    index_file = tmp / "index.bin"
    idx.save_index(str(index_file))

    qps, lat = [], []
    pred = None
    for _ in range(repeats):
        t = time.perf_counter()
        labels, _ = idx.knn_query(queries, k=10)
        elapsed = time.perf_counter() - t
        qps.append(len(queries) / elapsed)
        lat.append((elapsed / len(queries)) * 1000.0)
        pred = labels

    rec = recall_at_k(pred, gt, 10)
    disk = index_file.stat().st_size
    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "engine": "hnswlib",
        "params": {"M": 16, "efConstruction": 200, "efSearch": 64, "topk": 10},
        "ingestion_time_s": ingest,
        "index_build_time_s": build_base,
        "query_throughput_qps": float(np.mean(qps)),
        "avg_latency_ms": float(np.mean(lat)),
        "disk_usage_bytes": int(disk),
        "recall_at_10": rec,
    }


def run_faiss_flat(base, queries, gt, repeats: int):
    import faiss

    idx = faiss.IndexFlatL2(base.shape[1])
    t0 = time.perf_counter()
    idx.add(base)
    ingest = time.perf_counter() - t0

    qps, lat = [], []
    pred = None
    for _ in range(repeats):
        t = time.perf_counter()
        _, i = idx.search(queries, 10)
        elapsed = time.perf_counter() - t
        qps.append(len(queries) / elapsed)
        lat.append((elapsed / len(queries)) * 1000.0)
        pred = i
    tmp = Path(tempfile.mkdtemp(prefix="faiss_flat_"))
    fpath = tmp / "index.faiss"
    faiss.write_index(idx, str(fpath))
    disk = fpath.stat().st_size
    shutil.rmtree(tmp, ignore_errors=True)

    return {
        "engine": "faiss.IndexFlatL2",
        "params": {"topk": 10},
        "ingestion_time_s": ingest,
        "index_build_time_s": 0.0,
        "query_throughput_qps": float(np.mean(qps)),
        "avg_latency_ms": float(np.mean(lat)),
        "disk_usage_bytes": int(disk),
        "recall_at_10": recall_at_k(pred, gt, 10),
    }


def run_faiss_hnsw(base, queries, gt, repeats: int):
    import faiss

    idx = faiss.IndexHNSWFlat(base.shape[1], 32)
    idx.hnsw.efConstruction = 200
    idx.hnsw.efSearch = 64
    t0 = time.perf_counter()
    idx.add(base)
    ingest = time.perf_counter() - t0

    qps, lat = [], []
    pred = None
    for _ in range(repeats):
        t = time.perf_counter()
        _, i = idx.search(queries, 10)
        elapsed = time.perf_counter() - t
        qps.append(len(queries) / elapsed)
        lat.append((elapsed / len(queries)) * 1000.0)
        pred = i
    tmp = Path(tempfile.mkdtemp(prefix="faiss_hnsw_"))
    fpath = tmp / "index.faiss"
    faiss.write_index(idx, str(fpath))
    disk = fpath.stat().st_size
    shutil.rmtree(tmp, ignore_errors=True)

    return {
        "engine": "faiss.IndexHNSWFlat",
        "params": {"M": 32, "efConstruction": 200, "efSearch": 64, "topk": 10},
        "ingestion_time_s": ingest,
        "index_build_time_s": 0.0,
        "query_throughput_qps": float(np.mean(qps)),
        "avg_latency_ms": float(np.mean(lat)),
        "disk_usage_bytes": int(disk),
        "recall_at_10": recall_at_k(pred, gt, 10),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--libpomai", default="")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    base = load_f32bin(Path(args.dataset))
    queries = load_f32bin(Path(args.queries))
    gt = np.load(args.ground_truth)

    if args.engine == "pomai":
        result = run_pomai(base, queries, gt, Path(args.libpomai), args.repeats)
    elif args.engine == "hnswlib":
        result = run_hnswlib(base, queries, gt, args.repeats)
    elif args.engine == "faiss_flat":
        result = run_faiss_flat(base, queries, gt, args.repeats)
    elif args.engine == "faiss_hnsw":
        result = run_faiss_hnsw(base, queries, gt, args.repeats)
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mb"] = float(rss_kb / 1024.0)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
