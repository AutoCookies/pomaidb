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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


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
        ("index_type", ctypes.c_uint8),
        ("hnsw_m", ctypes.c_uint32),
        ("hnsw_ef_construction", ctypes.c_uint32),
        ("hnsw_ef_search", ctypes.c_uint32),
        ("adaptive_threshold", ctypes.c_uint32),
        ("metric", ctypes.c_uint8),
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
        ("flags", ctypes.c_uint32),
    ]


class PomaiSemanticPointer(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("raw_data_ptr", ctypes.c_void_p),
        ("dim", ctypes.c_uint32),
        ("quant_min", ctypes.c_float),
        ("quant_inv_scale", ctypes.c_float),
        ("session_id", ctypes.c_uint64),
    ]


class PomaiSearchResults(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("count", ctypes.c_size_t),
        ("ids", ctypes.POINTER(ctypes.c_uint64)),
        ("scores", ctypes.POINTER(ctypes.c_float)),
        ("shard_ids", ctypes.POINTER(ctypes.c_uint32)),
        ("zero_copy_pointers", ctypes.POINTER(PomaiSemanticPointer)),
    ]


def run_pomai(base, queries, gt, lib_path: Path, repeats: int, metric: str):
    lib = ctypes.CDLL(str(lib_path))
    lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]
    lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
    lib.pomai_put_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert), ctypes.c_size_t]
    lib.pomai_freeze.argtypes = [ctypes.c_void_p]
    lib.pomai_search.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiQuery), ctypes.POINTER(ctypes.POINTER(PomaiSearchResults))]
    lib.pomai_search_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiQuery), ctypes.c_size_t, ctypes.POINTER(ctypes.POINTER(PomaiSearchResults))]
    lib.pomai_search_results_free.argtypes = [ctypes.POINTER(PomaiSearchResults)]
    lib.pomai_search_batch_free.argtypes = [ctypes.POINTER(PomaiSearchResults), ctypes.c_size_t]
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
    opts.index_type = 1 # HNSW
    opts.hnsw_m = 32
    opts.hnsw_ef_construction = 200
    opts.hnsw_ef_search = 64
    opts.adaptive_threshold = 0 
    opts.metric = 1 if metric == "ip" else 0
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
        holder.clear() # Fix memory leak: allow GC of ctypes arrays
    ingestion_s = time.perf_counter() - ingest_start

    build_start = time.perf_counter()
    check(lib.pomai_freeze(db))
    build_s = time.perf_counter() - build_start

    all_lat = []
    qps = []
    pred = []
    
    num_queries = len(queries)
    batch_queries = (PomaiQuery * num_queries)()
    c_queries_arrays = [] # keep references to avoid GC
    for i in range(num_queries):
        cvec = (ctypes.c_float * base.shape[1])(*queries[i])
        c_queries_arrays.append(cvec)
        batch_queries[i].struct_size = ctypes.sizeof(PomaiQuery)
        batch_queries[i].vector = cvec
        batch_queries[i].dim = base.shape[1]
        batch_queries[i].topk = 10
        batch_queries[i].filter_expression = None
        batch_queries[i].alpha = ctypes.c_float(0.0)
        batch_queries[i].deadline_ms = 0
        batch_queries[i].flags = 0

    for r in range(repeats):
        out = ctypes.POINTER(PomaiSearchResults)()
        start = time.perf_counter()
        
        check(lib.pomai_search_batch(db, batch_queries, num_queries, ctypes.byref(out)))
        
        elapsed = time.perf_counter() - start
        
        run_pred = []
        for i in range(num_queries):
            run_pred.append([int(out[i].ids[j]) for j in range(min(10, out[i].count))])
            
        lib.pomai_search_batch_free(out, num_queries)
        
        qps.append(num_queries / elapsed)
        all_lat.append((elapsed / num_queries) * 1000.0) 
        pred = run_pred

    lib.pomai_close(db)
    disk_bytes = sum(f.stat().st_size for f in tmpdir.rglob("*") if f.is_file())
    shutil.rmtree(tmpdir, ignore_errors=True)

    pred_ids = np.array(pred, dtype=np.int64)
    rec = recall_at_k(pred_ids, gt, 10)
    return {
        "engine": "PomaiDB HNSW",
        "params": {"shards": 4, "topk": 10, "M": 32, "efConstruction": 200, "efSearch": 64},
        "ingestion_time_s": ingestion_s,
        "index_build_time_s": build_s,
        "query_throughput_qps": float(np.mean(qps)),
        "avg_latency_ms": float(np.mean(all_lat)),
        "disk_usage_bytes": int(disk_bytes),
        "recall_at_10": rec,
    }


def run_hnswlib(base, queries, gt, repeats: int, metric: str):
    import hnswlib

    idx = hnswlib.Index(space="l2" if metric == "l2" else "ip", dim=base.shape[1])
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


def run_faiss_flat(base, queries, gt, repeats: int, metric: str):
    import faiss

    if metric == "l2":
        idx = faiss.IndexFlatL2(base.shape[1])
        engine_name = "faiss.IndexFlatL2"
    else:
        idx = faiss.IndexFlatIP(base.shape[1])
        engine_name = "faiss.IndexFlatIP"
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
        "engine": engine_name,
        "params": {"topk": 10},
        "ingestion_time_s": ingest,
        "index_build_time_s": 0.0,
        "query_throughput_qps": float(np.mean(qps)),
        "avg_latency_ms": float(np.mean(lat)),
        "disk_usage_bytes": int(disk),
        "recall_at_10": recall_at_k(pred, gt, 10),
    }


def run_faiss_hnsw(base, queries, gt, repeats: int, metric: str):
    import faiss

    if metric == "l2":
        idx = faiss.IndexHNSWFlat(base.shape[1], 32)
        engine_name = "faiss.IndexHNSWFlat(L2)"
    else:
        idx = faiss.IndexHNSWFlat(base.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        engine_name = "faiss.IndexHNSWFlat(IP)"
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
        "engine": engine_name,
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
    p.add_argument("--metric", choices=["l2", "ip", "cosine"], default="ip")
    args = p.parse_args()

    base = load_f32bin(Path(args.dataset))
    queries = load_f32bin(Path(args.queries))
    if args.metric == "cosine":
        base = normalize_rows(base)
        queries = normalize_rows(queries)
    gt = np.load(args.ground_truth)

    if args.engine == "pomai":
        result = run_pomai(base, queries, gt, Path(args.libpomai), args.repeats, args.metric)
    elif args.engine == "hnswlib":
        result = run_hnswlib(base, queries, gt, args.repeats, args.metric)
    elif args.engine == "faiss_flat":
        result = run_faiss_flat(base, queries, gt, args.repeats, args.metric)
    elif args.engine == "faiss_hnsw":
        result = run_faiss_hnsw(base, queries, gt, args.repeats, args.metric)
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_mb"] = float(rss_kb / 1024.0)
    result["metric"] = args.metric

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
