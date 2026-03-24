#!/usr/bin/env python3
"""PomaiDB Python ctypes example (C ABI).

Layout mirrors include/pomai/c_types.h — keep in sync when the ABI changes.

Build & run (from repo root):
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --target pomai_c
  POMAI_C_LIB=./build/libpomai_c.so python3 examples/python_basic.py
"""
from __future__ import annotations

import ctypes
import os
import random
import sys
from pathlib import Path


# --- Structs: must match include/pomai/c_types.h --------------------------------


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
        ("partition_device_id", ctypes.c_char_p),
        ("partition_location_id", ctypes.c_char_p),
        ("as_of_ts", ctypes.c_uint64),
        ("as_of_lsn", ctypes.c_uint64),
        ("aggregate_op", ctypes.c_uint32),
        ("aggregate_field", ctypes.c_char_p),
        ("aggregate_topk", ctypes.c_uint32),
        ("mesh_detail_preference", ctypes.c_uint32),
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
        ("total_shards_count", ctypes.c_uint32),
        ("pruned_shards_count", ctypes.c_uint32),
        ("aggregate_value", ctypes.c_double),
        ("aggregate_op", ctypes.c_uint32),
        ("mesh_lod_level", ctypes.c_uint32),
        ("zero_copy_pointers", ctypes.POINTER(PomaiSemanticPointer)),
    ]


def load_lib() -> ctypes.CDLL:
    default_name = "libpomai_c.so" if sys.platform != "darwin" else "libpomai_c.dylib"
    lib_path = Path(os.environ.get("POMAI_C_LIB", f"./build/{default_name}")).resolve()
    if not lib_path.exists():
        raise FileNotFoundError(f"PomaiDB C library not found: {lib_path}")
    lib = ctypes.CDLL(str(lib_path))

    lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]
    lib.pomai_options_init.restype = None

    lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
    lib.pomai_open.restype = ctypes.c_void_p

    lib.pomai_close.argtypes = [ctypes.c_void_p]
    lib.pomai_close.restype = ctypes.c_void_p

    lib.pomai_put_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert), ctypes.c_size_t]
    lib.pomai_put_batch.restype = ctypes.c_void_p

    lib.pomai_freeze.argtypes = [ctypes.c_void_p]
    lib.pomai_freeze.restype = ctypes.c_void_p

    lib.pomai_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PomaiQuery),
        ctypes.POINTER(ctypes.POINTER(PomaiSearchResults)),
    ]
    lib.pomai_search.restype = ctypes.c_void_p

    lib.pomai_search_results_free.argtypes = [ctypes.POINTER(PomaiSearchResults)]
    lib.pomai_search_results_free.restype = None

    lib.pomai_status_message.argtypes = [ctypes.c_void_p]
    lib.pomai_status_message.restype = ctypes.c_char_p
    lib.pomai_status_free.argtypes = [ctypes.c_void_p]
    lib.pomai_status_free.restype = None

    return lib


def check_status(lib: ctypes.CDLL, status) -> None:
    if status:
        msg = lib.pomai_status_message(status).decode("utf-8", errors="replace")
        lib.pomai_status_free(status)
        raise RuntimeError(msg)


def fill_query_defaults(q: PomaiQuery, vec, dim: int, topk: int) -> None:
    """Zero optional fields; set required search fields."""
    q.struct_size = ctypes.sizeof(PomaiQuery)
    q.vector = ctypes.cast(vec, ctypes.POINTER(ctypes.c_float))
    q.dim = dim
    q.topk = topk
    q.filter_expression = None
    q.partition_device_id = None
    q.partition_location_id = None
    q.as_of_ts = 0
    q.as_of_lsn = 0
    q.aggregate_op = 0
    q.aggregate_field = None
    q.aggregate_topk = 0
    q.mesh_detail_preference = 0
    q.alpha = 1.0
    q.deadline_ms = 0
    q.flags = 0


def main() -> None:
    lib = load_lib()
    db = ctypes.c_void_p()

    opts = PomaiOptions()
    lib.pomai_options_init(ctypes.byref(opts))
    opts.struct_size = ctypes.sizeof(PomaiOptions)
    opts.path = str(Path("/tmp/pomai_example_py").resolve()).encode()
    opts.shards = 1
    opts.dim = 8
    opts.search_threads = 0

    check_status(lib, lib.pomai_open(ctypes.byref(opts), ctypes.byref(db)))

    random.seed(42)
    dim = opts.dim
    n = 100

    vectors = []
    upserts = (PomaiUpsert * n)()
    for i in range(n):
        vec = (ctypes.c_float * dim)(*[(random.random() * 2 - 1) for _ in range(dim)])
        vectors.append(vec)
        upserts[i].struct_size = ctypes.sizeof(PomaiUpsert)
        upserts[i].id = i
        upserts[i].vector = ctypes.cast(vec, ctypes.POINTER(ctypes.c_float))
        upserts[i].dim = dim
        upserts[i].metadata = None
        upserts[i].metadata_len = 0

    check_status(lib, lib.pomai_put_batch(db, upserts, n))
    check_status(lib, lib.pomai_freeze(db))

    query_vec = vectors[0]
    query = PomaiQuery()
    fill_query_defaults(query, query_vec, dim, 5)

    out_ptr = ctypes.POINTER(PomaiSearchResults)()
    check_status(lib, lib.pomai_search(db, ctypes.byref(query), ctypes.byref(out_ptr)))
    results = out_ptr.contents

    print("TopK results:")
    for i in range(results.count):
        print(f"  id={results.ids[i]} score={results.scores[i]:.4f}")

    lib.pomai_search_results_free(out_ptr)
    check_status(lib, lib.pomai_close(db))


if __name__ == "__main__":
    main()
