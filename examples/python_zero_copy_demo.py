#!/usr/bin/env python3
"""
Optional zero-copy vector read path (POMAI_QUERY_FLAG_ZERO_COPY).
Requires libpomai_c and numpy. Struct layouts must match include/pomai/c_types.h.

  POMAI_C_LIB=./build/libpomai_c.so python3 examples/python_zero_copy_demo.py
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

# --- Same layout as examples/python_basic.py / c_types.h --------------------


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


POMAI_QUERY_FLAG_ZERO_COPY = 1


def load_lib() -> ctypes.CDLL:
    default = "libpomai_c.so" if sys.platform != "darwin" else "libpomai_c.dylib"
    path = os.environ.get("POMAI_C_LIB", os.path.join("build", default))
    lib = ctypes.CDLL(path)

    lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]
    lib.pomai_options_init.restype = None
    lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
    lib.pomai_open.restype = ctypes.c_void_p
    lib.pomai_close.argtypes = [ctypes.c_void_p]
    lib.pomai_close.restype = ctypes.c_void_p
    lib.pomai_put.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert)]
    lib.pomai_put.restype = ctypes.c_void_p
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
    lib.pomai_release_pointer.argtypes = [ctypes.c_uint64]
    lib.pomai_release_pointer.restype = None
    return lib


def check_st(lib: ctypes.CDLL, st) -> None:
    if st:
        msg = lib.pomai_status_message(st).decode("utf-8", errors="replace")
        lib.pomai_status_free(st)
        raise RuntimeError(msg)


def main() -> None:
    lib = load_lib()
    opts = PomaiOptions()
    lib.pomai_options_init(ctypes.byref(opts))
    opts.struct_size = ctypes.sizeof(PomaiOptions)
    opts.path = b"test_db_zero_copy"
    opts.shards = 1
    opts.dim = 128

    db = ctypes.c_void_p()
    check_st(lib, lib.pomai_open(ctypes.byref(opts), ctypes.byref(db)))

    dim = 128
    vec = (ctypes.c_float * dim)(*[float(x * 0.01) for x in range(dim)])

    upsert = PomaiUpsert()
    upsert.struct_size = ctypes.sizeof(PomaiUpsert)
    upsert.id = 42
    upsert.vector = ctypes.cast(vec, ctypes.POINTER(ctypes.c_float))
    upsert.dim = dim
    upsert.metadata = None
    upsert.metadata_len = 0
    check_st(lib, lib.pomai_put(db, ctypes.byref(upsert)))
    check_st(lib, lib.pomai_freeze(db))

    q = PomaiQuery()
    q.struct_size = ctypes.sizeof(PomaiQuery)
    q.vector = ctypes.cast(vec, ctypes.POINTER(ctypes.c_float))
    q.dim = dim
    q.topk = 5
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
    q.flags = POMAI_QUERY_FLAG_ZERO_COPY

    res = ctypes.POINTER(PomaiSearchResults)()
    st = lib.pomai_search(db, ctypes.byref(q), ctypes.byref(res))
    if st:
        err = lib.pomai_status_message(st).decode("utf-8", errors="replace")
        lib.pomai_status_free(st)
        print("Search failed:", err)
        check_st(lib, lib.pomai_close(db))
        return

    count = res.contents.count
    print(f"Search OK, hits={count}")
    zp = res.contents.zero_copy_pointers
    if count > 0 and zp:
        ptr = zp[0]
        if ptr.raw_data_ptr:
            print(f"Zero-copy session_id={ptr.session_id}")
            addr = int(ctypes.cast(ptr.raw_data_ptr, ctypes.c_void_p).value or 0)
            buf = (ctypes.c_uint8 * dim).from_address(addr)
            arr = np.frombuffer(buf, dtype=np.uint8, count=dim)
            print(f"SQ8 head: {arr[:5]}")
            deq = arr.astype(np.float32) * ptr.quant_inv_scale + ptr.quant_min
            print(f"Dequant head: {deq[:5]}")
            lib.pomai_release_pointer(ptr.session_id)
        else:
            print("zero_copy raw_data_ptr is NULL (data may still be in memtable).")
    lib.pomai_search_results_free(res)
    check_st(lib, lib.pomai_close(db))


if __name__ == "__main__":
    main()
