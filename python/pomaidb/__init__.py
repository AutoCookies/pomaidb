"""
PomaiDB — embedded vector database for Edge AI.

Use the C library (libpomai_c.so / libpomai_c.dylib) via ctypes.
Set POMAI_C_LIB to the path to the shared library, or build from source
and point to build/libpomai_c.so (Linux) or build/libpomai_c.dylib (macOS).
"""

import ctypes
import json
import os
from pathlib import Path

__all__ = [
    "open_db", "close", "put_batch", "freeze", "search_batch",
    "resolve_effective_options",
    "meta_put", "meta_get", "meta_delete",
    "create_membrane_kind",
    "update_membrane_retention",
    "get_membrane_retention",
    "link_objects", "unlink_objects",
    "start_edge_gateway", "stop_edge_gateway",
    "list_membranes", "compact_membrane",
    "search_zero_copy", "release_zero_copy_session",
    "create_rag_membrane", "put_chunk", "search_rag",
    "ingest_document", "retrieve_context",
    "membrane_kind_capabilities",
    "MEMBRANE_KIND_VECTOR",
    "MEMBRANE_KIND_RAG",
    "MEMBRANE_KIND_GRAPH",
    "MEMBRANE_KIND_TEXT",
    "MEMBRANE_KIND_TIMESERIES",
    "MEMBRANE_KIND_KEYVALUE",
    "MEMBRANE_KIND_SKETCH",
    "MEMBRANE_KIND_BLOB",
    "MEMBRANE_KIND_SPATIAL",
    "MEMBRANE_KIND_MESH",
    "MEMBRANE_KIND_SPARSE",
    "MEMBRANE_KIND_BITSET",
    "MEMBRANE_KIND_META",
    "MEMBRANE_STABILITY_STABLE",
    "MEMBRANE_STABILITY_EXPERIMENTAL",
    "PomaiDBError",
]

# Mirror include/pomai/c_types.h (pomai::MembraneKind).
MEMBRANE_KIND_VECTOR = 0
MEMBRANE_KIND_RAG = 1
MEMBRANE_KIND_GRAPH = 2
MEMBRANE_KIND_TEXT = 3
MEMBRANE_KIND_TIMESERIES = 4
MEMBRANE_KIND_KEYVALUE = 5
MEMBRANE_KIND_SKETCH = 6
MEMBRANE_KIND_BLOB = 7
MEMBRANE_KIND_SPATIAL = 8
MEMBRANE_KIND_MESH = 9
MEMBRANE_KIND_SPARSE = 10
MEMBRANE_KIND_BITSET = 11
MEMBRANE_KIND_META = 12

MEMBRANE_STABILITY_STABLE = 0
MEMBRANE_STABILITY_EXPERIMENTAL = 1

# Default library path when running from repo (build dir)
def _find_lib():
    env = os.environ.get("POMAI_C_LIB")
    if env:
        return env
    # Try repo build dir relative to this file
    for base in [Path(__file__).resolve().parents[2], Path.cwd()]:
        for name in ["libpomai_c.so", "libpomai_c.dylib"]:
            p = base / "build" / name
            if p.exists():
                return str(p)
    return None


_lib_path = _find_lib()
_lib = None


def _ensure_lib():
    global _lib
    if _lib is not None:
        return
    path = _find_lib()
    if not path or not os.path.isfile(path):
        raise PomaiDBError(
            "PomaiDB C library not found. Set POMAI_C_LIB to path to libpomai_c.so (or .dylib), "
            "or build the project and run from repo root."
        )
    _lib = ctypes.CDLL(path)
    _register_api(_lib)


class PomaiDBError(Exception):
    """Raised when a PomaiDB operation fails."""
    pass


def _register_api(lib):
    # C types mirror include/pomai/c_types.h
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
            ("edge_profile", ctypes.c_uint8),
            ("gateway_rate_limit_per_sec", ctypes.c_uint32),
            ("gateway_idempotency_ttl_sec", ctypes.c_uint32),
            ("gateway_token_file", ctypes.c_char_p),
            ("gateway_upstream_sync_url", ctypes.c_char_p),
            ("gateway_upstream_sync_enabled", ctypes.c_bool),
            ("gateway_require_mtls_proxy_header", ctypes.c_bool),
            ("gateway_mtls_proxy_header", ctypes.c_char_p),
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
        # Must match include/pomai/c_types.h pomai_query_t (field order + padding).
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

    class PomaiMembraneCapabilities(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("kind", ctypes.c_uint8),
            ("stability", ctypes.c_uint8),
            ("reserved0", ctypes.c_uint8),
            ("reserved1", ctypes.c_uint8),
            ("read_path", ctypes.c_bool),
            ("write_path", ctypes.c_bool),
            ("unified_scan", ctypes.c_bool),
            ("snapshot_isolated_scan", ctypes.c_bool),
        ]

    lib.pomai_options_init.argtypes = [ctypes.POINTER(PomaiOptions)]
    lib.pomai_options_init.restype = None
    lib.pomai_open.argtypes = [ctypes.POINTER(PomaiOptions), ctypes.POINTER(ctypes.c_void_p)]
    lib.pomai_open.restype = ctypes.c_void_p
    lib.pomai_put_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(PomaiUpsert), ctypes.c_size_t]
    lib.pomai_put_batch.restype = ctypes.c_void_p
    lib.pomai_freeze.argtypes = [ctypes.c_void_p]
    lib.pomai_freeze.restype = ctypes.c_void_p
    lib.pomai_search_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PomaiQuery),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(PomaiSearchResults)),
    ]
    lib.pomai_search_batch.restype = ctypes.c_void_p
    lib.pomai_search_batch_free.argtypes = [ctypes.POINTER(PomaiSearchResults), ctypes.c_size_t]
    lib.pomai_search_batch_free.restype = None
    lib.pomai_close.argtypes = [ctypes.c_void_p]
    lib.pomai_close.restype = ctypes.c_void_p
    lib.pomai_meta_put.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.pomai_meta_put.restype = ctypes.c_void_p
    lib.pomai_meta_get.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.pomai_meta_get.restype = ctypes.c_void_p
    lib.pomai_meta_delete.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.pomai_meta_delete.restype = ctypes.c_void_p
    lib.pomai_create_membrane_kind_with_retention.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64,
    ]
    lib.pomai_create_membrane_kind_with_retention.restype = ctypes.c_void_p
    lib.pomai_link_objects.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    lib.pomai_link_objects.restype = ctypes.c_void_p
    lib.pomai_unlink_objects.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.pomai_unlink_objects.restype = ctypes.c_void_p
    lib.pomai_edge_gateway_start.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]
    lib.pomai_edge_gateway_start.restype = ctypes.c_void_p
    lib.pomai_edge_gateway_start_secure.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16, ctypes.c_char_p]
    lib.pomai_edge_gateway_start_secure.restype = ctypes.c_void_p
    lib.pomai_edge_gateway_stop.argtypes = [ctypes.c_void_p]
    lib.pomai_edge_gateway_stop.restype = ctypes.c_void_p
    lib.pomai_list_membranes_json.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.pomai_list_membranes_json.restype = ctypes.c_void_p
    lib.pomai_compact_membrane.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.pomai_compact_membrane.restype = ctypes.c_void_p
    lib.pomai_update_membrane_retention.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64
    ]
    lib.pomai_update_membrane_retention.restype = ctypes.c_void_p
    lib.pomai_get_membrane_retention_json.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)
    ]
    lib.pomai_get_membrane_retention_json.restype = ctypes.c_void_p
    lib.pomai_status_message.argtypes = [ctypes.c_void_p]
    lib.pomai_status_message.restype = ctypes.c_char_p
    lib.pomai_status_free.argtypes = [ctypes.c_void_p]
    lib.pomai_status_free.restype = None
    lib.pomai_membrane_capabilities_init.argtypes = [ctypes.POINTER(PomaiMembraneCapabilities)]
    lib.pomai_membrane_capabilities_init.restype = None
    lib.pomai_membrane_kind_capabilities.argtypes = [ctypes.c_uint8, ctypes.POINTER(PomaiMembraneCapabilities)]
    lib.pomai_membrane_kind_capabilities.restype = ctypes.c_void_p
    lib.pomai_options_resolve_json.argtypes = [
        ctypes.POINTER(PomaiOptions),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.pomai_options_resolve_json.restype = ctypes.c_void_p

    # RAG types
    class PomaiRagChunkOptions(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("max_chunk_bytes", ctypes.c_size_t),
            ("max_doc_bytes", ctypes.c_size_t),
            ("max_chunks_per_batch", ctypes.c_size_t),
            ("overlap_bytes", ctypes.c_size_t),
        ]

    class PomaiRagChunk(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("chunk_id", ctypes.c_uint64),
            ("doc_id", ctypes.c_uint64),
            ("token_ids", ctypes.POINTER(ctypes.c_uint32)),
            ("token_count", ctypes.c_size_t),
            ("vector", ctypes.POINTER(ctypes.c_float)),
            ("dim", ctypes.c_uint32),
            ("chunk_text", ctypes.c_char_p),
            ("chunk_text_len", ctypes.c_size_t),
        ]

    class PomaiRagQuery(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("token_ids", ctypes.POINTER(ctypes.c_uint32)),
            ("token_count", ctypes.c_size_t),
            ("vector", ctypes.POINTER(ctypes.c_float)),
            ("dim", ctypes.c_uint32),
            ("topk", ctypes.c_uint32),
        ]

    class PomaiRagSearchOptions(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("candidate_budget", ctypes.c_uint32),
            ("token_budget", ctypes.c_uint32),
            ("enable_vector_rerank", ctypes.c_bool),
        ]

    class PomaiRagHit(ctypes.Structure):
        _fields_ = [
            ("chunk_id", ctypes.c_uint64),
            ("doc_id", ctypes.c_uint64),
            ("score", ctypes.c_float),
            ("token_matches", ctypes.c_uint32),
            ("chunk_text", ctypes.c_char_p),
            ("chunk_text_len", ctypes.c_size_t),
        ]

    class PomaiRagSearchResult(ctypes.Structure):
        _fields_ = [
            ("hit_count", ctypes.c_size_t),
            ("hits", ctypes.POINTER(PomaiRagHit)),
        ]

    lib.pomai_create_rag_membrane.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]
    lib.pomai_create_rag_membrane.restype = ctypes.c_void_p
    lib.pomai_put_chunk.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(PomaiRagChunk)]
    lib.pomai_put_chunk.restype = ctypes.c_void_p
    lib.pomai_search_rag.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(PomaiRagQuery), ctypes.POINTER(PomaiRagSearchOptions),
        ctypes.POINTER(PomaiRagSearchResult),
    ]
    lib.pomai_search_rag.restype = ctypes.c_void_p
    lib.pomai_rag_search_result_free.argtypes = [ctypes.POINTER(PomaiRagSearchResult)]
    lib.pomai_rag_search_result_free.restype = None

    lib.pomai_rag_pipeline_create.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.POINTER(PomaiRagChunkOptions), ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.pomai_rag_pipeline_create.restype = ctypes.c_void_p
    lib.pomai_rag_ingest_document.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]
    lib.pomai_rag_ingest_document.restype = ctypes.c_void_p
    lib.pomai_rag_retrieve_context.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.pomai_rag_retrieve_context.restype = ctypes.c_void_p
    lib.pomai_rag_retrieve_context_buf.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.pomai_rag_retrieve_context_buf.restype = ctypes.c_void_p
    lib.pomai_rag_pipeline_free.argtypes = [ctypes.c_void_p]
    lib.pomai_rag_pipeline_free.restype = None
    lib.pomai_free.argtypes = [ctypes.c_void_p]
    lib.pomai_free.restype = None

    lib._pomai_options = PomaiOptions
    lib._pomai_upsert = PomaiUpsert
    lib._pomai_query = PomaiQuery
    lib._pomai_semantic_pointer = PomaiSemanticPointer
    lib._pomai_search_results = PomaiSearchResults
    lib._pomai_rag_chunk_options = PomaiRagChunkOptions
    lib._pomai_rag_chunk = PomaiRagChunk
    lib._pomai_rag_query = PomaiRagQuery
    lib._pomai_rag_search_options = PomaiRagSearchOptions
    lib._pomai_rag_hit = PomaiRagHit
    lib._pomai_rag_search_result = PomaiRagSearchResult
    lib._pomai_membrane_capabilities = PomaiMembraneCapabilities


def _check(st):
    if st:
        _ensure_lib()
        msg = _lib.pomai_status_message(st).decode("utf-8", errors="replace")
        _lib.pomai_status_free(st)
        raise PomaiDBError(msg)


def open_db(path, dim, *, shards=1, search_threads=0, fsync=False, metric="ip", profile=None, **hnsw_kw):
    """Open a PomaiDB database at `path` with dimension `dim`. Returns an opaque db handle."""
    _ensure_lib()
    opts = _lib._pomai_options()
    _lib.pomai_options_init(ctypes.byref(opts))
    opts.struct_size = ctypes.sizeof(_lib._pomai_options())
    opts.path = path.encode("utf-8")
    opts.shards = shards
    opts.dim = dim
    opts.search_threads = search_threads
    opts.fsync_policy = 1 if fsync else 0
    opts.metric = 1 if metric == "ip" else 0
    opts.index_type = 1
    opts.hnsw_m = hnsw_kw.get("hnsw_m", 32)
    opts.hnsw_ef_construction = hnsw_kw.get("hnsw_ef_construction", 200)
    opts.hnsw_ef_search = hnsw_kw.get("hnsw_ef_search", 64)
    opts.adaptive_threshold = hnsw_kw.get("adaptive_threshold", 0)
    profile_map = {
        None: 0,
        "user_defined": 0,
        "edge-low-ram": 1,
        "edge-balanced": 2,
        "edge-throughput": 3,
    }
    if profile not in profile_map:
        raise ValueError("profile must be one of: edge-low-ram, edge-balanced, edge-throughput, user_defined")
    opts.edge_profile = profile_map[profile]
    if "gateway_rate_limit_per_sec" in hnsw_kw:
        opts.gateway_rate_limit_per_sec = int(hnsw_kw.get("gateway_rate_limit_per_sec"))
    if "gateway_idempotency_ttl_sec" in hnsw_kw:
        opts.gateway_idempotency_ttl_sec = int(hnsw_kw.get("gateway_idempotency_ttl_sec"))
    if "gateway_token_file" in hnsw_kw and hnsw_kw.get("gateway_token_file"):
        opts.gateway_token_file = str(hnsw_kw.get("gateway_token_file")).encode("utf-8")
    if "gateway_upstream_sync_url" in hnsw_kw and hnsw_kw.get("gateway_upstream_sync_url"):
        opts.gateway_upstream_sync_url = str(hnsw_kw.get("gateway_upstream_sync_url")).encode("utf-8")
    opts.gateway_upstream_sync_enabled = bool(hnsw_kw.get("gateway_upstream_sync_enabled", False))
    opts.gateway_require_mtls_proxy_header = bool(hnsw_kw.get("gateway_require_mtls_proxy_header", False))
    if "gateway_mtls_proxy_header" in hnsw_kw and hnsw_kw.get("gateway_mtls_proxy_header"):
        opts.gateway_mtls_proxy_header = str(hnsw_kw.get("gateway_mtls_proxy_header")).encode("utf-8")
    db = ctypes.c_void_p()
    _check(_lib.pomai_open(ctypes.byref(opts), ctypes.byref(db)))
    return db


def resolve_effective_options(path, dim, *, shards=1, search_threads=0, fsync=False, metric="ip", profile=None, **hnsw_kw):
    """Return effective runtime options (JSON string) after profile application."""
    _ensure_lib()
    opts = _lib._pomai_options()
    _lib.pomai_options_init(ctypes.byref(opts))
    opts.struct_size = ctypes.sizeof(_lib._pomai_options())
    opts.path = path.encode("utf-8")
    opts.shards = shards
    opts.dim = dim
    opts.search_threads = search_threads
    opts.fsync_policy = 1 if fsync else 0
    opts.metric = 1 if metric == "ip" else 0
    opts.index_type = 1
    opts.hnsw_m = hnsw_kw.get("hnsw_m", 32)
    opts.hnsw_ef_construction = hnsw_kw.get("hnsw_ef_construction", 200)
    opts.hnsw_ef_search = hnsw_kw.get("hnsw_ef_search", 64)
    opts.adaptive_threshold = hnsw_kw.get("adaptive_threshold", 0)
    profile_map = {
        None: 0,
        "user_defined": 0,
        "edge-low-ram": 1,
        "edge-balanced": 2,
        "edge-throughput": 3,
    }
    if profile not in profile_map:
        raise ValueError("profile must be one of: edge-low-ram, edge-balanced, edge-throughput, user_defined")
    opts.edge_profile = profile_map[profile]
    if "gateway_rate_limit_per_sec" in hnsw_kw:
        opts.gateway_rate_limit_per_sec = int(hnsw_kw.get("gateway_rate_limit_per_sec"))
    if "gateway_idempotency_ttl_sec" in hnsw_kw:
        opts.gateway_idempotency_ttl_sec = int(hnsw_kw.get("gateway_idempotency_ttl_sec"))
    if "gateway_token_file" in hnsw_kw and hnsw_kw.get("gateway_token_file"):
        opts.gateway_token_file = str(hnsw_kw.get("gateway_token_file")).encode("utf-8")
    if "gateway_upstream_sync_url" in hnsw_kw and hnsw_kw.get("gateway_upstream_sync_url"):
        opts.gateway_upstream_sync_url = str(hnsw_kw.get("gateway_upstream_sync_url")).encode("utf-8")
    opts.gateway_upstream_sync_enabled = bool(hnsw_kw.get("gateway_upstream_sync_enabled", False))
    opts.gateway_require_mtls_proxy_header = bool(hnsw_kw.get("gateway_require_mtls_proxy_header", False))
    if "gateway_mtls_proxy_header" in hnsw_kw and hnsw_kw.get("gateway_mtls_proxy_header"):
        opts.gateway_mtls_proxy_header = str(hnsw_kw.get("gateway_mtls_proxy_header")).encode("utf-8")
    out = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    _check(_lib.pomai_options_resolve_json(ctypes.byref(opts), ctypes.byref(out), ctypes.byref(out_len)))
    try:
        if not out:
            return "{}"
        return ctypes.string_at(out, out_len.value).decode("utf-8", errors="replace")
    finally:
        if out:
            _lib.pomai_free(out)


def close(db):
    """Close the database and free resources."""
    if _lib is None:
        return
    _check(_lib.pomai_close(db))


def put_batch(db, ids, vectors):
    """Insert vectors. `ids`: sequence of int; `vectors`: 2D array-like (n, dim)."""
    _ensure_lib()
    n = len(ids)
    if n != len(vectors):
        raise ValueError("ids and vectors length mismatch")
    dim = len(vectors[0])
    batch = (_lib._pomai_upsert * n)()
    vec_holders = []
    for i in range(n):
        v = (ctypes.c_float * dim)(*vectors[i])
        vec_holders.append(v)
        batch[i].struct_size = ctypes.sizeof(_lib._pomai_upsert())
        batch[i].id = int(ids[i])
        batch[i].vector = v
        batch[i].dim = dim
        batch[i].metadata = None
        batch[i].metadata_len = 0
    _check(_lib.pomai_put_batch(db, batch, n))


def freeze(db):
    """Flush memtable to segment and build index. Call before search for new data to be visible."""
    _ensure_lib()
    _check(_lib.pomai_freeze(db))


def meta_put(db, membrane_name, gid, value):
    """Store metadata payload by global id in a kMeta membrane."""
    _ensure_lib()
    _check(_lib.pomai_meta_put(db, membrane_name.encode("utf-8"), gid.encode("utf-8"), value.encode("utf-8")))


def meta_get(db, membrane_name, gid):
    """Load metadata payload by global id from a kMeta membrane."""
    _ensure_lib()
    out = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    _check(_lib.pomai_meta_get(db, membrane_name.encode("utf-8"), gid.encode("utf-8"), ctypes.byref(out), ctypes.byref(out_len)))
    try:
        if not out:
            return ""
        return ctypes.string_at(out, out_len.value).decode("utf-8", errors="replace")
    finally:
        if out:
            _lib.pomai_free(out)


def meta_delete(db, membrane_name, gid):
    """Delete metadata payload by global id from a kMeta membrane."""
    _ensure_lib()
    _check(_lib.pomai_meta_delete(db, membrane_name.encode("utf-8"), gid.encode("utf-8")))

def create_membrane_kind(
    db, name, dim, shard_count, kind, *,
    ttl_sec=0, retention_max_count=0, retention_max_bytes=0
):
    """Create and open a membrane with optional retention policy."""
    _ensure_lib()
    _check(_lib.pomai_create_membrane_kind_with_retention(
        db,
        str(name).encode("utf-8"),
        int(dim),
        int(shard_count),
        int(kind),
        int(ttl_sec),
        int(retention_max_count),
        int(retention_max_bytes),
    ))


def link_objects(db, gid, vector_id, graph_vertex_id, mesh_id):
    """Link vector/graph/mesh records under a single GID."""
    _ensure_lib()
    _check(_lib.pomai_link_objects(db, gid.encode("utf-8"), int(vector_id), int(graph_vertex_id), int(mesh_id)))


def unlink_objects(db, gid):
    """Remove a previously registered GID link."""
    _ensure_lib()
    _check(_lib.pomai_unlink_objects(db, gid.encode("utf-8")))


def start_edge_gateway(db, http_port=8080, ingest_port=8090, auth_token=None):
    """Start embedded HTTP + ingestion listeners."""
    _ensure_lib()
    if auth_token:
        _check(_lib.pomai_edge_gateway_start_secure(
            db, int(http_port), int(ingest_port), auth_token.encode("utf-8")
        ))
    else:
        _check(_lib.pomai_edge_gateway_start(db, int(http_port), int(ingest_port)))


def stop_edge_gateway(db):
    """Stop embedded HTTP + ingestion listeners."""
    _ensure_lib()
    _check(_lib.pomai_edge_gateway_stop(db))

def list_membranes(db):
    """Return membrane names as a Python list."""
    _ensure_lib()
    out = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    _check(_lib.pomai_list_membranes_json(db, ctypes.byref(out), ctypes.byref(out_len)))
    try:
        if not out:
            return []
        return json.loads(ctypes.string_at(out, out_len.value).decode("utf-8", errors="replace"))
    finally:
        if out:
            _lib.pomai_free(out)

def compact_membrane(db, membrane_name):
    """Trigger manual compaction on a membrane."""
    _ensure_lib()
    _check(_lib.pomai_compact_membrane(db, membrane_name.encode("utf-8")))

def update_membrane_retention(db, membrane_name, *, ttl_sec=0, retention_max_count=0, retention_max_bytes=0):
    """Update retention policy for an existing membrane."""
    _ensure_lib()
    _check(_lib.pomai_update_membrane_retention(
        db,
        membrane_name.encode("utf-8"),
        int(ttl_sec),
        int(retention_max_count),
        int(retention_max_bytes),
    ))

def get_membrane_retention(db, membrane_name):
    """Return retention policy for an existing membrane as a dict."""
    _ensure_lib()
    out = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    _check(_lib.pomai_get_membrane_retention_json(
        db, membrane_name.encode("utf-8"), ctypes.byref(out), ctypes.byref(out_len)
    ))
    try:
        if not out:
            return {}
        return json.loads(ctypes.string_at(out, out_len.value).decode("utf-8", errors="replace"))
    finally:
        if out:
            _lib.pomai_free(out)


def search_batch(db, queries, topk=10):
    """Run batch search. `queries`: 2D array-like (n_queries, dim). Returns list of (ids, scores) per query."""
    _ensure_lib()
    n = len(queries)
    dim = len(queries[0])
    batch = (_lib._pomai_query * n)()
    q_holders = []
    for i in range(n):
        q = (ctypes.c_float * dim)(*queries[i])
        q_holders.append(q)
        batch[i].struct_size = ctypes.sizeof(_lib._pomai_query())
        batch[i].vector = q
        batch[i].dim = dim
        batch[i].topk = topk
        batch[i].filter_expression = None
        batch[i].partition_device_id = None
        batch[i].partition_location_id = None
        batch[i].as_of_ts = 0
        batch[i].as_of_lsn = 0
        batch[i].aggregate_op = 0
        batch[i].aggregate_field = None
        batch[i].aggregate_topk = 0
        batch[i].mesh_detail_preference = 0
        batch[i].alpha = 1.0
        batch[i].deadline_ms = 0
        batch[i].flags = 0
    out = ctypes.POINTER(_lib._pomai_search_results)()
    _check(_lib.pomai_search_batch(db, batch, n, ctypes.byref(out)))
    try:
        return [
            (
                [out[i].ids[j] for j in range(min(topk, out[i].count))],
                [out[i].scores[j] for j in range(min(topk, out[i].count))],
            )
            for i in range(n)
        ]
    finally:
        _lib.pomai_search_batch_free(out, n)


def create_rag_membrane(db, name, dim, shard_count=1):
    """Create and open a RAG membrane. Use it for put_chunk and search_rag."""
    _ensure_lib()
    _check(_lib.pomai_create_rag_membrane(db, name.encode("utf-8"), dim, shard_count))


def put_chunk(db, membrane_name, chunk_id, doc_id, token_ids, vector=None):
    """Insert a RAG chunk. token_ids: list of int (token IDs); vector: optional list of float (embedding)."""
    _ensure_lib()
    chunk = _lib._pomai_rag_chunk()
    chunk.struct_size = ctypes.sizeof(_lib._pomai_rag_chunk())
    chunk.chunk_id = int(chunk_id)
    chunk.doc_id = int(doc_id)
    tokens = (ctypes.c_uint32 * len(token_ids))(*token_ids)
    chunk.token_ids = tokens
    chunk.token_count = len(token_ids)
    if vector is not None and len(vector) > 0:
        vec = (ctypes.c_float * len(vector))(*vector)
        chunk.vector = vec
        chunk.dim = len(vector)
    else:
        chunk.vector = None
        chunk.dim = 0
    chunk.chunk_text = None
    chunk.chunk_text_len = 0
    _check(_lib.pomai_put_chunk(db, membrane_name.encode("utf-8"), ctypes.byref(chunk)))


def search_rag(db, membrane_name, token_ids=None, vector=None, topk=10, candidate_budget=200, enable_vector_rerank=True):
    """RAG search. Provide token_ids and/or vector. Returns list of (chunk_id, doc_id, score, token_matches)."""
    _ensure_lib()
    opts = _lib._pomai_rag_search_options()
    opts.struct_size = ctypes.sizeof(_lib._pomai_rag_search_options())
    opts.candidate_budget = candidate_budget
    opts.token_budget = 0
    opts.enable_vector_rerank = enable_vector_rerank

    query = _lib._pomai_rag_query()
    query.struct_size = ctypes.sizeof(_lib._pomai_rag_query())
    query.topk = topk
    if token_ids and len(token_ids) > 0:
        q_tokens = (ctypes.c_uint32 * len(token_ids))(*token_ids)
        query.token_ids = q_tokens
        query.token_count = len(token_ids)
    else:
        query.token_ids = None
        query.token_count = 0
    if vector and len(vector) > 0:
        q_vec = (ctypes.c_float * len(vector))(*vector)
        query.vector = q_vec
        query.dim = len(vector)
    else:
        query.vector = None
        query.dim = 0

    if (not token_ids or query.token_count == 0) and (not vector or query.dim == 0):
        raise ValueError("search_rag requires token_ids or vector")

    result = _lib._pomai_rag_search_result()
    _check(_lib.pomai_search_rag(db, membrane_name.encode("utf-8"), ctypes.byref(query), ctypes.byref(opts), ctypes.byref(result)))
    try:
        hits = []
        for i in range(result.hit_count):
            h = result.hits[i]
            chunk_text = None
            if h.chunk_text and h.chunk_text_len:
                chunk_text = ctypes.string_at(h.chunk_text, h.chunk_text_len).decode("utf-8", errors="replace")
            hits.append((h.chunk_id, h.doc_id, h.score, h.token_matches, chunk_text))
        return hits
    finally:
        _lib.pomai_rag_search_result_free(ctypes.byref(result))


def ingest_document(db, membrane_name, doc_id, text, *, max_chunk_bytes=512, max_doc_bytes=4 * 1024 * 1024):
    """Ingest a document into the RAG membrane: chunk, embed (mock), tokenize, store. Offline; no external API."""
    _ensure_lib()
    opts = _lib._pomai_rag_chunk_options()
    opts.struct_size = ctypes.sizeof(_lib._pomai_rag_chunk_options())
    opts.max_chunk_bytes = max_chunk_bytes
    opts.max_doc_bytes = max_doc_bytes
    opts.max_chunks_per_batch = 32
    opts.overlap_bytes = 0
    pipeline = ctypes.c_void_p()
    dim = 4  # match mock embedder; membrane dim must match
    _check(_lib.pomai_rag_pipeline_create(db, membrane_name.encode("utf-8"), dim, ctypes.byref(opts), ctypes.byref(pipeline)))
    try:
        text_buf = text.encode("utf-8")
        _check(_lib.pomai_rag_ingest_document(pipeline, int(doc_id), text_buf, len(text_buf)))
    finally:
        _lib.pomai_rag_pipeline_free(pipeline)


def retrieve_context(db, membrane_name, query, top_k=5, *, embedding_dim=4):
    """Retrieve context for a query: embed query, search RAG, return concatenated chunk text. Offline."""
    _ensure_lib()
    opts = _lib._pomai_rag_chunk_options()
    opts.struct_size = ctypes.sizeof(_lib._pomai_rag_chunk_options())
    opts.max_chunk_bytes = 512
    opts.max_doc_bytes = 4 * 1024 * 1024
    opts.max_chunks_per_batch = 32
    opts.overlap_bytes = 0
    pipeline = ctypes.c_void_p()
    _check(_lib.pomai_rag_pipeline_create(db, membrane_name.encode("utf-8"), embedding_dim, ctypes.byref(opts), ctypes.byref(pipeline)))
    try:
        query_buf = query.encode("utf-8")
        # Use caller-provided buffer to avoid cross-DLL free issues
        max_len = 65536
        out_buf = ctypes.create_string_buffer(max_len)
        out_len = ctypes.c_size_t()
        _check(_lib.pomai_rag_retrieve_context_buf(pipeline, query_buf, len(query_buf), top_k, out_buf, max_len, ctypes.byref(out_len)))
        if out_len.value == 0:
            return ""
        return out_buf.value[:out_len.value].decode("utf-8", errors="replace")
    finally:
        _lib.pomai_rag_pipeline_free(pipeline)


def membrane_kind_capabilities(kind: int):
    """Return capability flags for a membrane kind (0=vector … 12=meta). No DB handle required."""
    _ensure_lib()
    caps = _lib._pomai_membrane_capabilities()
    _lib.pomai_membrane_capabilities_init(ctypes.byref(caps))
    caps.struct_size = ctypes.sizeof(caps)
    _check(_lib.pomai_membrane_kind_capabilities(int(kind) & 0xFF, ctypes.byref(caps)))
    stab = "stable" if caps.stability == MEMBRANE_STABILITY_STABLE else "experimental"
    return {
        "kind": int(caps.kind),
        "stability": stab,
        "read_path": bool(caps.read_path),
        "write_path": bool(caps.write_path),
        "unified_scan": bool(caps.unified_scan),
        "snapshot_isolated_scan": bool(caps.snapshot_isolated_scan),
    }


from .zero_copy import release_zero_copy_session, search_zero_copy
