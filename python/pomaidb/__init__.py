"""
PomaiDB — embedded vector database for Edge AI.

Use the C library (libpomai_c.so / libpomai_c.dylib) via ctypes.
Set POMAI_C_LIB to the path to the shared library, or build from source
and point to build/libpomai_c.so (Linux) or build/libpomai_c.dylib (macOS).
"""

import ctypes
import os
from pathlib import Path

__all__ = [
    "open_db", "close", "put_batch", "freeze", "search_batch",
    "create_rag_membrane", "put_chunk", "search_rag",
    "PomaiDBError",
]

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

    class PomaiSearchResults(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("count", ctypes.c_size_t),
            ("ids", ctypes.POINTER(ctypes.c_uint64)),
            ("scores", ctypes.POINTER(ctypes.c_float)),
            ("shard_ids", ctypes.POINTER(ctypes.c_uint32)),
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
    lib.pomai_status_message.argtypes = [ctypes.c_void_p]
    lib.pomai_status_message.restype = ctypes.c_char_p
    lib.pomai_status_free.argtypes = [ctypes.c_void_p]
    lib.pomai_status_free.restype = None

    # RAG types
    class PomaiRagChunk(ctypes.Structure):
        _fields_ = [
            ("struct_size", ctypes.c_uint32),
            ("chunk_id", ctypes.c_uint64),
            ("doc_id", ctypes.c_uint64),
            ("token_ids", ctypes.POINTER(ctypes.c_uint32)),
            ("token_count", ctypes.c_size_t),
            ("vector", ctypes.POINTER(ctypes.c_float)),
            ("dim", ctypes.c_uint32),
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

    lib._pomai_options = PomaiOptions
    lib._pomai_upsert = PomaiUpsert
    lib._pomai_query = PomaiQuery
    lib._pomai_search_results = PomaiSearchResults
    lib._pomai_rag_chunk = PomaiRagChunk
    lib._pomai_rag_query = PomaiRagQuery
    lib._pomai_rag_search_options = PomaiRagSearchOptions
    lib._pomai_rag_hit = PomaiRagHit
    lib._pomai_rag_search_result = PomaiRagSearchResult


def _check(st):
    if st:
        _ensure_lib()
        msg = _lib.pomai_status_message(st).decode("utf-8", errors="replace")
        _lib.pomai_status_free(st)
        raise PomaiDBError(msg)


def open_db(path, dim, *, shards=1, search_threads=0, fsync=False, metric="ip", **hnsw_kw):
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
    db = ctypes.c_void_p()
    _check(_lib.pomai_open(ctypes.byref(opts), ctypes.byref(db)))
    return db


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
        batch[i].alpha = ctypes.c_float(0.0)
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
            hits.append((h.chunk_id, h.doc_id, h.score, h.token_matches))
        return hits
    finally:
        _lib.pomai_rag_search_result_free(ctypes.byref(result))
