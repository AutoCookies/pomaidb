# Python API

PomaiDB is exposed to Python via the **C API** and **ctypes**. The official package is **`pomaidb`** (pip-installable from the `python/` directory). You can also use the C library directly with ctypes (see examples in `examples/` and `benchmarks/`).

## Installation

1. Build the C library (from repo root):
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target pomai_c
   ```
2. Install the Python package:
   ```bash
   pip install ./python
   ```
3. Set `POMAI_C_LIB` to the path of `libpomai_c.so` (Linux) or `libpomai_c.dylib` (macOS) if the package cannot find it (e.g. `export POMAI_C_LIB=$PWD/build/libpomai_c.so`).

## High-level API (pomaidb package)

| Function        | Description |
|----------------|-------------|
| `open_db(path, dim, **opts)` | Open database at `path` with vector dimension `dim`. Options: `shards`, `search_threads`, `fsync`, `metric` ("ip" or "l2"), `hnsw_m`, `hnsw_ef_construction`, `hnsw_ef_search`, `adaptive_threshold`. Returns opaque db handle. |
| `put_batch(db, ids, vectors)` | Insert vectors. `ids`: list of int; `vectors`: list of list of float (n × dim). |
| `freeze(db)`   | Flush memtable to segment and build index. Must be called before new data is visible to search. |
| `search_batch(db, queries, topk=10)` | Batch search. `queries`: list of list of float (n_queries × dim). Returns list of `(ids, scores)` per query. |
| `close(db)`    | Close the database. |

Exceptions: `pomaidb.PomaiDBError` on any failing call.

### Example

```python
import pomaidb

db = pomaidb.open_db("/tmp/vec_db", dim=128, shards=1, metric="ip")
pomaidb.put_batch(db, ids=[1, 2], vectors=[[0.1] * 128, [0.2] * 128])
pomaidb.freeze(db)
for ids, scores in pomaidb.search_batch(db, [[0.15] * 128], topk=5):
    print(ids, scores)
pomaidb.close(db)
```

## C API (ctypes)

The shared library exposes:

- `pomai_options_init(opts)` — initialize options struct
- `pomai_open(opts, &db)` — open DB; returns status (null = ok)
- `pomai_put_batch(db, upserts, n)` — batch insert
- `pomai_freeze(db)` — flush and build index
- `pomai_search_batch(db, queries, n, &out)` — batch search; `out` is array of `PomaiSearchResults`
- `pomai_search_batch_free(out, n)` — free search results
- `pomai_close(db)` — close DB
- `pomai_status_message(status)` / `pomai_status_free(status)` — error message

Struct layouts and constants are in `include/pomai/c_types.h`. The `pomaidb` package registers these for you.

## Versioning

The Python package follows the same version as the project (see [VERSIONING.md](VERSIONING.md)). Compatibility is maintained with the C ABI within a MAJOR version.
