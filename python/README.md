# pomaidb — Python package

PomaiDB Python bindings (ctypes). Requires the C library `libpomai_c.so` (Linux) or `libpomai_c.dylib` (macOS).

## Install

From the repo root (after building the C library):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target pomai_c
pip install ./python
```

Set `POMAI_C_LIB` to the path to the shared library if it is not in `./build/`:

```bash
export POMAI_C_LIB=/path/to/build/libpomai_c.so
pip install ./python
```

## Usage

```python
import pomaidb

db = pomaidb.open_db("/tmp/my_db", dim=128, shards=1)
pomaidb.put_batch(db, ids=[1, 2, 3], vectors=[[0.1] * 128, [0.2] * 128, [0.3] * 128])
pomaidb.freeze(db)
results = pomaidb.search_batch(db, queries=[[0.15] * 128], topk=5)
pomaidb.close(db)
```

See [docs/PYTHON_API.md](../docs/PYTHON_API.md) for full API and ctypes details.
