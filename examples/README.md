# PomaiDB examples

Single-file samples for the **C++ API** (`pomai::DB`) and the **C ABI** (`libpomai_c`). Build the library once, then run any demo.

## Build (shared C ABI)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pomai_c
```

On macOS the shared library is `libpomai_c.dylib` instead of `.so`.

**ABI note:** `pomai_options_t`, `pomai_query_t`, and `pomai_search_results_t` are defined in `include/pomai/c_types.h`. If you use ctypes / ffi-napi / cgo, **field order and padding must match** the C headers (use `pomai_options_init()` for defaults). Python wrappers in `python/pomaidb/__init__.py` mirror these definitions.

---

## Python (`python/pomaidb` package — recommended)

From the repo root (after `cmake --build build --target pomai_c`):

```bash
export POMAI_C_LIB="$PWD/build/libpomai_c.so"
python3 -c "import sys; sys.path.insert(0,'python'); import pomaidb; print('ok')"
```

**RAG quickstart:** `examples/rag_quickstart.py` (uses `ingest_document` / `retrieve_context`).

---

## Python (ctypes, standalone)

**File:** `examples/python_basic.py`

```bash
POMAI_C_LIB=./build/libpomai_c.so python3 examples/python_basic.py
```

---

## Python (zero-copy search flag)

**File:** `examples/python_zero_copy_demo.py` — optional `POMAI_QUERY_FLAG_ZERO_COPY` path; requires `numpy`.

```bash
POMAI_C_LIB=./build/libpomai_c.so python3 examples/python_zero_copy_demo.py
```

---

## JavaScript (Node + ffi-napi)

**File:** `examples/js_basic.mjs`

`ffi-napi` uses native addons; use an **LTS Node** if builds fail on very new releases.

```bash
npm install ffi-napi ref-napi ref-struct-di
POMAI_C_LIB=./build/libpomai_c.so node examples/js_basic.mjs
```

---

## TypeScript (Node + ts-node)

**File:** `examples/ts_basic.ts`

```bash
npm install ffi-napi ref-napi ref-struct-di ts-node typescript @types/node
POMAI_C_LIB=./build/libpomai_c.so npx ts-node --compilerOptions '{"module":"commonjs"}' examples/ts_basic.ts
```

---

## Go (cgo)

**File:** `examples/go_basic.go`

Uses `runtime.Pinner` (Go 1.21+) so vector pointers are safe to pass through cgo.

```bash
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
go run examples/go_basic.go
```

---

## C (C API)

**File:** `examples/c_basic.c`

```bash
cc -std=c11 -I./include examples/c_basic.c -L./build -lpomai_c -o /tmp/pomai_c_basic
LD_LIBRARY_PATH=$PWD/build /tmp/pomai_c_basic
```

---

## C++ (embedded `pomai::DB` API)

**File:** `examples/cpp_basic.cpp`

```bash
c++ -std=c++20 -I./include examples/cpp_basic.cpp -L./build -lpomai -lpthread -lcrypto -ldl -o /tmp/pomai_cpp_basic
LD_LIBRARY_PATH=$PWD/build /tmp/pomai_cpp_basic
```

Removes the need for `libpomai_c`; links against `libpomai.a` / `libpomai.so` (see `build/` for your platform).

**Optional:** `examples/pomai_multimodal_demo.cpp` — `Put` with metadata, `SearchMultiModal`, filters.

---

## Utilities

| File | Purpose |
|------|---------|
| `python_ctypes_demo.py` | Print `pomai_abi_version()` / `pomai_version_string()` |
| `c_scan_export.c` | Snapshot + iterator JSON export (empty DB yields `[]`) |

---

## Environment

- Set **`POMAI_C_LIB`** to the full path of `libpomai_c` if it is not under `./build/`.
- Default vector membrane name in the C API is **`__default__`** (C++ `Freeze("__default__")` / membrane APIs).
