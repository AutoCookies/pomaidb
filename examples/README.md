# PomaiDB Examples

Each example is a **single file** and uses the embedded PomaiDB API (C++ or C ABI). Build the library once and then run any language demo.

## Build (shared C ABI)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pomai_c
```

> Note: On macOS the shared library is `libpomai_c.dylib` instead of `.so`.

---

## Python (ctypes)

**File:** `examples/python_basic.py`

```bash
POMAI_C_LIB=./build/libpomai_c.so python3 examples/python_basic.py
```

Expected output (example):

```
TopK results:
  id=0 score=0.9312
  id=17 score=0.9134
```

---

## JavaScript (Node + ffi-napi)

**File:** `examples/js_basic.mjs`

```bash
npm install ffi-napi ref-napi ref-struct-di
POMAI_C_LIB=./build/libpomai_c.so node examples/js_basic.mjs
```

---

## TypeScript (Node + ts-node)

**File:** `examples/ts_basic.ts`

```bash
npm install ffi-napi ref-napi ref-struct-di ts-node typescript
POMAI_C_LIB=./build/libpomai_c.so npx ts-node --compiler-options '{"module":"commonjs"}' examples/ts_basic.ts
```

---

## Go (cgo)

**File:** `examples/go_basic.go`

```bash
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
go run examples/go_basic.go
```

---

## C++ (embedded API)

**File:** `examples/cpp_basic.cpp`

```bash
c++ -std=c++20 -I./include examples/cpp_basic.cpp -L./build -lpomai -lpthread -o /tmp/pomai_cpp_basic
/tmp/pomai_cpp_basic
```

---

### Notes

- Each demo opens a local DB path under `/tmp`, inserts vectors, runs a search, and prints top-K results.
- If your shared library lives elsewhere, set `POMAI_C_LIB` to the full path.
