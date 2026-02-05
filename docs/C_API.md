# PomaiDB C API (Stable ABI)

This document defines the stable C ABI contract for embedding PomaiDB from any FFI-capable language.

## Design principles

- **C-only ABI**: all exported functions use `extern "C"` and plain C types.
- **Cross-platform exports**: all public symbols are declared with `POMAI_API`.
- **Opaque handles**: `pomai_db_t`, `pomai_snapshot_t`, `pomai_iter_t` are opaque.
- **Explicit status model**: every API returns `pomai_status_t*`; `NULL` means success.
- **Single ownership rule**: caller owns inputs; Pomai owns returned result objects.
- **Low-copy hot paths**: put/search accept `pointer + len` buffers.

## ABI versioning

Macros:
- `POMAI_ABI_VERSION_MAJOR`
- `POMAI_ABI_VERSION_MINOR`
- `POMAI_ABI_VERSION_PATCH`

Runtime query:
- `pomai_abi_version()` returns `(major << 16) | (minor << 8) | patch`.

Rules:
- **MAJOR**: may break ABI.
- **MINOR**: additive only (new symbols/functions allowed; existing signatures/struct layout must remain compatible).
- **PATCH**: must not change ABI.

## Forward-compatible public structs

All public structs that cross the ABI boundary include:
- `uint32_t struct_size` as **field #1**,
- new fields appended at the end only,
- runtime validation in API entrypoints (`struct_size >= minimum expected size`).

This permits newer clients/libraries to negotiate struct growth safely.

## Status model

Convention:
- `NULL` status => success.
- non-`NULL` status => failure, call `pomai_status_free()`.

Use:
- `pomai_status_code(st)` for machine handling.
- `pomai_status_message(st)` for diagnostics.

Ownership:
- status message memory is owned by the `pomai_status_t` object and stays valid until `pomai_status_free()`.

## Ownership and lifetime

- **Inputs (`pomai_upsert_t`, `pomai_query_t`, `pomai_scan_options_t`)**: caller-owned; valid for call duration.
- **`pomai_record_t*` from `pomai_get()`**: **owned** record object; stable until `pomai_record_free()`.
- **`pomai_search_results_t*` from `pomai_search()`**: buffers are stable until `pomai_search_results_free()`.
- **Iterator row view (`pomai_record_view_t`)**: view pointers are valid **only** until the next `pomai_iter_next()` or `pomai_iter_free()`.
- **Snapshots/iterators**: release with dedicated free functions.

## Deadline contract

`deadline_ms` fields are optional on options/query/scan structs:
- `0` => no deadline.
- non-zero => Unix epoch milliseconds deadline.
- API returns `POMAI_STATUS_DEADLINE_EXCEEDED` deterministically when `now >= deadline_ms` at documented checks.

## Thread safety contract

### Handle-level

| Handle | Thread-safe? | Notes |
|---|---|---|
| `pomai_db_t*` | Yes | Safe for concurrent reads; writes are internally serialized per shard. |
| `pomai_snapshot_t*` | Yes | Immutable point-in-time view. |
| `pomai_iter_t*` | No | Iterator handles are not thread-safe. |
| `pomai_record_t*` (owned get result) | No | Caller may share with external synchronization only. |
| `pomai_search_results_t*` | No | Caller may share with external synchronization only. |
| `pomai_status_t*` | No | Caller may share with external synchronization only. |

### Function-level

| Function | Thread-safe when called concurrently? | Notes |
|---|---|---|
| `pomai_options_init`, `pomai_scan_options_init` | Yes | Distinct caller-owned structs. |
| `pomai_open`, `pomai_close` | Conditionally | Do not race `pomai_close` with other calls on same DB handle. |
| `pomai_put`, `pomai_put_batch`, `pomai_delete` | Yes | Writes serialized internally per shard. |
| `pomai_get`, `pomai_exists`, `pomai_search`, `pomai_get_snapshot` | Yes | Safe against concurrent reads/writes under DB internal synchronization. |
| `pomai_scan` | Yes | Creating iterators is thread-safe; each iterator remains thread-confined. |
| `pomai_iter_valid`, `pomai_iter_next`, `pomai_iter_status`, `pomai_iter_get_record` | No | Same iterator must not be used concurrently. |
| `pomai_iter_free`, `pomai_snapshot_free`, `pomai_record_free`, `pomai_search_results_free`, `pomai_status_free` | Conditionally | Must not race with other use of the same object. |
| `pomai_status_code`, `pomai_status_message`, `pomai_version_string`, `pomai_abi_version` | Yes | Read-only operations. |

## Search partial-failure policy

Pomai preserves shard-level partial failure semantics:
- search may still return best-effort hits,
- API returns `POMAI_STATUS_PARTIAL_FAILURE` while still setting `out` results.

## Deterministic scan ordering

For a fixed snapshot, iterator order is deterministic and stable for that snapshot. New writes after snapshot creation are excluded from that iterator.

## Windows build/export support

The C API shared library builds as a `.dll` on Windows and exports the same symbols through `POMAI_API`.

### Minimal Windows build commands

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DPOMAI_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Example (C)

See:
- `examples/c_basic.c`
- `examples/c_scan_export.c`

## FFI notes

- **Python (`ctypes`)**: map opaque handles as `ctypes.c_void_p`; always free status/results.
- **Go (`cgo`)**: wrap status to Go `error`; use `unsafe.Slice` for results arrays.
- **Rust (`bindgen`)**: model handles as opaque enums; RAII-drop wrappers call free functions.
