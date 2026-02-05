# PomaiDB C API (Stable ABI)

This document defines the stable C ABI contract for embedding PomaiDB from any FFI-capable language.

## Design principles

- **C-only ABI**: all exported functions use `extern "C"` and plain C types.
- **Opaque handles**: `pomai_db_t`, `pomai_snapshot_t`, `pomai_iter_t` are opaque.
- **Explicit status model**: every API returns `pomai_status_t*`; `NULL` means success.
- **Single ownership rule**: caller owns inputs; Pomai owns returned result objects.
- **Low-copy hot paths**: put/search accept `pointer + len` buffers.

## Versioning

- ABI version is packed into `POMAI_C_ABI_VERSION` (`major<<16 | minor<<8 | patch`).
- Query at runtime via `pomai_abi_version()`.
- Engine version is returned by `pomai_version_string()`.

## Status model

Convention:
- `NULL` status => success.
- non-`NULL` status => failure, call `pomai_status_free()`.

Use:
- `pomai_status_code(st)` for machine handling.
- `pomai_status_message(st)` for diagnostics.

## Ownership and lifetime

- **Inputs (`pomai_upsert_t`, `pomai_query_t`)**: caller-owned; valid for call duration.
- **`pomai_record_t*` from `pomai_get()`**: Pomai-owned result; free with `pomai_record_free()`.
- **`pomai_search_results_t*` from `pomai_search()`**: free with `pomai_search_results_free()`.
- **Iterator row view (`pomai_record_view_t`)**: points into iterator-owned memory;
  valid only until `pomai_iter_next()` or `pomai_iter_free()`.
- **Snapshots/iterators**: release with dedicated free functions.

## Thread safety

| Handle | Thread-safe? | Notes |
|---|---|---|
| `pomai_db_t*` | Yes | Concurrent API calls are supported by underlying DB synchronization. |
| `pomai_snapshot_t*` | Yes | Immutable point-in-time view. |
| `pomai_iter_t*` | No | Single-threaded cursor; external synchronization required. |
| returned result objects | No | Treat as thread-confined unless caller synchronizes. |

## Search partial-failure policy

Pomai preserves shard-level partial failure semantics:
- search may still return best-effort hits,
- API returns `POMAI_STATUS_PARTIAL_FAILURE` while still setting `out` results.

## Deterministic scan ordering

For a fixed snapshot, iterator order is deterministic and stable for that snapshot. New writes after snapshot creation are excluded from that iterator.

## Example (C)

See:
- `examples/c_basic.c`
- `examples/c_scan_export.c`

## FFI notes

- **Python (`ctypes`)**: map opaque handles as `ctypes.c_void_p`; always free status/results.
- **Go (`cgo`)**: wrap status to Go `error`; use `unsafe.Slice` for results arrays.
- **Rust (`bindgen`)**: model handles as opaque enums; RAII-drop wrappers call free functions.

## ABI stability policy

- Existing symbols and struct field order are stable across ABI-compatible releases.
- Additive changes only in minor/patch versions.
- Breaking changes require ABI major bump.
