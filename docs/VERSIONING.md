# Versioning and API/ABI Stability

## Current version

PomaiDB uses **semantic versioning** of the form `MAJOR.MINOR.PATCH`:

- **MAJOR**: Incompatible API or ABI changes.
- **MINOR**: New backward-compatible functionality.
- **PATCH**: Backward-compatible bug fixes and small improvements.

The project version is defined in the root `CMakeLists.txt` (`project(pomai VERSION ...)`) and can be read programmatically from the build.

## API stability promise

- **C++ API** (headers under `include/pomai/`, `pomai/pomai.h`, `pomai/options.h`, etc.):  
  We aim to keep the **public C++ API** backward-compatible within a **MAJOR** version. New MINOR/PATCH releases may add new types, functions, or options but will not remove or change the meaning of existing public APIs without a MAJOR bump.

- **C API** (headers in `include/pomai/`, C functions such as `pomai_open`, `pomai_put_batch`, `pomai_freeze`, `pomai_search_batch`, etc.):  
  The **C ABI** is stable within a MAJOR version: existing struct layouts and function signatures will not change. New functions or optional fields may be added in MINOR releases. Callers should check `struct_size` (or equivalent) when provided for forward compatibility.

- **Python package** (`pomaidb` on PyPI / `python/` in repo):  
  The high-level Python API (e.g. `pomaidb.open()`, `pomaidb.put_batch()`, `pomaidb.search_batch()`) follows the same MAJOR-version compatibility as the C API it wraps. Breaking changes require a MAJOR bump of the Python package.

## What we do not guarantee (yet)

- **Internal headers** and **source files** under `src/` are not part of the public API; they may change at any time.
- **Storage format** (WAL, segment, manifest layout) may evolve. We may provide migration tools for MINOR upgrades but do not promise backward-readable storage across MAJOR versions.
- **Build system** (CMake options, target names): we try to keep them stable but may add or rename options in MINOR releases.

## Deprecation policy

- Deprecated APIs will be marked in documentation and, when possible, with compiler attributes or comments. They will be removed no earlier than the next MAJOR release.
- We will document replacements and migration paths in release notes and in this doc when deprecations are introduced.

## Checking the version from build

```bash
# From CMake
grep "project(pomai" CMakeLists.txt
```

From C++: the version is not yet exported as preprocessor defines; you can rely on the tagged release or git describe. We may add `POMAI_VERSION_MAJOR`, `POMAI_VERSION_MINOR`, `POMAI_VERSION_PATCH` in a future release.

---

*This policy applies as of the first release that documents it. For pre-1.0 versions, we may still make breaking changes in MINOR versions with clear release notes.*
