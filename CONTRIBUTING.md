# Contributing to PomaiDB

## Development setup

### Build
```bash
cmake -S . -B build -DPOMAI_BUILD_TESTS=ON
cmake --build build -j
```

### Tests
```bash
ctest --test-dir build --output-on-failure
```

### Labels
You can target specific labels (unit/integ/crash/tsan/recall):
```bash
ctest --test-dir build -L unit
```

## Code structure overview
- `include/`: Public API headers. (Source: `include/pomai/pomai.h`.)
- `src/api/`: API implementation. (Source: `src/api/db.cc`.)
- `src/core/`: Engine, shard runtime, mailbox, and indexing. (Source: `src/core/...`.)
- `src/storage/`: WAL and manifest logic. (Source: `src/storage/...`.)
- `src/table/`: MemTable and segment format. (Source: `src/table/...`.)
- `tests/`: Unit, integration, crash, recall, and TSAN tests.

## Pull request requirements
- **Build must pass** with `POMAI_BUILD_TESTS=ON`.
- **Tests must pass** for the areas affected.
- **Docs updated** when behavior changes; refer to `docs/` as the single source of truth.
- **No behavioral claims** without a code pointer or test reference.

## Commit message conventions
- Format: `<area>: <summary>`
- Examples:
  - `docs: define consistency model`
  - `storage: fix WAL replay edge case`

## Style rules
- C++20, warnings enabled (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`). (Source: `CMakeLists.txt`.)
- Keep public API in `include/pomai/` minimal and stable.

## Security and responsible disclosure
- Report security issues privately to the maintainers (see `GOVERNANCE.md`).
- Do not open public issues for vulnerabilities before a fix is ready.
