# Release Checklist (v1.0.0 Readiness)

## Gate A: Behavior Contract
- [ ] **Consistency Model**: Code matches `docs/CONSISTENCY_MODEL.md`.
     - [ ] Writes visible only after Freeze/Soft-Freeze.
     - [ ] Snapshot isolation checked.
- [ ] **API Semantics**: Unique IDs enforced, membranes isolate data.

## Gate B: Crash Safety
- [ ] **Crash Harness**: `pomai_crash_replay` passes 50/50 rounds.
- [ ] **Persistence**: Named membranes survive restart.
- [ ] **Atomic Updates**: Global manifest and Shard manifest updates use `fsync` + `rename` + `dir-fsync`.

## Gate C: On-Disk Format
- [ ] **Versioning**: All new files write valid headers (`pomai.manifest.v3`, `pomai.seg.v1`).
- [ ] **Inspection**: `pomai_inspect` can read generated artifacts.

## Gate D: Testing Hardness
- [ ] **Unit Tests**: All unit tests pass (`ctest -L unit`).
- [ ] **Sanitizers**: ASAN and TSAN builds are clean.
- [ ] **Valgrind/Leak (Optional)**: No massive leaks on shutdown.

## Gate E: Operations
- [ ] **Runbook**: `docs/RUNBOOK.md` is accurate and tested.
- [ ] **Observability**: (Partial) `pomai_inspect` available for post-mortem.

## Release Steps
1. Update `POMAI_VERSION` in `include/pomai/version.h`.
2. Update `CHANGELOG.md` with features and breaking changes.
3. Tag commit: `git tag -a v1.0.0 -m "Release v1.0.0"`.
4. Build release binaries and archive.
