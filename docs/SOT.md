# Single Source of Truth (SOT)

PomaiDB defines a single owner for each core contract. When behavior changes, update the owner and its documentation, not multiple call sites.

## SOT map

### Active membrane view
- **Owner:** `Shard` publishes `ShardState` via atomic swap.
- **Readers:** `MembraneRouter` and search paths only read shared immutable snapshots.
- **Why:** guarantees lock-free reads and deterministic views.

### Query contract (QUALITY vs LATENCY)
- **Owner:** `MembraneRouter::Search`.
- **Rule:** Only this layer decides how `SearchMode::Quality` or `SearchMode::Latency` affects results and failures.

### Budget enforcement
- **Owner:** `MembraneRouter` computes budgets via `WhisperGrain` and normalizes filter budgets.
- **Enforcement:** `Shard::Search` / `Seed::SearchSnapshot` enforce budget fields (visits/time) but do not redefine budgets.

### Filter semantics
- **Owner:** `Seed::Store::MatchFilter` defines match semantics.
- **Normalization:** `core/filter.h::NormalizeFilter` canonicalizes filters and enforces tag limits for Search/Scan.

## Practical guidance
- Do not re-normalize search parameters outside `core/search_contract.h`.
- Do not add filter normalization in multiple places; call `NormalizeFilter` once and propagate the result.
- New error paths must return `Status`/`Result<T>` or log with a clear event name.
