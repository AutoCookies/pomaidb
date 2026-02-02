# Pomai Manifesto

## The Pomegranate
PomaiDB is a pomegranate:
- **Grain** = an immutable segment built from a frozen memtable.
- **Membrane** = the atomic, immutable view of all grains + live snapshot.
- **One atomic swap publishes a new world.** Readers never chase partial state.

## What Pomai Refuses to Be
- **Not an ML library.** No training, no pipelines, no model lifecycle.
- **Not a multi-tenant cloud.** Single machine, single user, local-first.
- **Not “best effort correctness.”** If the system can’t meet the contract, it says so.

## The Discipline
- **Correctness > cleverness.** Deterministic recovery is non-negotiable.
- **Bounded memory.** Always account for memory and fail explicitly when budgets are exceeded.
- **Deterministic recovery.** WAL + checkpoints are the source of truth.

## Engineering Values
- **SOT (Single Source of Truth).** One owner for each core contract and invariant.
- **Contracts are explicit.** Quality vs latency must be chosen, not guessed.
- **Immutability first.** Readers see immutable snapshots only.
- **Tests first.** Correctness comes before optimizations.
