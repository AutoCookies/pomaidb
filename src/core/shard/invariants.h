#pragma once

namespace pomai::core::invariants
{
    // =========================================================================
    // PomaiDB System Invariants
    // =========================================================================

    // INVARIANT 1: Snapshot Immutability
    // All data structures in ShardSnapshot are immutable after publication.
    // Frozen memtables and segments must NOT be modified after being added to snapshot.
    // Enforced by: std::shared_ptr const access + ShardRuntime assertions.

    // INVARIANT 2: Snapshot Ordering
    // Snapshots represent a prefix of WAL history.
    // Snapshot N contains all operations committed before Freeze N.
    // Version number is monotonic.

    // INVARIANT 3: Delete Visibility
    // Deletes become visible after soft freeze.
    // A delete in active memtable shadows values in frozen/segments ONLY after rotation.
    // Until rotation, reads (Search/Get/Exists) observe the old value (Bounded Staleness).

    // INVARIANT 4: Memory Safety
    // Snapshots hold shared ownership via shared_ptr.
    // Memory reclaimed automatically when readers release snapshot.
    // No reader ever observes partially constructed state.

    // INVARIANT 5: Search Consistency
    // Search MUST use the same snapshot for candidate generation and scoring
    // (currently simplified to brute-force on snapshot to guarantee this).

} // namespace pomai::core::invariants
