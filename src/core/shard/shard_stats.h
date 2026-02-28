// shard_stats.h — Per-shard observability for edge tuning.
//
// Phase 4: Lightweight stats snapshot from a ShardRuntime.
// All fields are lock-free atomic reads — safe to call from any thread.
//
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once
#include <cstdint>

namespace pomai::core {

struct ShardStats {
    std::uint32_t shard_id{0};
    std::uint64_t ops_processed{0};       // Total commands processed by the worker
    std::uint64_t queue_depth{0};         // Current pending commands in mailbox
    std::uint64_t candidates_scanned{0};  // Candidates scanned in last query
    std::uint64_t memtable_entries{0};    // Current active MemTable size
    std::uint64_t palloc_mem_committed{0}; // Shard-local heap committed bytes
    std::uint64_t palloc_mem_used{0};      // Shard-local heap used bytes
};

} // namespace pomai::core
