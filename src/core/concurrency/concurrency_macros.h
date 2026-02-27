// concurrency_macros.h — Hardware-sympathetic alignment & padding.
// Inspired by DragonflyDB & ScyllaDB core engine.
// Copyright 2026 PomaiDB authors. MIT License.

#pragma once

#include <cstddef>

/**
 * Standard L1 cache line size on most modern x86/ARM hardware is 64 bytes.
 * Aligning hot-path structures to this boundary prevents "False Sharing"
 * where multiple cores fight for ownership of the same cache line.
 */
#ifndef POMAI_CACHE_LINE_SIZE
#define POMAI_CACHE_LINE_SIZE 64
#endif

// Strictly align a structure or member to a cache line.
#define POMAI_CACHE_ALIGNED alignas(POMAI_CACHE_LINE_SIZE)

/**
 * POMAI_PAD: Explicitly pad a structure to a multiple of cache lines.
 * Useful for sharded state to ensure neighboring shards don't overlap in L1/L2.
 */
#define POMAI_PAD_TO_CACHELINE(current_size) \
    char _pomai_pad[POMAI_CACHE_LINE_SIZE - ((current_size) % POMAI_CACHE_LINE_SIZE)]

/**
 * POMAI_HOT: Mark a function for aggressive inlining and optimization.
 */
#if defined(__GNUC__) || defined(__clang__)
#define POMAI_HOT [[gnu::hot]] inline
#else
#define POMAI_HOT inline
#endif
