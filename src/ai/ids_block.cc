/*
 * src/ai/ids_block.cc
 *
 * Lightweight implementation file for IDs/offsets block helpers.
 *
 * This translation unit intentionally contains no global state. It provides a
 * small number of helper functions that operate on the on-disk/in-memory
 * uint64_t entries. The header `ids_block.h` is the primary developer-facing
 * API; this file exists to hold any future complex helpers without exposing
 * them in the header.
 *
 * The design keeps packing/unpacking extremely cheap (bit ops).
 */

#include "src/ai/ids_block.h"

namespace pomai::ai::soa
{
    // No runtime state required; current operations are inline bit manipulations
    // implemented in the header. This TU is a placeholder for potential helpers
    // (atomic-store wrappers, mapped-region conveniences) to keep the header minimal.
} // namespace pomai::ai::soa