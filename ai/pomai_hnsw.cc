// ai/pomai_hnsw.cc
//
// Minimal PPHNSW glue-layer implementation.
// - Wraps upstream HierarchicalNSW (hnswlib) with PomaiSpace so every stored
//   vector has a PPEHeader preceding it.
// - Implements a thin wrapper addPoint that constructs the header + vector
//   buffer and delegates to HierarchicalNSW::addPoint.
// - Implements searchKnnAdaptive which temporarily adjusts ef_ based on panic_factor.
//
// Notes:
// - This file purposely keeps logic small and conservative: full advanced
//   predictive eviction / mmap demotion / quantization belong in later stages.

#include "pomai_hnsw.h"
#include "ppe.h"
#include "pomai_space.h"

#include <cstring>
#include <cstdlib>
#include <cassert>

namespace pomai::ai
{

    // The PPHNSW methods were declared templated in pomai_hnsw.h.
    // Most logic is inline there. We provide here any out-of-line helpers if needed.
    // Currently the header-only PPHNSW implementation is sufficient.

    // However we'll provide an explicit instantiation for float to help linkers in some build setups.
    template class PPHNSW<float>;

} // namespace pomai::ai