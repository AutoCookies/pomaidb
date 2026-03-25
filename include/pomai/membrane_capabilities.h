#pragma once

#include <cstdint>

#include "options.h"

namespace pomai {

/**
 * Product / support tier for a membrane kind (edge release vs preview APIs).
 * Mirrors docs/EDGE_RELEASE.md — keep in sync when changing assignments.
 */
enum class MembraneStability : uint8_t {
    kStable = 0,
    kExperimental = 1,
};

/**
 * Capability flags for one MembraneKind (embedded multi-modal surface).
 *
 * - read_path: typed point reads / search APIs exist for this kind.
 * - write_path: typed mutation APIs exist (put / ingest / append / etc.).
 * - unified_scan: full export via NewMembraneRecordIterator / pomai_membrane_scan.
 * - snapshot_isolated_scan: when unified_scan is true, scan uses the same snapshot
 *   isolation as vector iterators (point-in-time). Only kVector; all other kinds read
 *   live engine state during scan — see membrane_iterator.h.
 */
struct MembraneKindCapabilities {
    MembraneKind kind = MembraneKind::kVector;
    bool read_path = false;
    bool write_path = false;
    bool unified_scan = false;
    bool snapshot_isolated_scan = false;
    MembraneStability stability = MembraneStability::kStable;
};

/** Short stable name for logs and JSON (ASCII, no spaces). */
constexpr const char* MembraneKindApiName(MembraneKind kind) noexcept {
    switch (kind) {
    case MembraneKind::kVector: return "vector";
    case MembraneKind::kRag: return "rag";
    case MembraneKind::kGraph: return "graph";
    case MembraneKind::kText: return "text";
    case MembraneKind::kTimeSeries: return "timeseries";
    case MembraneKind::kKeyValue: return "keyvalue";
    case MembraneKind::kSketch: return "sketch";
    case MembraneKind::kBlob: return "blob";
    case MembraneKind::kSpatial: return "spatial";
    case MembraneKind::kMesh: return "mesh";
    case MembraneKind::kSparse: return "sparse";
    case MembraneKind::kBitset: return "bitset";
    case MembraneKind::kMeta: return "meta";
    }
    return "unknown";
}

constexpr bool IsValidMembraneKindValue(uint8_t v) noexcept { return v <= static_cast<uint8_t>(MembraneKind::kMeta); }

constexpr MembraneKindCapabilities GetMembraneKindCapabilities(MembraneKind kind) noexcept {
    switch (kind) {
    case MembraneKind::kVector:
        return {MembraneKind::kVector, true, true, true, true, MembraneStability::kStable};
    case MembraneKind::kRag:
        return {MembraneKind::kRag, true, true, true, false, MembraneStability::kStable};
    case MembraneKind::kGraph:
        return {MembraneKind::kGraph, true, true, true, false, MembraneStability::kStable};
    case MembraneKind::kText:
        return {MembraneKind::kText, true, true, true, false, MembraneStability::kStable};
    case MembraneKind::kTimeSeries:
        return {MembraneKind::kTimeSeries, true, true, true, false, MembraneStability::kStable};
    case MembraneKind::kKeyValue:
        return {MembraneKind::kKeyValue, true, true, true, false, MembraneStability::kStable};
    case MembraneKind::kSketch:
        return {MembraneKind::kSketch, true, true, true, false, MembraneStability::kStable};
    case MembraneKind::kBlob:
        return {MembraneKind::kBlob, true, true, true, false, MembraneStability::kStable};
    case MembraneKind::kSpatial:
        return {MembraneKind::kSpatial, true, true, true, false, MembraneStability::kExperimental};
    case MembraneKind::kMesh:
        return {MembraneKind::kMesh, true, true, true, false, MembraneStability::kExperimental};
    case MembraneKind::kSparse:
        return {MembraneKind::kSparse, true, true, true, false, MembraneStability::kExperimental};
    case MembraneKind::kBitset:
        return {MembraneKind::kBitset, true, true, true, false, MembraneStability::kExperimental};
    case MembraneKind::kMeta:
        return {MembraneKind::kMeta, true, true, true, false, MembraneStability::kStable};
    default:
        return {};
    }
}

} // namespace pomai
