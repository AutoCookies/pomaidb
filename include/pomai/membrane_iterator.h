#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "options.h"
#include "status.h"
#include "types.h"

namespace pomai {

/**
 * Controls full-scan behaviour for NewMembraneRecordIterator (production guardrails).
 * - max_records: stop after this many rows (0 = unlimited).
 * - max_materialized_keys: for key/value–backed membranes, max distinct keys loaded into memory
 *   for sorting (0 = unlimited). Exceeding returns ResourceExhausted at iterator creation.
 * - deadline_ms: wall-clock budget from iterator creation (0 = none); checked during
 *   materialization and periodically while advancing vector scans.
 * - max_field_bytes: soft cap for key/value text fields (truncated with "..." suffix).
 */
struct MembraneScanOptions {
    uint64_t max_records = 1'000'000;
    uint64_t max_materialized_keys = 5'000'000;
    uint32_t deadline_ms = 0;
    size_t max_field_bytes = 4 * 1024 * 1024;
};

/**
 * One row from a full-scan over a single membrane, regardless of MembraneKind.
 * See class comment on MembraneRecordIterator for semantics and isolation guarantees.
 */
struct MembraneRecord {
    MembraneKind kind = MembraneKind::kVector;
    uint64_t id = 0;
    std::string key;
    std::string value;
    std::vector<float> vector;
};

/**
 * Unified iterator: walk primary records in a membrane.
 *
 * Isolation:
 * - kVector: uses the same snapshot iterator as NewIterator (point-in-time, tombstone filtered).
 * - Other kinds: read from current in-memory engine state (not snapshot-isolated); suitable
 *   for export/ops, not for strict transactional read-your-writes across concurrent mutations.
 *
 * Call Valid() before Record(). After the scan completes, check ScanStatus() for errors and
 * Truncated() if max_records cut the result short.
 */
class MembraneRecordIterator {
public:
    virtual ~MembraneRecordIterator() = default;
    virtual bool Valid() const = 0;
    virtual void Next() = 0;
    virtual const MembraneRecord& Record() const = 0;
    /** Iterator-level error (e.g. deadline during vector scan). Ok when scan completed cleanly. */
    virtual Status ScanStatus() const { return Status::Ok(); }
    /** True if more rows existed but were not returned due to max_records. */
    virtual bool Truncated() const { return false; }
};

} // namespace pomai
