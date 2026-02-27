#pragma once

#include <vector>
#include <string>
#include <memory>
#include "pomai/status.h"

namespace pomai::table {
    class SegmentReader;
}

namespace pomai::storage {

/**
 * CompactionTask: Describes which segments to merge and where the result goes.
 */
struct CompactionTask {
    int input_level;
    int output_level;
    std::vector<std::string> input_segments;
    bool valid = false;
};

/**
 * CompactionManager: Implements Leveled Compaction Lite.
 * Focuses on minimizing WAF for SD cards while maintaining search performance.
 */
class CompactionManager {
public:
    static constexpr int kMaxLevels = 4;
    
    struct LevelStats {
        int level;
        uint64_t total_size;
        int file_count;
        double score;
    };

    CompactionManager(uint64_t l1_base_size_bytes = 64 * 1024 * 1024); // Default L1: 64MB

    /**
     * Pick a compaction task based on level scores.
     * Prioritizes L0 (flushed memtables) to reduce read amplification.
     */
    CompactionTask PickCompaction(const std::vector<LevelStats>& stats);

    /**
     * Calculate target size for a given level.
     * Level N target = L1_Base * 10^(N-1)
     */
    uint64_t GetTargetSize(int level) const;

private:
    uint64_t l1_base_size_;
};

} // namespace pomai::storage
