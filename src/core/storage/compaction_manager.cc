#include "core/storage/compaction_manager.h"
#include <algorithm>
#include <cmath>

namespace pomai::storage {

CompactionManager::CompactionManager(uint64_t l1_base_size_bytes)
    : l1_base_size_(l1_base_size_bytes) {}

uint64_t CompactionManager::GetTargetSize(int level) const {
    if (level <= 0) return 0;
    // Level N target = Base * 10^(N-1)
    return l1_base_size_ * static_cast<uint64_t>(std::pow(10, level - 1));
}

CompactionTask CompactionManager::PickCompaction(const std::vector<LevelStats>& stats) {
    CompactionTask task;
    
    // Sort levels by score (descending) to find the most "distressed" level
    auto sorted_stats = stats;
    std::sort(sorted_stats.begin(), sorted_stats.end(), [](const LevelStats& a, const LevelStats& b) {
        return a.score > b.score;
    });

    for (const auto& s : sorted_stats) {
        if (s.score <= 1.0) continue;

        // Level-0 is a special case: many files trigger compaction to L1
        if (s.level == 0) {
            task.input_level = 0;
            task.output_level = 1;
            task.valid = true;
            return task;
        }

        // Leveled compaction: Pick Level N to merge into Level N+1
        if (s.level < kMaxLevels - 1) {
            task.input_level = s.level;
            task.output_level = s.level + 1;
            task.valid = true;
            return task;
        }
    }

    return task;
}

} // namespace pomai::storage
