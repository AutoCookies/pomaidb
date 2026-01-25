// Added minimal, safe, high-performance implementation of PomaiOrbit::apply_thermal_policy()
// Purpose:
//  - Provide a light-weight background decay of centroid "temperature" based on last access.
//  - Keep CPU & memory costs small; safe to call frequently (e.g., every 20ms from PomaiDB background worker).
//  - This fixes the missing symbol link error and provides deterministic, conservative thermal decay.
//
// Design notes:
//  - Use atomic loads/stores only. No locks.
//  - Decay is coarse-grained: subtract one temperature unit per minute of inactivity.
//    This is intentionally conservative and deterministic.
//  - No heavy work (no scans, no allocations).
//  - Implementation respects production constraints: predictable latency, O(num_centroids).
//
// Cost model:
//  - O(num_centroids) per call. For typical centroid counts (hundreds -> a few thousands) this is cheap.
//  - Called from low-frequency background thread; avoids impacting latency-sensitive paths.

#include "src/ai/pomai_orbit.h"
#include <ctime>
#include <cstdint>
#include <algorithm>

namespace pomai::ai::orbit
{

    void PomaiOrbit::apply_thermal_policy()
    {
        // Fast-path: nothing to do.
        if (thermal_map_.empty() || last_access_epoch_.empty())
            return;

        // Use UTC epoch seconds for inexpensive checks.
        const uint32_t now = static_cast<uint32_t>(std::time(nullptr));

        // Decay interval controls how often a single temperature step is applied.
        // Conservative default: 60s. This is intentionally coarse to minimize maintenance cost.
        constexpr uint32_t DECAY_INTERVAL_S = 60u;

        // Iterate linearly over the centroids; each iteration does only a few atomic ops.
        // This keeps the maintenance O(num_centroids) and cheap enough to run frequently.
        size_t n = thermal_map_.size();
        for (size_t i = 0; i < n; ++i)
        {
            // Load last access time (relaxed is fine for heuristic maintenance)
            uint32_t last = last_access_epoch_[i].load(std::memory_order_relaxed);

            // If last access is recent, skip.
            if (now <= last || (now - last) < DECAY_INTERVAL_S)
                continue;

            // Try to reduce temperature by 1 (clamped to 0).
            uint8_t cur = thermal_map_[i].load(std::memory_order_relaxed);
            if (cur > 0)
            {
                // Write relaxed; exact ordering not required for this heuristic.
                thermal_map_[i].store(static_cast<uint8_t>(cur - 1), std::memory_order_relaxed);
            }

            // Advance last_access_epoch_ so we won't repeatedly decay the same slot in a tight loop.
            // Store now (relaxed) to mark we've processed this centroid for this interval.
            last_access_epoch_[i].store(now, std::memory_order_relaxed);
        }
    }

} // namespace pomai::ai::orbit