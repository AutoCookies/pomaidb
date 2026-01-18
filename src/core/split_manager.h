#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <random>
#include "src/core/config.h"

namespace pomai::core {

    class SplitManager {
    public:
        explicit SplitManager(const pomai::config::PomaiConfig& cfg);
        ~SplitManager() = default;

        std::vector<uint64_t> train_indices;
        std::vector<uint64_t> val_indices;
        std::vector<uint64_t> test_indices;

        // V1: Random Split (0..N)
        void execute_random_split(size_t total_vectors, float train_pct, float val_pct, float test_pct);
        
        // V1.1: Random Split with explicit Item IDs
        void execute_split_with_items(const std::vector<uint64_t>& items, float train_pct, float val_pct, float test_pct);

        // [NEW] V2: Stratified Split (Chia đều theo nhãn)
        void execute_stratified_split(
            const std::vector<uint64_t>& items, 
            const std::vector<uint64_t>& labels, // Label hash
            float train_pct, float val_pct, float test_pct
        );

        void reset();
        bool has_split() const { return !train_indices.empty(); }
        bool save(const std::string& data_root);
        bool load(const std::string& data_root);

    private:
        const pomai::config::PomaiConfig& cfg_;
        // Make MAGIC inline to avoid ODR/linkage issues and allow use across TUs
        inline static constexpr char MAGIC[8] = {'P','M','S','P','L','I','T','1'};
    };

} // namespace pomai::core