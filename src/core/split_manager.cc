/*
 * src/core/split_manager.cc
 * Implementation of Data Splitting Logic.
 * [FIXED] Added <map> include and handled unused parameter warnings.
 */

#include "src/core/split_manager.h"
#include <map>       // [FIXED] Critical for std::map
#include <vector>
#include <numeric>
#include <cstring>
#include <algorithm> // for std::shuffle
#include <random>    // for std::mt19937

namespace pomai::core {

    SplitManager::SplitManager(const pomai::config::PomaiConfig& cfg) : cfg_(cfg) {}

    void SplitManager::reset() {
        train_indices.clear();
        val_indices.clear();
        test_indices.clear();
        
        // Force release RAM
        train_indices.shrink_to_fit();
        val_indices.shrink_to_fit();
        test_indices.shrink_to_fit();
    }

    void SplitManager::execute_random_split(size_t total_vectors, float train_pct, float val_pct, float test_pct) {
        reset();
        if (total_vectors == 0) return;

        // 1. Create sequential indices [0, 1, ... N-1]
        std::vector<uint64_t> items(total_vectors);
        std::iota(items.begin(), items.end(), 0);

        execute_split_with_items(items, train_pct, val_pct, test_pct);
    }

    void SplitManager::execute_split_with_items(const std::vector<uint64_t>& items, float train_pct, float val_pct, float test_pct) {
        // [FIXED] Silence unused warning (test_pct is implicitly used as "the rest")
        (void)test_pct;

        reset();
        if (items.empty()) return;

        std::vector<uint64_t> shuffled = items;
        
        // 2. Shuffle Deterministically
        std::mt19937_64 rng;
        if (cfg_.rng_seed.has_value()) {
            rng.seed(*cfg_.rng_seed);
        } else {
            rng.seed(std::random_device{}());
        }
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        size_t n = shuffled.size();
        size_t n_train = static_cast<size_t>(n * train_pct);
        size_t n_val = static_cast<size_t>(n * val_pct);
        
        size_t current = 0;
        
        // Assign Train
        if (n_train > 0) {
            train_indices.insert(train_indices.end(), shuffled.begin(), shuffled.begin() + n_train);
            current += n_train;
        }

        // Assign Val
        if (n_val > 0 && current < n) {
            val_indices.insert(val_indices.end(), shuffled.begin() + current, shuffled.begin() + current + n_val);
            current += n_val;
        }

        // Assign Test (Remaining)
        if (current < n) {
            test_indices.insert(test_indices.end(), shuffled.begin() + current, shuffled.end());
        }
    }

    void SplitManager::execute_stratified_split(
        const std::vector<uint64_t>& items, 
        const std::vector<uint64_t>& labels,
        float train_pct, float val_pct, float test_pct) 
    {
        // [FIXED] Silence unused warning
        (void)test_pct;

        reset();
        if (items.empty() || items.size() != labels.size()) return;

        // 1. Group items by Label
        std::map<uint64_t, std::vector<uint64_t>> groups;
        for(size_t i=0; i<items.size(); ++i) {
            groups[labels[i]].push_back(items[i]);
        }

        // 2. Setup RNG
        std::mt19937_64 rng;
        if (cfg_.rng_seed.has_value()) {
            rng.seed(*cfg_.rng_seed);
        } else {
            rng.seed(std::random_device{}());
        }

        // 3. Split each group
        for(auto& kv : groups) {
            std::vector<uint64_t>& group_items = kv.second;
            std::shuffle(group_items.begin(), group_items.end(), rng);

            size_t n = group_items.size();
            size_t n_train = static_cast<size_t>(n * train_pct);
            size_t n_val = static_cast<size_t>(n * val_pct);
            
            auto it = group_items.begin();
            
            if (n_train > 0) {
                train_indices.insert(train_indices.end(), it, it + n_train);
                it += n_train;
            }
            
            if (n_val > 0) {
                val_indices.insert(val_indices.end(), it, it + n_val);
                it += n_val;
            }
            
            // Remaining goes to test
            if (it != group_items.end()) {
                test_indices.insert(test_indices.end(), it, group_items.end());
            }
        }

        // 4. Shuffle final sets to mix labels
        std::shuffle(train_indices.begin(), train_indices.end(), rng);
        std::shuffle(val_indices.begin(), val_indices.end(), rng);
        std::shuffle(test_indices.begin(), test_indices.end(), rng);
    }

    bool SplitManager::save(const std::string& data_root) {
        std::string path = data_root + "/splits.bin";
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) return false;

        ofs.write(MAGIC, 8);
        
        uint64_t sz_tr = train_indices.size();
        uint64_t sz_va = val_indices.size();
        uint64_t sz_te = test_indices.size();

        ofs.write(reinterpret_cast<const char*>(&sz_tr), sizeof(sz_tr));
        ofs.write(reinterpret_cast<const char*>(&sz_va), sizeof(sz_va));
        ofs.write(reinterpret_cast<const char*>(&sz_te), sizeof(sz_te));

        if(sz_tr) ofs.write(reinterpret_cast<const char*>(train_indices.data()), sz_tr * sizeof(uint64_t));
        if(sz_va) ofs.write(reinterpret_cast<const char*>(val_indices.data()), sz_va * sizeof(uint64_t));
        if(sz_te) ofs.write(reinterpret_cast<const char*>(test_indices.data()), sz_te * sizeof(uint64_t));

        return true;
    }

    bool SplitManager::load(const std::string& data_root) {
        std::string path = data_root + "/splits.bin";
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;

        char magic[8];
        ifs.read(magic, 8);
        if(std::strncmp(magic, MAGIC, 8) != 0) return false;

        uint64_t sz_tr, sz_va, sz_te;
        ifs.read(reinterpret_cast<char*>(&sz_tr), sizeof(sz_tr));
        ifs.read(reinterpret_cast<char*>(&sz_va), sizeof(sz_va));
        ifs.read(reinterpret_cast<char*>(&sz_te), sizeof(sz_te));

        reset();
        if(sz_tr) {
            train_indices.resize(sz_tr);
            ifs.read(reinterpret_cast<char*>(train_indices.data()), sz_tr * sizeof(uint64_t));
        }
        if(sz_va) {
            val_indices.resize(sz_va);
            ifs.read(reinterpret_cast<char*>(val_indices.data()), sz_va * sizeof(uint64_t));
        }
        if(sz_te) {
            test_indices.resize(sz_te);
            ifs.read(reinterpret_cast<char*>(test_indices.data()), sz_te * sizeof(uint64_t));
        }
        return true;
    }

} // namespace pomai::core