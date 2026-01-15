#pragma once
/*
 * src/core/pomai_db.h
 * ... (Copyright/Intro)
 */

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <shared_mutex>
#include <iostream>
#include <optional>
#include <thread>
#include <atomic>

#include "src/memory/shard_arena.h"
#include "src/ai/pomai_orbit.h"
#include "src/memory/wal_manager.h"
#include "src/core/metadata_index.h"
#include "src/core/hot_tier.h"

namespace pomai::core
{
    // ... (MembranceConfig and Membrance structs remain unchanged)
    struct MembranceConfig
    {
        size_t dim = 0;
        size_t ram_mb = 256;
        std::string engine = "orbit";
        bool enable_metadata_index = true;
    };

    struct Membrance
    {
        std::string name;
        size_t dim;
        size_t ram_mb;
        std::string data_path;
        std::unique_ptr<pomai::memory::ShardArena> arena;
        std::unique_ptr<pomai::ai::orbit::PomaiOrbit> orbit;
        std::unique_ptr<HotTier> hot_tier;
        std::unique_ptr<MetadataIndex> meta_index;

        Membrance(const std::string &name, const MembranceConfig &cfg, const std::string &data_root);

        Membrance(const Membrance &) = delete;
        Membrance &operator=(const Membrance &) = delete;
    };

    class PomaiDB
    {
    public:
        explicit PomaiDB(const std::string &data_root = std::string());
        ~PomaiDB();

        // ... (Public API remains unchanged)
        PomaiDB(const PomaiDB &) = delete;
        PomaiDB &operator=(const PomaiDB &) = delete;

        bool create_membrance(const std::string &name, const MembranceConfig &cfg);
        bool drop_membrance(const std::string &name);

        Membrance *get_membrance(const std::string &name);
        const Membrance *get_membrance(const std::string &name) const;
        std::vector<std::string> list_membrances() const;

        bool insert(const std::string &membr, const float *vec, uint64_t label);
        std::vector<std::pair<uint64_t, float>> search(const std::string &membr, const float *query, size_t k);
        bool get(const std::string &membr, uint64_t label, std::vector<float> &out);
        bool remove(const std::string &membr, uint64_t label);

        size_t get_membrance_dim(const std::string &name) const;

        bool save_manifest();
        bool save_all_membrances();
        bool insert_batch(const std::string &membr, const std::vector<std::pair<uint64_t, std::vector<float>>> &batch);

    private:
        bool load_manifest();
        bool create_membrance_internal(const std::string &name, const MembranceConfig &cfg);

        // [Added] Background Worker: Drains HotTier -> Orbit
        void background_worker();

        std::unordered_map<std::string, std::unique_ptr<Membrance>> membrances_;
        mutable std::shared_mutex mu_;

        std::string data_root_;
        std::string manifest_path_;

        pomai::memory::WalManager wal_;

        // [Added] Thread management
        std::atomic<bool> running_{false};
        std::thread merger_thread_;
    };

} // namespace pomai::core