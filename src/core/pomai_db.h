#pragma once

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <shared_mutex>
#include <iostream>
#include <optional>
#include <thread>
#include <atomic>
#include <functional>

#include "src/memory/shard_arena.h"
#include "src/ai/pomai_orbit.h"
#include "src/memory/wal_manager.h"
#include "src/core/metadata_index.h"
#include "src/core/hot_tier.h"
#include "src/core/config.h"
#include "src/core/split_manager.h"
#include "src/core/types.h"

namespace pomai::core
{
    struct MembranceConfig
    {
        size_t dim = 0;
        size_t ram_mb = 256;
        std::string engine = "orbit";
        bool enable_metadata_index = true;
        pomai::core::DataType data_type = pomai::core::DataType::FLOAT32;
    };

    struct Membrance
    {
        std::string name;
        size_t dim;
        size_t ram_mb;
        std::string data_path;
        pomai::core::DataType data_type; // configured storage type

        std::unique_ptr<pomai::memory::ShardArena> arena;
        std::unique_ptr<pomai::ai::orbit::PomaiOrbit> orbit;
        std::unique_ptr<HotTier> hot_tier;

        // shared metadata index
        std::shared_ptr<MetadataIndex> meta_index;

        std::unique_ptr<SplitManager> split_mgr;

        // Constructor takes Global Config
        Membrance(const std::string &nm, const MembranceConfig &cfg,
                  const std::string &data_root, const pomai::config::PomaiConfig &global_cfg);

        // Disable copy
        Membrance(const Membrance &) = delete;
        Membrance &operator=(const Membrance &) = delete;
    };

    class PomaiDB
    {
    public:
        explicit PomaiDB(const pomai::config::PomaiConfig &config);
        ~PomaiDB();

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

        // New: iterate in batches and deliver raw stored bytes to a consumer callback.
        // - mode: "TRAIN"/"VAL"/"TEST" or empty/other to mean "all"
        // - off/lim: offset and limit within the selected index list
        // - batch_size: number of vectors per invocation to the consumer
        // - consumer: function(ids, concatenated_raw_bytes, per_vector_bytes)
        //   per_vector_bytes is the element_size * dim for this membrance (fixed per call)
        bool iterate_batch(const std::string &membr,
                           const std::string &mode,
                           size_t off,
                           size_t lim,
                           size_t batch_size,
                           const std::function<void(const std::vector<uint64_t> &ids, const std::string &concat_buf, uint32_t per_vec_bytes)> &consumer);

        bool fetch_batch_raw(const std::string &membr, const std::vector<uint64_t> &ids, std::vector<std::string> &outs);
        bool checkpoint_all();

    private:
        bool load_manifest();
        bool create_membrance_internal(const std::string &name, const MembranceConfig &cfg);
        void background_worker();

        std::unordered_map<std::string, std::unique_ptr<Membrance>> membrances_;
        mutable std::shared_mutex mu_;

        const pomai::config::PomaiConfig &config_;
        pomai::memory::WalManager wal_;

        std::thread bg_thread_;
        std::atomic<bool> bg_running_;
    };
}