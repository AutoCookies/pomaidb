#pragma once
/*
 * src/core/pomai_db.h
 *
 * PomaiDB: Quả lựu vector database – đa màng lưu, mỗi màng dim riêng.
 * Persistent manifest on disk (simple, no third-party deps).
 *
 * Persistence strategy (simple and robust):
 *  - A repository data directory is chosen from env POMAI_DB_DIR or defaults to "./data/pomai_db".
 *  - A small manifest file "<data_root>/membrances.manifest" stores one membrance per line:
 *        <name>|<dim>|<ram_mb>
 *    (pipe-separated, text). This is written atomically via write-to-temp + rename.
 *  - Each membrance gets its own subdirectory: "<data_root>/<name>" and PomaiOrbit is configured
 *    with that path so it may persist its own schema (pomai_schema.bin).
 *
 * WAL:
 *  - A WAL file "<data_root>/wal.log" is used to record create/drop membrance operations (WAL_REC_CREATE_MEMBRANCE / WAL_REC_DROP_MEMBRANCE).
 *  - On startup WAL is replayed (after loading manifest) to bring DB to latest state.
 *
 * Concurrency:
 *  - To allow concurrent SEARCH (read) while DROP/CREATE (write) happen, the central guard is std::shared_mutex.
 *    Readers take std::shared_lock, writers take std::unique_lock.
 */

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>
#include <shared_mutex>
#include <iostream>
#include <optional>

#include "src/memory/shard_arena.h"
#include "src/ai/pomai_orbit.h"
#include "src/memory/wal_manager.h"
#include "src/core/metadata_index.h" // metadata index for per-membrance pre-filtering

namespace pomai::core
{

    struct MembranceConfig
    {
        size_t dim = 0;
        size_t ram_mb = 256;
        std::string engine = "orbit";

        // Optional toggle: if true the Membrance will allocate a MetadataIndex on construction.
        // Useful to avoid the small memory cost for membs that won't use metadata filtering.
        bool enable_metadata_index = true;
    };

    struct Membrance
    {
        std::string name;
        size_t dim;
        size_t ram_mb;
        std::string data_path; // directory on disk for this membrance
        std::unique_ptr<pomai::memory::ShardArena> arena;
        std::unique_ptr<pomai::ai::orbit::PomaiOrbit> orbit;

        // Optional per-membrance metadata inverted index (Tag -> [label ids]).
        // Server checks for non-null before using it. Kept as unique_ptr so we can
        // lazily enable/disable and avoid persisting if not needed.
        std::unique_ptr<MetadataIndex> meta_index;

        // Construct a membrance and initialize its arena + orbit.
        // data_root is the base directory where per-membrance directory will be created.
        Membrance(const std::string &name, const MembranceConfig &cfg, const std::string &data_root);

        // Non-copyable
        Membrance(const Membrance &) = delete;
        Membrance &operator=(const Membrance &) = delete;
    };

    class PomaiDB
    {
    public:
        // Create/load DB. data_root can be provided via env POMAI_DB_DIR or passed here.
        explicit PomaiDB(const std::string &data_root = std::string());
        ~PomaiDB();

        PomaiDB(const PomaiDB &) = delete;
        PomaiDB &operator=(const PomaiDB &) = delete;

        // Create/Drop membrance
        bool create_membrance(const std::string &name, const MembranceConfig &cfg);
        bool drop_membrance(const std::string &name);

        // Accessors
        Membrance *get_membrance(const std::string &name);
        const Membrance *get_membrance(const std::string &name) const;
        std::vector<std::string> list_membrances() const;

        // Vector API forwarded to membrance orbit
        bool insert(const std::string &membr, const float *vec, uint64_t label);
        std::vector<std::pair<uint64_t, float>> search(const std::string &membr, const float *query, size_t k);
        bool get(const std::string &membr, uint64_t label, std::vector<float> &out);
        bool remove(const std::string &membr, uint64_t label);

        size_t get_membrance_dim(const std::string &name) const;

        // Expose manifest/wal helpers (for tests/tools)
        bool save_manifest();
        bool save_all_membrances();
        bool insert_batch(const std::string &membr, const std::vector<std::pair<uint64_t, std::vector<float>>> &batch);

    private:
        // Persistence helpers
        bool load_manifest();                                                                // populates membrances_ from manifest
        bool create_membrance_internal(const std::string &name, const MembranceConfig &cfg); // doesn't persist manifest/WAL

        // members
        std::unordered_map<std::string, std::unique_ptr<Membrance>> membrances_;
        mutable std::shared_mutex mu_; // allow concurrent readers for searches while writers create/drop

        // persistence
        std::string data_root_;     // base data directory
        std::string manifest_path_; // file path of manifest

        // WAL manager
        pomai::memory::WalManager wal_;
    };

} // namespace pomai::core