#ifndef POMAIDB_HNSW_H
#define POMAIDB_HNSW_H

#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <mutex>
#include <memory>
#include <functional>

namespace pomai::hnsw {

/**
 * Minimalist, high-performance HNSW implementation.
 * Decoupled from FAISS infrastructure.
 */

using storage_idx_t = int32_t;

struct NodeDist {
    float dist;
    storage_idx_t id;

    bool operator<(const NodeDist& other) const {
        return dist > other.dist; // Max-heap (farther elements first)
    }
};

struct NodeDistCloser {
    float dist;
    storage_idx_t id;

    bool operator<(const NodeDistCloser& other) const {
        return dist < other.dist; // Min-heap (closer elements first)
    }
};

class HNSW {
public:
    explicit HNSW(int M = 32, int ef_construction = 200);
    ~HNSW();

    // Configuration
    int M;
    int ef_construction;
    int ef_search;

    // Distance function: (id1, id2) -> float
    using DistanceComputer = std::function<float(storage_idx_t, storage_idx_t)>;
    // Query distance function: (query_vec_id, target_id) -> float
    using QueryDistanceComputer = std::function<float(storage_idx_t)>;

    /**
     * Adds a point to the graph. 
     * @param id The internal storage ID.
     * @param level The level to insert the point at (-1 for random).
     * @param qdis Distance computer for the new point against existing points.
     */
    void add_point(storage_idx_t id, int level, DistanceComputer& qdis);

    /**
     * Searches for the nearest neighbors of a query.
     * @param qdis Distance computer for the query against existing points.
     * @param k Top-k results.
     * @param ef Search expansion factor.
     * @param out_ids Output IDs.
     * @param out_dists Output distances.
     */
    void search(QueryDistanceComputer& qdis, int k, int ef, 
                std::vector<storage_idx_t>& out_ids, 
                std::vector<float>& out_dists) const;

    // Persistence
    void save(FILE* f) const;
    void load(FILE* f);

    int get_random_level();

private:
    struct LevelData {
        std::vector<int> cum_nneighbor_per_level;
        std::vector<int> levels;           // [ntotal]
        std::vector<size_t> offsets;      // [ntotal + 1]
        std::vector<storage_idx_t> neighbors; // flat pool
    } graph;

    storage_idx_t entry_point = -1;
    int max_level = -1;
    
    std::mt19937 rng;
    double level_mult;

    // Internal helpers
    void neighbor_range(storage_idx_t id, int level, size_t& begin, size_t& end) const;
    void shrink_neighbor_list(DistanceComputer& qdis, storage_idx_t cur, 
                             std::priority_queue<NodeDist>& candidates, int max_size);
    
    mutable std::mutex graph_mutex;
    std::vector<std::unique_ptr<std::mutex>> node_locks;
};

} // namespace pomai::hnsw

#endif // POMAIDB_HNSW_H
