#include "hnsw.h"
#include <algorithm>
#include <cstdio>

namespace pomai::hnsw {

HNSW::HNSW(int M, int ef_construction)
    : M(M), ef_construction(ef_construction), ef_search(64), rng(42) {
    level_mult = 1.0 / log(double(M));
    graph.offsets.push_back(0);
}

HNSW::~HNSW() = default;

int HNSW::get_random_level() {
    std::uniform_real_distribution<double> dist(0, 1.0);
    double r = -log(dist(rng)) * level_mult;
    return (int)r;
}

void HNSW::neighbor_range(storage_idx_t id, int level, size_t& begin, size_t& end) const {
    size_t o = graph.offsets[id];
    int nn_at_level = (level == 0) ? M * 2 : M;
    int cum_before = 0;
    for (int l = 0; l < level; ++l) {
        cum_before += (l == 0) ? M * 2 : M;
    }
    begin = o + cum_before;
    end = begin + nn_at_level;
}

void HNSW::add_point(storage_idx_t id, int level, DistanceComputer& qdis) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    
    if (level < 0) level = get_random_level();
    
    // Resize graph metadata
    if (id >= (storage_idx_t)graph.levels.size()) {
        graph.levels.resize(id + 1);
        node_locks.emplace_back(std::make_unique<std::mutex>());
    }
    graph.levels[id] = level + 1;

    int nn_total = 0;
    for (int l = 0; l <= level; ++l) {
        nn_total += (l == 0) ? M * 2 : M;
    }
    
    size_t offset = graph.offsets.back();
    graph.offsets.push_back(offset + nn_total);
    graph.neighbors.resize(graph.offsets.back(), -1);

    if (entry_point == -1) {
        entry_point = id;
        max_level = level;
        return;
    }

    storage_idx_t curr = entry_point;
    float d_curr = qdis(id, curr);

    // 1. Greedy search to find the entry point at the target level
    for (int l = max_level; l > level; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            size_t begin, end;
            neighbor_range(curr, l, begin, end);
            for (size_t i = begin; i < end; ++i) {
                storage_idx_t v = graph.neighbors[i];
                if (v < 0) break;
                float d = qdis(id, v);
                if (d < d_curr) {
                    d_curr = d;
                    curr = v;
                    changed = true;
                }
            }
        }
    }

    // 2. Insert into each level
    for (int l = std::min(level, max_level); l >= 0; --l) {
        std::priority_queue<NodeDist> candidates;
        std::priority_queue<NodeDistCloser> visited;
        
        candidates.push({d_curr, curr});
        visited.push({d_curr, curr});
        
        std::vector<storage_idx_t> seen;
        seen.push_back(curr);
        seen.push_back(id);

        while (!candidates.empty()) {
            NodeDist top = candidates.top();
            if (top.dist > visited.top().dist && (int)visited.size() >= ef_construction) break;
            candidates.pop();

            size_t begin, end;
            neighbor_range(top.id, l, begin, end);
            for (size_t i = begin; i < end; ++i) {
                storage_idx_t v = graph.neighbors[i];
                if (v < 0) break;
                if (std::find(seen.begin(), seen.end(), v) != seen.end()) continue;
                seen.push_back(v);

                float d = qdis(id, v);
                if ((int)visited.size() < ef_construction || d < visited.top().dist) {
                    candidates.push({d, v});
                    visited.push({d, v});
                    if ((int)visited.size() > ef_construction) visited.pop();
                }
            }
        }

        // Convert visited to NodeDist for shrinking
        std::priority_queue<NodeDist> results;
        while (!visited.empty()) {
            results.push({visited.top().dist, visited.top().id});
            visited.pop();
        }

        int max_m = (l == 0) ? M * 2 : M;
        shrink_neighbor_list(qdis, id, results, max_m);

        // Add bidirectional links
        size_t b_id, e_id;
        neighbor_range(id, l, b_id, e_id);
        size_t idx = b_id;
        while (!results.empty() && idx < e_id) {
            storage_idx_t neighbor = results.top().id;
            graph.neighbors[idx++] = neighbor;
            
            // Backlink
            size_t b_nb, e_nb;
            neighbor_range(neighbor, l, b_nb, e_nb);
            // Simple backlink: find first empty or farthest replaced if full
            // For simplicity in this standalone version, we just find the first -1
            bool found = false;
            for (size_t j = b_nb; j < e_nb; ++j) {
                if (graph.neighbors[j] == -1) {
                    graph.neighbors[j] = id;
                    found = true;
                    break;
                }
            }
            // If full, we should ideally shrink. 
            // Here we skip for brevity, as HNSW is robust to some missing links.
            
            results.pop();
        }
    }

    if (level > max_level) {
        max_level = level;
        entry_point = id;
    }
}

void HNSW::shrink_neighbor_list(DistanceComputer& qdis, storage_idx_t cur, 
                             std::priority_queue<NodeDist>& candidates, int max_size) {
    if ((int)candidates.size() <= max_size) return;
    
    std::vector<NodeDist> result;
    while (!candidates.empty()) {
        NodeDist top = candidates.top();
        candidates.pop();
        
        bool good = true;
        for (const auto& r : result) {
            if (qdis(top.id, r.id) < top.dist) {
                good = false;
                break;
            }
        }
        if (good) result.push_back(top);
        if ((int)result.size() >= max_size) break;
    }
    
    while (!candidates.empty()) candidates.pop();
    for (const auto& r : result) candidates.push(r);
}

void HNSW::search(QueryDistanceComputer& qdis, int k, int ef, 
                std::vector<storage_idx_t>& out_ids, 
                std::vector<float>& out_dists) const {
    if (entry_point == -1) return;

    storage_idx_t curr = entry_point;
    float d_curr = qdis(curr);

    for (int l = max_level; l > 0; --l) {
        bool changed = true;
        while (changed) {
            changed = false;
            size_t begin, end;
            neighbor_range(curr, l, begin, end);
            for (size_t i = begin; i < end; ++i) {
                storage_idx_t v = graph.neighbors[i];
                if (v < 0) break;
                float d = qdis(v);
                if (d < d_curr) {
                    d_curr = d;
                    curr = v;
                    changed = true;
                }
            }
        }
    }

    std::priority_queue<NodeDist> candidates;
    std::priority_queue<NodeDistCloser> top_k;
    
    candidates.push({d_curr, curr});
    top_k.push({d_curr, curr});
    
    std::vector<storage_idx_t> seen;
    seen.push_back(curr);

    while (!candidates.empty()) {
        NodeDist top = candidates.top();
        if (top.dist > top_k.top().dist && (int)top_k.size() >= ef) break;
        candidates.pop();

        size_t begin, end;
        neighbor_range(top.id, 0, begin, end);
        for (size_t i = begin; i < end; ++i) {
            storage_idx_t v = graph.neighbors[i];
            if (v < 0) break;
            if (std::find(seen.begin(), seen.end(), v) != seen.end()) continue;
            seen.push_back(v);

            float d = qdis(v);
            if ((int)top_k.size() < ef || d < top_k.top().dist) {
                candidates.push({d, v});
                top_k.push({d, v});
                if ((int)top_k.size() > ef) top_k.pop();
            }
        }
    }

    // Extract top-k
    std::vector<NodeDist> final_results;
    while (!top_k.empty()) {
        final_results.push_back({top_k.top().dist, top_k.top().id});
        top_k.pop();
    }
    std::sort(final_results.begin(), final_results.end(), [](const NodeDist& a, const NodeDist& b) {
        return a.dist < b.dist;
    });

    for (int i = 0; i < std::min((int)final_results.size(), k); ++i) {
        out_ids.push_back(final_results[i].id);
        out_dists.push_back(final_results[i].dist);
    }
}

void HNSW::save(FILE* f) const {
    fwrite(&M, sizeof(int), 1, f);
    fwrite(&ef_construction, sizeof(int), 1, f);
    fwrite(&entry_point, sizeof(storage_idx_t), 1, f);
    fwrite(&max_level, sizeof(int), 1, f);
    
    size_t n = graph.levels.size();
    fwrite(&n, sizeof(size_t), 1, f);
    fwrite(graph.levels.data(), sizeof(int), n, f);
    
    size_t n_offsets = graph.offsets.size();
    fwrite(&n_offsets, sizeof(size_t), 1, f);
    fwrite(graph.offsets.data(), sizeof(size_t), n_offsets, f);
    
    size_t n_neigh = graph.neighbors.size();
    fwrite(&n_neigh, sizeof(size_t), 1, f);
    fwrite(graph.neighbors.data(), sizeof(storage_idx_t), n_neigh, f);
}

void HNSW::load(FILE* f) {
    fread(&M, sizeof(int), 1, f);
    fread(&ef_construction, sizeof(int), 1, f);
    fread(&entry_point, sizeof(storage_idx_t), 1, f);
    fread(&max_level, sizeof(int), 1, f);
    
    size_t n;
    fread(&n, sizeof(size_t), 1, f);
    graph.levels.resize(n);
    fread(graph.levels.data(), sizeof(int), n, f);
    
    size_t n_offsets;
    fread(&n_offsets, sizeof(size_t), 1, f);
    graph.offsets.resize(n_offsets);
    fread(graph.offsets.data(), sizeof(size_t), n_offsets, f);
    
    size_t n_neigh;
    fread(&n_neigh, sizeof(size_t), 1, f);
    graph.neighbors.resize(n_neigh);
    fread(graph.neighbors.data(), sizeof(storage_idx_t), n_neigh, f);
    
    node_locks.clear();
    for (size_t i = 0; i < n; ++i) node_locks.emplace_back(std::make_unique<std::mutex>());
}

} // namespace pomai::hnsw
