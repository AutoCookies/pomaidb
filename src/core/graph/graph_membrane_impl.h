#pragma once

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

#include "pomai/graph.h"
#include "pomai/status.h"
#include "storage/wal/wal.h"
#include "core/graph/graph_key.h"

namespace pomai::core {

/**
 * @brief Internal implementation of GraphMembrane.
 */
class GraphMembraneImpl : public pomai::GraphMembrane {
public:
    GraphMembraneImpl(std::unique_ptr<storage::Wal> wal) : wal_(std::move(wal)) {}

    Status AddVertex(VertexId id, TagId tag, const Metadata& meta) override {
        // 1. Persist to WAL
        std::string key = GraphKey::EncodeVertex(id, tag);
        Status st = wal_->AppendRawKV(4 /* kRawKV */, Slice(key), Slice(meta.tenant));
        if (!st.ok()) return st;

        // 2. Update RAM index (structural only)
        if (adj_lists_.find(id) == adj_lists_.end()) {
            adj_lists_[id] = {};
        }
        return Status::Ok();
    }

    Status AddEdge(VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta) override {
        // 1. Persist to WAL
        std::string key = GraphKey::EncodeEdge(src, type, rank, dst);
        Status st = wal_->AppendRawKV(4 /* kRawKV */, Slice(key), Slice(meta.tenant));
        if (!st.ok()) return st;

        // 2. Update RAM index (structural only)
        // Add to contiguous store
        Neighbor n{dst, type, rank};
        auto& list = adj_lists_[src];
        list.push_back(n);
        return Status::Ok();
    }

    Status GetNeighbors(VertexId src, std::vector<Neighbor>* out) override {
        auto it = adj_lists_.find(src);
        if (it != adj_lists_.end()) {
            *out = it->second;
        }
        return Status::Ok();
    }

    Status GetNeighbors(VertexId src, EdgeType type, std::vector<Neighbor>* out) override {
        auto it = adj_lists_.find(src);
        if (it != adj_lists_.end()) {
            for (const auto& n : it->second) {
                if (n.type == type) {
                    out->push_back(n);
                }
            }
        }
        return Status::Ok();
    }

    Status Flush() override {
        return wal_->Flush();
    }

    Status BeginBatch() { return wal_ ? wal_->BeginBatch() : Status::Ok(); }
    Status EndBatch() { return wal_ ? wal_->EndBatch() : Status::Ok(); }

    // Called during Database::Open()
    Status WarmUp() {
        // WarmUp can replay WAL entries to rebuild adj_lists_
        return Status::Ok();
    }

    void ForEachVertex(const std::function<void(pomai::VertexId id, std::size_t out_degree)>& fn) const {
        for (const auto& [vid, neigh] : adj_lists_) fn(vid, neigh.size());
    }

private:
    std::unique_ptr<storage::Wal> wal_;
    // Contiguous Adjacency Store (Simplified for now - using map to vectors but intended for mmap)
    // In a full implementation, this would be a single large buffer + offset index.
    std::unordered_map<VertexId, std::vector<Neighbor>> adj_lists_;
};

} // namespace pomai::core
