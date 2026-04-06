#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <endian.h>
#include <functional>
#include <unordered_map>
#include <vector>

#include "pomai/graph.h"
#include "pomai/slice.h"
#include "pomai/status.h"
#include "storage/wal/wal.h"
#include "core/graph/graph_key.h"

namespace pomai::core {

/**
 * @brief Internal implementation of GraphMembrane.
 *
 * Persists vertex and edge writes to a WAL using kRawKV records.
 * On restart, WarmUp() replays the WAL to rebuild adj_lists_.
 * Deletions use tombstone key prefixes ('T' for vertex, 'X' for edge)
 * that are also replayed during WarmUp().
 */
class GraphMembraneImpl : public pomai::GraphMembrane {
public:
    explicit GraphMembraneImpl(std::unique_ptr<storage::Wal> wal) : wal_(std::move(wal)) {}

    Status AddVertex(VertexId id, TagId tag, const Metadata& meta) override {
        std::string key = GraphKey::EncodeVertex(id, tag);
        Status st = wal_->AppendRawKV(4 /* kRawKV */, Slice(key), Slice(meta.tenant));
        if (!st.ok()) return st;
        if (adj_lists_.find(id) == adj_lists_.end()) {
            adj_lists_[id] = {};
            bytes_used_ += sizeof(VertexId) + sizeof(std::vector<Neighbor>);
        }
        return Status::Ok();
    }

    Status AddEdge(VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta) override {
        std::string key = GraphKey::EncodeEdge(src, type, rank, dst);
        Status st = wal_->AppendRawKV(4 /* kRawKV */, Slice(key), Slice(meta.tenant));
        if (!st.ok()) return st;
        Neighbor n{dst, type, rank};
        adj_lists_[src].push_back(n);
        bytes_used_ += sizeof(Neighbor);
        return Status::Ok();
    }

    Status DeleteVertex(VertexId id) override {
        // Tombstone key: 'T' (1) | VertexId (8)
        std::string key(1, 'T');
        uint64_t v_be = htobe64(id);
        key.append(reinterpret_cast<const char*>(&v_be), 8);
        Status st = wal_->AppendRawKV(4 /* kRawKV */, Slice(key), Slice(""));
        if (!st.ok()) return st;
        auto it = adj_lists_.find(id);
        if (it != adj_lists_.end()) {
            bytes_used_ -= sizeof(VertexId) + sizeof(std::vector<Neighbor>) +
                           it->second.size() * sizeof(Neighbor);
            adj_lists_.erase(it);
        }
        return Status::Ok();
    }

    Status DeleteEdge(VertexId src, VertexId dst, EdgeType type) override {
        // Tombstone key: 'X' (1) | SrcID (8) | EdgeType (4) | Rank=0 (4) | DstID (8)
        std::string key(1, 'X');
        uint64_t s_be = htobe64(src);
        uint32_t t_be = htobe32(static_cast<uint32_t>(type));
        uint32_t r_be = 0;
        uint64_t d_be = htobe64(dst);
        key.append(reinterpret_cast<const char*>(&s_be), 8);
        key.append(reinterpret_cast<const char*>(&t_be), 4);
        key.append(reinterpret_cast<const char*>(&r_be), 4);
        key.append(reinterpret_cast<const char*>(&d_be), 8);
        Status st = wal_->AppendRawKV(4 /* kRawKV */, Slice(key), Slice(""));
        if (!st.ok()) return st;
        auto it = adj_lists_.find(src);
        if (it != adj_lists_.end()) {
            auto& v = it->second;
            auto before = v.size();
            v.erase(std::remove_if(v.begin(), v.end(),
                [dst, type](const Neighbor& n) { return n.id == dst && n.type == type; }),
                v.end());
            bytes_used_ -= (before - v.size()) * sizeof(Neighbor);
        }
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
                if (n.type == type) out->push_back(n);
            }
        }
        return Status::Ok();
    }

    Status Flush() override { return wal_->Flush(); }

    Status BeginBatch() { return wal_ ? wal_->BeginBatch() : Status::Ok(); }
    Status EndBatch()   { return wal_ ? wal_->EndBatch()   : Status::Ok(); }

    /**
     * Called during Database::Open() to rebuild adj_lists_ from the WAL.
     * Delegates to Wal::ReplayGraphInto which calls ReplayEntry() for each
     * kRawKV record found in the WAL segments.
     */
    Status WarmUp() { return wal_->ReplayGraphInto(this); }

    /**
     * Called by Wal::ReplayGraphInto for each kRawKV key decoded from the WAL.
     * Reconstructs adj_lists_ and bytes_used_ from the encoded key bytes.
     */
    void ReplayEntry(pomai::Slice key) {
        if (key.size() < 1) return;
        const auto* p = static_cast<const uint8_t*>(static_cast<const void*>(key.data()));
        const uint8_t prefix = p[0];

        if (prefix == GraphKey::kVertex && key.size() >= 13) {
            uint64_t vid_be;
            std::memcpy(&vid_be, p + 1, 8);
            VertexId vid = be64toh(vid_be);
            if (adj_lists_.find(vid) == adj_lists_.end()) {
                adj_lists_[vid] = {};
                bytes_used_ += sizeof(VertexId) + sizeof(std::vector<Neighbor>);
            }

        } else if (prefix == GraphKey::kEdge && key.size() >= 25) {
            uint64_t src_be, dst_be;
            uint32_t type_be, rank_be;
            std::memcpy(&src_be,  p + 1,  8);
            std::memcpy(&type_be, p + 9,  4);
            std::memcpy(&rank_be, p + 13, 4);
            std::memcpy(&dst_be,  p + 17, 8);
            VertexId src = be64toh(src_be);
            Neighbor n{be64toh(dst_be), be32toh(type_be), be32toh(rank_be)};
            adj_lists_[src].push_back(n);
            bytes_used_ += sizeof(Neighbor);

        } else if (prefix == 'T' && key.size() >= 9) {
            // Vertex tombstone
            uint64_t vid_be;
            std::memcpy(&vid_be, p + 1, 8);
            VertexId vid = be64toh(vid_be);
            auto it = adj_lists_.find(vid);
            if (it != adj_lists_.end()) {
                bytes_used_ -= sizeof(VertexId) + sizeof(std::vector<Neighbor>) +
                               it->second.size() * sizeof(Neighbor);
                adj_lists_.erase(it);
            }

        } else if (prefix == 'X' && key.size() >= 25) {
            // Edge tombstone
            uint64_t src_be, dst_be;
            uint32_t type_be;
            std::memcpy(&src_be,  p + 1,  8);
            std::memcpy(&type_be, p + 9,  4);
            // rank at +13 is ignored for tombstone matching
            std::memcpy(&dst_be,  p + 17, 8);
            VertexId src   = be64toh(src_be);
            VertexId dst   = be64toh(dst_be);
            EdgeType etype = be32toh(type_be);
            auto it = adj_lists_.find(src);
            if (it != adj_lists_.end()) {
                auto& v = it->second;
                auto before = v.size();
                v.erase(std::remove_if(v.begin(), v.end(),
                    [dst, etype](const Neighbor& n) { return n.id == dst && n.type == etype; }),
                    v.end());
                bytes_used_ -= (before - v.size()) * sizeof(Neighbor);
            }
        }
    }

    std::size_t MemoryBytesUsed() const { return bytes_used_; }

    void ForEachVertex(const std::function<void(pomai::VertexId id, std::size_t out_degree)>& fn) const {
        for (const auto& [vid, neigh] : adj_lists_) fn(vid, neigh.size());
    }

private:
    std::unique_ptr<storage::Wal> wal_;
    // In-memory adjacency store. Rebuilt from WAL on open via WarmUp().
    std::unordered_map<VertexId, std::vector<Neighbor>> adj_lists_;
    // Approximate memory usage for backpressure / quota reporting.
    std::size_t bytes_used_ = 0;
};

} // namespace pomai::core
