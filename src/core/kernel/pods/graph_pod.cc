#include "core/kernel/pods/graph_pod.h"
#include <cstring>

namespace pomai::core {
    namespace {
    void SetStatus(Status* out, const Status& st) {
        if (out) *out = st;
    }
    }

    struct AddVertexPayload {
        VertexId id;
        TagId tag;
        // Metadata handled separately or as tail
    };

    struct AddEdgePayload {
        VertexId src;
        VertexId dst;
        EdgeType type;
        uint32_t rank;
    };

    void GraphPod::Handle(Message&& msg) {
        switch (msg.opcode) {
            case Op::kAddVertex: {
                if (msg.payload.size() < sizeof(AddVertexPayload)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("graph add-vertex payload too small"));
                    return;
                }
                const auto* p = reinterpret_cast<const AddVertexPayload*>(msg.payload.data());
                // Metadata is empty for now or extracted from tail
                auto st = runtime_->AddVertex(p->id, p->tag, Metadata());
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kAddEdge: {
                if (msg.payload.size() < sizeof(AddEdgePayload)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("graph add-edge payload too small"));
                    return;
                }
                const auto* p = reinterpret_cast<const AddEdgePayload*>(msg.payload.data());
                auto st = runtime_->AddEdge(p->src, p->dst, p->type, p->rank, Metadata());
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kGetNeighbors: {
                if (msg.payload.size() < sizeof(VertexId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("graph get-neighbors payload too small"));
                    return;
                }
                VertexId src = *reinterpret_cast<const VertexId*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::vector<Neighbor>*>(msg.result_ptr);
                    (void)runtime_->GetNeighbors(src, out);
                }
                break;
            }
            case Op::kGetNeighborsWithType: {
                struct P { VertexId src; EdgeType type; };
                if (msg.payload.size() < sizeof(P)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("graph get-neighbors-type payload too small"));
                    return;
                }
                const auto* p = reinterpret_cast<const P*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::vector<Neighbor>*>(msg.result_ptr);
                    (void)runtime_->GetNeighbors(p->src, p->type, out);
                }
                break;
            }
            case Op::kDeleteVertex: {
                if (msg.payload.size() < sizeof(VertexId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("graph delete-vertex payload too small"));
                    return;
                }
                VertexId id = *reinterpret_cast<const VertexId*>(msg.payload.data());
                SetStatus(msg.status_ptr, runtime_->DeleteVertex(id));
                break;
            }
            case Op::kDeleteEdge: {
                struct P { VertexId src; VertexId dst; EdgeType type; };
                if (msg.payload.size() < sizeof(P)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("graph delete-edge payload too small"));
                    return;
                }
                const auto* p = reinterpret_cast<const P*>(msg.payload.data());
                SetStatus(msg.status_ptr, runtime_->DeleteEdge(p->src, p->dst, p->type));
                break;
            }
            default:
                SetStatus(msg.status_ptr, Status::InvalidArgument("graph pod unsupported opcode"));
                break;
        }
    }

} // namespace pomai::core
