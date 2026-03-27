#include "core/kernel/pods/query_pod.h"
#include "pomai/search.h"

namespace pomai::core {

    void QueryPod::Handle(Message&& msg) {
        switch (msg.opcode) {
            case Op::kSearchMultiModal: {
                struct ResultEnvelope {
                    SearchResult* out;
                    Status* st;
                };
                // Payload contains pointer to MultiModalQuery
                if (msg.payload.size() < sizeof(void*)) {
                    if (msg.result_ptr) {
                        auto* env = static_cast<ResultEnvelope*>(msg.result_ptr);
                        if (env && env->st) *env->st = Status::InvalidArgument("query payload too small");
                    }
                    if (msg.status_ptr) *msg.status_ptr = Status::InvalidArgument("query payload too small");
                    return;
                }
                const MultiModalQuery* query = *reinterpret_cast<const MultiModalQuery* const*>(msg.payload.data());
                auto* env = static_cast<ResultEnvelope*>(msg.result_ptr);
                if (!query || !env || !env->out || !env->st) {
                    if (env && env->st) *env->st = Status::InvalidArgument("query envelope invalid");
                    if (msg.status_ptr) *msg.status_ptr = Status::InvalidArgument("query envelope invalid");
                    return;
                }
                *env->st = planner_->Execute(msg.membrane_id, *query, env->out);
                if (msg.status_ptr) *msg.status_ptr = *env->st;
                break;
            }
            default:
                if (msg.status_ptr) *msg.status_ptr = Status::InvalidArgument("query pod unsupported opcode");
                break;
        }
    }

} // namespace pomai::core
