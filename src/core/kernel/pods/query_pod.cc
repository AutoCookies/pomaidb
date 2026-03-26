#include "core/kernel/pods/query_pod.h"
#include "pomai/search.h"

namespace pomai::core {

    void QueryPod::Handle(Message&& msg) {
        switch (msg.opcode) {
            case 0x08: { // SearchMultiModal
                // Payload contains pointer to MultiModalQuery
                if (msg.payload.size() < sizeof(void*)) return;
                const MultiModalQuery* query = *reinterpret_cast<const MultiModalQuery* const*>(msg.payload.data());
                
                if (msg.result_ptr) {
                    auto* out = static_cast<SearchResult*>(msg.result_ptr);
                    (void)planner_->Execute(msg.membrane_id, *query, out);
                }
                break;
            }
            default:
                break;
        }
    }

} // namespace pomai::core
