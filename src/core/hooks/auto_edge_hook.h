#pragma once

#include "pomai/hooks.h"
#include "pomai/database.h"
#include "pomai/metadata.h"

namespace pomai::core {

/**
 * @brief Hook that automatically creates a graph edge when a vector is ingested with a src_vid.
 */
class AutoEdgeHook : public pomai::PostPutHook {
public:
    explicit AutoEdgeHook(pomai::Database* db) : db_(db) {}

    void OnPostPut(VectorId id, std::span<const float> vec, const Metadata& meta) override {
        if (meta.src_vid != 0 && db_) {
            // Create a semantic edge from the source vertex to the new vector vertex (same ID)
            // Note: In PomaiDB, VertexId and VectorId share the same namespace for simplicity in this MVP.
            (void)db_->AddEdge(meta.src_vid, id, 1 /* SemanticLink */, 0 /* Rank */, meta);
        }
    }

private:
    pomai::Database* db_;
};

} // namespace pomai::core
