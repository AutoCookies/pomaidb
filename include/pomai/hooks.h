#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "pomai/types.h"
#include "pomai/metadata.h"

namespace pomai {

/**
 * @brief Interface for post-ingestion triggers.
 * Hooks are called after a successful Put or AddVector operation.
 */
class PostPutHook {
public:
    virtual ~PostPutHook() = default;

    /**
     * @brief Called after a vector is added to a membrane.
     * @param id The vector ID.
     * @param vec The vector data (zero-copy view).
     * @param meta Optional metadata.
     */
    virtual void OnPostPut(VectorId id, std::span<const float> vec, const Metadata& meta) = 0;
};

} // namespace pomai
