#pragma once
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "options.h"
#include "search.h"
#include "status.h"
#include "types.h"
#include "write_batch.h"

namespace pomai
{

    class DB
    {
    public:
        virtual ~DB() = default;

        // DB lifetime
        virtual Status Flush() = 0;
        virtual Status Close() = 0;

        // Default membrane (optional semantic; can map to "default")
        virtual Status Put(VectorId id, std::span<const float> vec) = 0;
        virtual Status Delete(VectorId id) = 0;
        virtual Status Search(std::span<const float> query, uint32_t topk,
                              SearchResult *out) = 0;

        // WriteBatch API (atomic batch writes)
        virtual Status Write(const WriteBatch &batch) = 0;

        // Membrane API
        virtual Status CreateMembrane(const MembraneSpec &spec) = 0;
        virtual Status DropMembrane(std::string_view name) = 0;
        virtual Status OpenMembrane(std::string_view name) = 0;
        virtual Status CloseMembrane(std::string_view name) = 0;
        virtual Status ListMembranes(std::vector<std::string> *out) const = 0;

        virtual Status Put(std::string_view membrane, VectorId id,
                           std::span<const float> vec) = 0;
        virtual Status Delete(std::string_view membrane, VectorId id) = 0;
        virtual Status Search(std::string_view membrane, std::span<const float> query,
                              uint32_t topk, SearchResult *out) = 0;

        // WriteBatch API (per-membrane atomic batch writes)
        virtual Status Write(std::string_view membrane, const WriteBatch &batch) = 0;

        static Status Open(const DBOptions &options, std::unique_ptr<DB> *out);
    };

} // namespace pomai
