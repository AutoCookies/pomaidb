#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pomai/types.h"

namespace pomai
{

    /**
     * WriteBatch represents a batch of Put and Delete operations that can be
     * atomically applied to the database (per shard). This is more efficient
     * than individual operations because:
     * - Single WAL sync per batch (instead of per operation)
     * - Amortized command queue overhead
     * - Better throughput for bulk ingest
     *
     * Example usage:
     *   WriteBatch batch;
     *   batch.Put(1, vec1);
     *   batch.Put(2, vec2);
     *   batch.Delete(3);
     *   db->Write(batch);
     */
    class WriteBatch
    {
    public:
        enum class OpType : std::uint8_t
        {
            kPut = 1,
            kDelete = 2
        };

        struct Op
        {
            OpType type;
            VectorId id;
            std::vector<float> vec; // empty for Delete

            Op(OpType t, VectorId i, std::vector<float> v = {})
                : type(t), id(i), vec(std::move(v)) {}
        };

        WriteBatch() = default;

        /**
         * Add a Put operation to the batch.
         * The vector data is copied into the batch.
         */
        void Put(VectorId id, std::span<const float> vec)
        {
            ops_.emplace_back(OpType::kPut, id, std::vector<float>(vec.begin(), vec.end()));
        }

        /**
         * Add a Delete operation to the batch.
         */
        void Delete(VectorId id)
        {
            ops_.emplace_back(OpType::kDelete, id);
        }

        /**
         * Clear all operations from the batch.
         */
        void Clear() { ops_.clear(); }

        /**
         * Return the number of operations in the batch.
         */
        std::size_t Count() const { return ops_.size(); }

        /**
         * Return true if the batch is empty.
         */
        bool Empty() const { return ops_.empty(); }

        /**
         * Access the operations (internal use by DB implementation).
         */
        const std::vector<Op> &Ops() const { return ops_; }

    private:
        std::vector<Op> ops_;
    };

} // namespace pomai
