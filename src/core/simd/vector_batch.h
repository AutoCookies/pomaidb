#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <span>

namespace pomai::core {

/**
 * @brief Representation of how data is physically laid out in a vector.
 * Distilled from DuckDB's UnifiedVectorFormat.
 */
enum class VectorFormat : uint8_t {
    FLAT,       ///< Contiguous array of values
    CONSTANT,   ///< Single value repeated N times
    DICTIONARY, ///< Indirection through a selection vector
    SEQUENCE    ///< Auto-incrementing sequence (useful for ALP)
};

/**
 * @brief Selection vector for DICTIONARY format or filtered results.
 */
using SelectionVector = std::span<const uint32_t>;

/**
 * @brief A batch of vectors (rows) for SIMD processing.
 * Typically 1024 or 2048 rows.
 */
template <typename T>
class VectorBatch {
public:
    VectorBatch(uint32_t capacity, uint32_t dim)
        : capacity_(capacity), dim_(dim), size_(0), format_(VectorFormat::FLAT) {
        data_.resize(capacity * dim);
    }

    // Accessors
    uint32_t size() const { return size_; }
    uint32_t dim() const { return dim_; }
    VectorFormat format() const { return format_; }
    
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    void set_size(uint32_t s) { size_ = s; }
    void set_format(VectorFormat f) { format_ = f; }

    /// Return a pointer to the i-th vector in the batch (assuming FLAT)
    const T* get_vector(uint32_t i) const {
        return data_.data() + (i * dim_);
    }

    /// Selection vector for DICTIONARY or filtered views
    void set_selection(SelectionVector sel) {
        selection_ = sel;
        format_ = VectorFormat::DICTIONARY;
    }

    const uint32_t* selection() const {
        return selection_.data();
    }

private:
    uint32_t capacity_;
    uint32_t dim_;
    uint32_t size_;
    VectorFormat format_;
    std::vector<T> data_;
    SelectionVector selection_;
};

using FloatBatch = VectorBatch<float>;

} // namespace pomai::core
