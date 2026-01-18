#pragma once
// src/core/types.h
//
// Centralized data type enum and helpers used across the codebase.
// Provides mapping between textual names and enum values, and size utilities.

#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace pomai::core
{
    enum class DataType : uint8_t
    {
        FLOAT32 = 0,
        FLOAT64 = 1,
        INT32   = 2,
        INT8    = 3,
        FLOAT16 = 4
    };

    static inline std::size_t dtype_size(DataType t) noexcept
    {
        switch (t)
        {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT64: return 8;
        case DataType::INT32:   return 4;
        case DataType::INT8:    return 1;
        case DataType::FLOAT16: return 2;
        default: return 4;
        }
    }

    static inline std::string dtype_name(DataType t)
    {
        switch (t)
        {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT64: return "float64";
        case DataType::INT32:   return "int32";
        case DataType::INT8:    return "int8";
        case DataType::FLOAT16: return "float16";
        default: return "float32";
        }
    }

    static inline DataType parse_dtype(const std::string &s)
    {
        if (s == "float32" || s == "FLOAT32" || s == "float" || s == "FLOAT")
            return DataType::FLOAT32;
        if (s == "float64" || s == "FLOAT64" || s == "double" || s == "DOUBLE")
            return DataType::FLOAT64;
        if (s == "int32" || s == "INT32")
            return DataType::INT32;
        if (s == "int8" || s == "INT8")
            return DataType::INT8;
        if (s == "float16" || s == "FLOAT16" || s == "half" || s == "FP16")
            return DataType::FLOAT16;
        throw std::invalid_argument("unknown data type: " + s);
    }
}