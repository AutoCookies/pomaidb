#pragma once

#include <cerrno>
#include <cstring>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <optional>

namespace pomai
{
    enum class PomaiErrc : std::uint8_t
    {
        Ok = 0,
        InvalidArgument = 1,
        NotFound = 2,
        AlreadyExists = 3,
        Corruption = 4,
        IoError = 5,
        ResourceExhausted = 6,
        Timeout = 7,
        Cancelled = 8,
        Unavailable = 9,
        NotSupported = 10,
        BudgetExhausted = 11,
        Internal = 12
    };

    struct Status
    {
        PomaiErrc code{PomaiErrc::Ok};
        std::string msg{};

        constexpr bool ok() const { return code == PomaiErrc::Ok; }

        static Status Ok() { return {}; }

        static Status FromErrno(PomaiErrc code, std::string_view what)
        {
            Status s;
            s.code = code;
            s.msg = std::string(what) + ": " + std::string(std::strerror(errno));
            return s;
        }

        static Status Invalid(std::string msg)
        {
            return {PomaiErrc::InvalidArgument, std::move(msg)};
        }

        static Status Io(std::string msg)
        {
            return {PomaiErrc::IoError, std::move(msg)};
        }

        static Status Corrupt(std::string msg)
        {
            return {PomaiErrc::Corruption, std::move(msg)};
        }

        static Status Exhausted(std::string msg)
        {
            return {PomaiErrc::ResourceExhausted, std::move(msg)};
        }

        static Status Unavailable(std::string msg)
        {
            return {PomaiErrc::Unavailable, std::move(msg)};
        }

        static Status NotSupported(std::string msg)
        {
            return {PomaiErrc::NotSupported, std::move(msg)};
        }

        static Status Internal(std::string msg)
        {
            return {PomaiErrc::Internal, std::move(msg)};
        }
    };

    template <typename T>
    class Result
    {
    public:
        Result() : status_(Status::Ok()) {}
        Result(Status s) : status_(std::move(s)) {}
        Result(T value) : status_(Status::Ok()), value_(std::move(value)) {}

        bool ok() const { return status_.ok(); }
        const Status &status() const { return status_; }

        T &value()
        {
            return *value_;
        }

        const T &value() const
        {
            return *value_;
        }

        T &&move_value()
        {
            return std::move(*value_);
        }

    private:
        Status status_{};
        std::optional<T> value_{};
    };

    template <>
    class Result<void>
    {
    public:
        Result() : status_(Status::Ok()) {}
        Result(Status s) : status_(std::move(s)) {}

        bool ok() const { return status_.ok(); }
        const Status &status() const { return status_; }

    private:
        Status status_{};
    };
} // namespace pomai
