#pragma once
#include <cstdint>
#include <string>
#include <string_view>

namespace pomai
{

    enum class ErrorCode : std::uint32_t
    {
        kOk = 0,
        kInvalidArgument,
        kNotFound,
        kIoError,
        kCorruption,
        kBusy,
        kAborted,
        kInternal,
    };

    class Status
    {
    public:
        Status() = default;
        static Status Ok() { return Status(); }
        static Status InvalidArgument(std::string_view m) { return Status(ErrorCode::kInvalidArgument, m); }
        static Status NotFound(std::string_view m) { return Status(ErrorCode::kNotFound, m); }
        static Status IoError(std::string_view m) { return Status(ErrorCode::kIoError, m); }
        static Status Corruption(std::string_view m) { return Status(ErrorCode::kCorruption, m); }
        static Status Busy(std::string_view m) { return Status(ErrorCode::kBusy, m); }
        static Status Aborted(std::string_view m) { return Status(ErrorCode::kAborted, m); }
        static Status Internal(std::string_view m) { return Status(ErrorCode::kInternal, m); }

        bool ok() const noexcept { return code_ == ErrorCode::kOk; }
        ErrorCode code() const noexcept { return code_; }
        const std::string &message() const noexcept { return message_; }

    private:
        Status(ErrorCode c, std::string_view m) : code_(c), message_(m) {}
        ErrorCode code_ = ErrorCode::kOk;
        std::string message_;
    };

} // namespace pomai
