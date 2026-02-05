#pragma once
#include <string>
#include <string_view>

namespace pomai
{

    enum class ErrorCode : int
    {
        kOk = 0,

        kInvalidArgument,
        kNotFound,
        kAlreadyExists,
        kPermissionDenied,

        kResourceExhausted,
        kFailedPrecondition,
        kAborted,

        kIO, // I/O error
        kInternal,
        kPartial, // Partial failure (some shards failed)
        kUnknown,
    };

    class Status
    {
    public:
        Status() : code_(ErrorCode::kOk) {}
        explicit Status(ErrorCode code, std::string msg = {})
            : code_(code), msg_(std::move(msg)) {}

        bool ok() const noexcept { return code_ == ErrorCode::kOk; }
        ErrorCode code() const noexcept { return code_; }
        const std::string &message() const noexcept { return msg_; }

        // ---- Canonical factories (bigtech style) ----
        static Status Ok() { return Status(); }

        static Status InvalidArgument(std::string_view m)
        {
            return Status(ErrorCode::kInvalidArgument, std::string(m));
        }
        static Status NotFound(std::string_view m)
        {
            return Status(ErrorCode::kNotFound, std::string(m));
        }
        static Status AlreadyExists(std::string_view m)
        {
            return Status(ErrorCode::kAlreadyExists, std::string(m));
        }
        static Status PermissionDenied(std::string_view m)
        {
            return Status(ErrorCode::kPermissionDenied, std::string(m));
        }
        static Status ResourceExhausted(std::string_view m)
        {
            return Status(ErrorCode::kResourceExhausted, std::string(m));
        }
        static Status FailedPrecondition(std::string_view m)
        {
            return Status(ErrorCode::kFailedPrecondition, std::string(m));
        }
        static Status Aborted(std::string_view m)
        {
            return Status(ErrorCode::kAborted, std::string(m));
        }

        // Some codepaths call Busy()/Corruption()/IOError(); keep compatibility.
        static Status Busy(std::string_view m) { return Aborted(m); }

        static Status Corruption(std::string_view m)
        {
            // In embedded DBs, corruption is often treated as aborted/failed-precondition.
            return Status(ErrorCode::kAborted, std::string(m));
        }

        static Status IOError(std::string_view m)
        {
            return Status(ErrorCode::kIO, std::string(m));
        }
        static Status IoError(std::string_view m) { return IOError(m); }

        static Status Internal(std::string_view m)
        {
            return Status(ErrorCode::kInternal, std::string(m));
        }
        static Status Partial(std::string_view m)
        {
            return Status(ErrorCode::kPartial, std::string(m));
        }
        static Status Unknown(std::string_view m)
        {
            return Status(ErrorCode::kUnknown, std::string(m));
        }

    private:
        ErrorCode code_;
        std::string msg_;
    };

} // namespace pomai
