#pragma once
#include <string>
#include <utility>

namespace pomai
{

    enum class StatusCode
    {
        Ok = 0,
        Busy,
        InvalidArgument,
        NotFound,
        IOError,
        Internal,
    };

    struct Status
    {
        StatusCode code{StatusCode::Ok};
        std::string message;

        static Status OK() { return {StatusCode::Ok, ""}; }
        static Status Busy(std::string msg = "busy") { return {StatusCode::Busy, std::move(msg)}; }
        static Status Invalid(std::string msg) { return {StatusCode::InvalidArgument, std::move(msg)}; }

        // --- THÊM DÒNG NÀY ---
        static Status NotFound(std::string msg) { return {StatusCode::NotFound, std::move(msg)}; }
        // ---------------------

        static Status IO(std::string msg) { return {StatusCode::IOError, std::move(msg)}; }
        static Status Internal(std::string msg) { return {StatusCode::Internal, std::move(msg)}; }

        bool ok() const { return code == StatusCode::Ok; }
    };

    template <class T>
    struct Result
    {
        Status status;
        T value;

        static Result<T> Ok(T v) { return {Status::OK(), std::move(v)}; }
        static Result<T> Err(Status s) { return {std::move(s), T{}}; }
    };

} // namespace pomai