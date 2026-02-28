#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <iostream>

namespace pomai {

enum class ErrorCode : uint8_t {
    kOk = 0,
    kInvalidArgument = 1,
    kNotFound = 2,
    kAlreadyExists = 3,
    kPermissionDenied = 4,
    kResourceExhausted = 5,
    kFailedPrecondition = 6,
    kAborted = 7,
    kIO = 8,
    kInternal = 9,
    kPartial = 10,
    kCorruption = 11,
    kUnknown = 255,
};

/**
 * Status: A compact, stack-friendly representation of an operation result.
 * Avoids heap allocation for most cases by using static messages or small representations.
 */
class [[nodiscard]] Status {
 public:
  Status() noexcept : code_(ErrorCode::kOk), subcode_(0), state_(nullptr) {}
  
  Status(ErrorCode code, std::string_view msg = "") noexcept
      : code_(code), subcode_(0), state_(msg.empty() ? nullptr : msg.data()) {
      // Note: This assumes msg points to a static string or outlives Status.
      // For a truly robust Lean DB, we prefer static messages.
  }

  bool ok() const noexcept { return code_ == ErrorCode::kOk; }
  ErrorCode code() const noexcept { return code_; }
  
  const char* message() const noexcept {
    return state_ ? state_ : "";
  }

  // --- Canonical Factories ---
  static Status Ok() noexcept { return Status(); }
  static Status NotFound(std::string_view m = "Not Found") noexcept { return Status(ErrorCode::kNotFound, m); }
  static Status InvalidArgument(std::string_view m) noexcept { return Status(ErrorCode::kInvalidArgument, m); }
  static Status IOError(std::string_view m) noexcept { return Status(ErrorCode::kIO, m); }
  static Status Corruption(std::string_view m) noexcept { return Status(ErrorCode::kCorruption, m); }
  static Status Internal(std::string_view m) noexcept { return Status(ErrorCode::kInternal, m); }
  static Status NotSupported(std::string_view m = "Not Supported") noexcept { return Status(ErrorCode::kAborted, m); }
  static Status Aborted(std::string_view m = "Aborted") noexcept { return Status(ErrorCode::kAborted, m); }
  static Status Deleted(std::string_view m = "Deleted") noexcept { return Status(ErrorCode::kNotFound, m); }
  static Status ResourceExhausted(std::string_view m) noexcept { return Status(ErrorCode::kResourceExhausted, m); }
  static Status AlreadyExists(std::string_view m = "Already Exists") noexcept { return Status(ErrorCode::kAlreadyExists, m); }

  // Backward compatibility aliases
  static Status IoError(std::string_view m) noexcept { return IOError(m); }
  static Status Busy(std::string_view m) noexcept { return Status(ErrorCode::kAborted, m); }

  std::string ToString() const {
    if (ok()) return "OK";
    std::string s = "Error: ";
    s += std::to_string(static_cast<int>(code_));
    if (state_) {
        s += " (";
        s += state_;
        s += ")";
    }
    return s;
  }

 private:
  ErrorCode code_;
  uint8_t subcode_;
  const char* state_; // Points to a static string or owned state (if extended)
};

} // namespace pomai
