#pragma once

#include <string_view>
#include <format>
#include <source_location>
#include <iostream>
#include <chrono>
#include <mutex>

namespace pomai::util
{
    enum class LogLevel
    {
        kDebug,
        kInfo,
        kWarn,
        kError,
        kFatal
    };

    /**
     * @brief Premium Logger for PomaiDB.
     * Use POMAI_LOG_* macros for automatic file/line capture.
     */
    class Logger
    {
    public:
        static Logger& Instance();

        void SetLevel(LogLevel level);
        LogLevel GetLevel() const;

        template <typename... Args>
        void Log(LogLevel level, std::source_location loc, std::string_view fmt, Args&&... args)
        {
            if (level < min_level_) return;

            try {
                std::string message = std::vformat(fmt, std::make_format_args(args...));
                Write(level, loc, message);
            } catch (...) {
                // Fallback if formatting fails
                Write(level, loc, std::string(fmt));
            }
        }

    private:
        Logger();
        void Write(LogLevel level, std::source_location loc, const std::string& message);

        LogLevel min_level_;
        std::mutex mutex_;
    };

} // namespace pomai::util

// ── Macros ──────────────────────────────────────────────────────────────────

#define POMAI_LOG_DEBUG(fmt, ...) \
    ::pomai::util::Logger::Instance().Log(::pomai::util::LogLevel::kDebug, std::source_location::current(), fmt, ##__VA_ARGS__)

#define POMAI_LOG_INFO(fmt, ...) \
    ::pomai::util::Logger::Instance().Log(::pomai::util::LogLevel::kInfo, std::source_location::current(), fmt, ##__VA_ARGS__)

#define POMAI_LOG_WARN(fmt, ...) \
    ::pomai::util::Logger::Instance().Log(::pomai::util::LogLevel::kWarn, std::source_location::current(), fmt, ##__VA_ARGS__)

#define POMAI_LOG_ERROR(fmt, ...) \
    ::pomai::util::Logger::Instance().Log(::pomai::util::LogLevel::kError, std::source_location::current(), fmt, ##__VA_ARGS__)

#define POMAI_LOG_FATAL(fmt, ...) \
    ::pomai::util::Logger::Instance().Log(::pomai::util::LogLevel::kFatal, std::source_location::current(), fmt, ##__VA_ARGS__)
