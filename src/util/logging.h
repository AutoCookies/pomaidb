#pragma once

#include <functional>
#include <string_view>

namespace pomai::util
{
    enum class LogLevel
    {
        kInfo,
        kWarn,
        kError,
    };

    using LogSink = std::function<void(LogLevel, std::string_view)>;

    void SetLogSink(LogSink sink);
    void Log(LogLevel level, std::string_view message);
} // namespace pomai::util
