#pragma once

#include <mutex>
#include <string>
#include <string_view>

namespace pomai
{
    enum class LogLevel
    {
        debug = 0,
        info = 1,
        warn = 2,
        error = 3
    };

    class Logger
    {
    public:
        Logger() = default;

        void SetFile(std::string path);
        void SetLevel(LogLevel lvl);
        void EnableDebug(bool enabled);

        void Debug(std::string_view event, std::string_view msg);
        void Info(std::string_view event, std::string_view msg);
        void Warn(std::string_view event, std::string_view msg);
        void Error(std::string_view event, std::string_view msg);

    private:
        void Log(LogLevel lvl, std::string_view event, std::string_view msg);
        static const char *ToStr(LogLevel lvl);

        std::mutex mu_;
        std::string file_path_;
        LogLevel level_{LogLevel::info};
        bool debug_enabled_{false};
    };
} // namespace pomai
