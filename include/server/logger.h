#pragma once
#include <mutex>
#include <string>

namespace pomai::server
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

        void Debug(const std::string &msg);
        void Info(const std::string &msg);
        void Warn(const std::string &msg);
        void Error(const std::string &msg);

    private:
        void Log(LogLevel lvl, const std::string &msg);
        static const char *ToStr(LogLevel lvl);

        std::mutex mu_;
        std::string file_path_;
        LogLevel level_{LogLevel::info};
    };

} // namespace pomai::server
