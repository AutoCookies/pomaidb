#include "util/logging.h"
#include <iostream>
#include <iomanip>
#include <filesystem>

namespace pomai::util
{
    Logger::Logger() : min_level_(LogLevel::kWarn)
    {
        const char* env = std::getenv("POMAI_LOG_LEVEL");
        if (env) {
            std::string s(env);
            if (s == "DEBUG") min_level_ = LogLevel::kDebug;
            else if (s == "INFO")  min_level_ = LogLevel::kInfo;
            else if (s == "WARN")  min_level_ = LogLevel::kWarn;
            else if (s == "ERROR") min_level_ = LogLevel::kError;
            else if (s == "FATAL") min_level_ = LogLevel::kFatal;
        }
    }

    Logger& Logger::Instance()
    {
        static Logger instance;
        return instance;
    }

    void Logger::SetLevel(LogLevel level)
    {
        min_level_ = level;
    }

    LogLevel Logger::GetLevel() const
    {
        return min_level_;
    }

    void Logger::Write(LogLevel level, std::source_location loc, const std::string& message)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // 1. Timestamp
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::cout << "[" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S") 
                  << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";

        // 2. Level with Color
        const char* color = "";
        const char* level_str = "";
        switch (level)
        {
            case LogLevel::kDebug: color = "\033[36m"; level_str = "DEBUG"; break; // Cyan
            case LogLevel::kInfo:  color = "\033[32m"; level_str = "INFO "; break; // Green
            case LogLevel::kWarn:  color = "\033[33m"; level_str = "WARN "; break; // Yellow
            case LogLevel::kError: color = "\033[31m"; level_str = "ERROR"; break; // Red
            case LogLevel::kFatal: color = "\033[41m"; level_str = "FATAL"; break; // Red Background
        }
        std::cout << color << level_str << "\033[0m ";

        // 3. Location
        std::filesystem::path p(loc.file_name());
        std::cout << "[" << p.filename().string() << ":" << loc.line() << "] ";

        // 4. Message
        std::cout << message << std::endl;
        
        if (level == LogLevel::kFatal) std::abort();
    }

    // Compat for older calls if any (manual Log function was in之前的 util/logging.h)
    // We already updated the header, but if any .cc still calls the old Log(level, msg)
    // we should bridge it or fix them.
}
