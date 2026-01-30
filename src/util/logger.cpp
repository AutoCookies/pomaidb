#include <pomai/util/logger.h>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>

namespace pomai
{
    void Logger::SetFile(std::string path)
    {
        std::lock_guard<std::mutex> lk(mu_);
        file_path_ = std::move(path);
    }

    void Logger::SetLevel(LogLevel lvl)
    {
        std::lock_guard<std::mutex> lk(mu_);
        level_ = lvl;
    }

    void Logger::EnableDebug(bool enabled)
    {
        std::lock_guard<std::mutex> lk(mu_);
        debug_enabled_ = enabled;
        if (enabled && level_ > LogLevel::debug)
            level_ = LogLevel::debug;
    }

    const char *Logger::ToStr(LogLevel lvl)
    {
        switch (lvl)
        {
        case LogLevel::debug:
            return "DEBUG";
        case LogLevel::info:
            return "INFO";
        case LogLevel::warn:
            return "WARN";
        case LogLevel::error:
            return "ERROR";
        }
        return "INFO";
    }

    void Logger::Log(LogLevel lvl, std::string_view event, std::string_view msg)
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (static_cast<int>(lvl) < static_cast<int>(level_))
            return;
        if (lvl == LogLevel::debug && !debug_enabled_)
            return;

        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        char timebuf[64];
        std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));

        std::string line = std::string(timebuf) + " [" + ToStr(lvl) + "] ";
        if (!event.empty())
        {
            line += "[";
            line += event;
            line += "] ";
        }
        line += msg;
        line += "\n";

        if (!file_path_.empty())
        {
            std::ofstream out(file_path_, std::ios::app);
            if (out)
                out << line;
            else
                std::cerr << line;
        }
        else
        {
            std::cerr << line;
        }
    }

    void Logger::Debug(std::string_view event, std::string_view msg) { Log(LogLevel::debug, event, msg); }
    void Logger::Info(std::string_view event, std::string_view msg) { Log(LogLevel::info, event, msg); }
    void Logger::Warn(std::string_view event, std::string_view msg) { Log(LogLevel::warn, event, msg); }
    void Logger::Error(std::string_view event, std::string_view msg) { Log(LogLevel::error, event, msg); }
} // namespace pomai
