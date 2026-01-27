#include "server/logger.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>

namespace pomai::server
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

    void Logger::Log(LogLevel lvl, const std::string &msg)
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (static_cast<int>(lvl) < static_cast<int>(level_))
            return;

        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);

        char timebuf[64];
        std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));

        std::string line = std::string(timebuf) + " [" + ToStr(lvl) + "] " + msg + "\n";

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

    void Logger::Debug(const std::string &msg) { Log(LogLevel::debug, msg); }
    void Logger::Info(const std::string &msg) { Log(LogLevel::info, msg); }
    void Logger::Warn(const std::string &msg) { Log(LogLevel::warn, msg); }
    void Logger::Error(const std::string &msg) { Log(LogLevel::error, msg); }

} // namespace pomai::server
