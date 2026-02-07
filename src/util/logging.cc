#include "util/logging.h"

#include <mutex>

namespace pomai::util
{
    namespace
    {
        std::mutex log_mutex;
        LogSink log_sink;
    }

    void SetLogSink(LogSink sink)
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        log_sink = std::move(sink);
    }

    void Log(LogLevel level, std::string_view message)
    {
        LogSink sink;
        {
            std::lock_guard<std::mutex> lock(log_mutex);
            sink = log_sink;
        }
        if (sink)
        {
            sink(level, message);
        }
    }
} // namespace pomai::util
