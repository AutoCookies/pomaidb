#include "tests/common/test_main.h"
#include "util/logging.h"
#include <string>

static std::string g_fatal_message;

static void NoAbortFatalHandler(const std::string& message) {
    g_fatal_message = message;
}

POMAI_TEST(Logging_Basic) {
    // This test primarily verifies that the logging macros compile and run.
    // Visual verification of colors/timestamps is manual from stdout.
    
    pomai::util::Logger::Instance().SetLevel(pomai::util::LogLevel::kDebug);
    
    POMAI_LOG_DEBUG("This is a DEBUG message with arg: {}", 42);
    POMAI_LOG_INFO("This is an INFO message with string: {}", "pomai");
    POMAI_LOG_WARN("This is a WARN message");
    POMAI_LOG_ERROR("This is an ERROR message");
    
    // Test level filtering
    pomai::util::Logger::Instance().SetLevel(pomai::util::LogLevel::kWarn);
    POMAI_LOG_INFO("This should NOT be visible");
    POMAI_LOG_WARN("This SHOULD be visible");
}

POMAI_TEST(Logging_FatalHandler) {
    // Verify configurable fatal handler: use a handler that does not abort.
    pomai::util::FatalHandler prev = pomai::util::GetFatalHandler();
    pomai::util::SetFatalHandler(NoAbortFatalHandler);
    g_fatal_message.clear();
    pomai::util::Logger::Instance().SetLevel(pomai::util::LogLevel::kFatal);
    POMAI_LOG_FATAL("Fatal test message {}", 123);
    POMAI_EXPECT_TRUE(!g_fatal_message.empty());
    POMAI_EXPECT_TRUE(g_fatal_message.find("123") != std::string::npos);
    pomai::util::SetFatalHandler(prev);
}
