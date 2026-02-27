#include "tests/common/test_main.h"
#include "util/logging.h"

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
