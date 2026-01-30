#pragma once

#include <cstdlib>
#include <cstdio>

namespace pomai
{
    inline void PomaiAssertFail(const char *expr, const char *file, int line, const char *msg)
    {
        if (msg)
            std::fprintf(stderr, "POMAI_ASSERT failed: %s (%s) at %s:%d\n", msg, expr, file, line);
        else
            std::fprintf(stderr, "POMAI_ASSERT failed: %s at %s:%d\n", expr, file, line);
        std::fflush(stderr);
        std::abort();
    }
}

#ifndef NDEBUG
#define POMAI_ASSERT(expr, msg) ((expr) ? (void)0 : ::pomai::PomaiAssertFail(#expr, __FILE__, __LINE__, msg))
#else
#define POMAI_ASSERT(expr, msg) ((void)0)
#endif
