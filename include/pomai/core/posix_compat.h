#pragma once

#include <fcntl.h>
#include <unistd.h>

#if defined(_WIN32)
#include <io.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif

inline int fdatasync(int fd)
{
    return _commit(fd);
}

inline int fsync(int fd)
{
    return _commit(fd);
}
#endif
