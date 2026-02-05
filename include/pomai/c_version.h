#ifndef POMAI_C_VERSION_H
#define POMAI_C_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "pomai/c_types.h"

#define POMAI_ABI_VERSION_MAJOR 1u
#define POMAI_ABI_VERSION_MINOR 1u
#define POMAI_ABI_VERSION_PATCH 0u

#define POMAI_C_ABI_VERSION_MAJOR POMAI_ABI_VERSION_MAJOR
#define POMAI_C_ABI_VERSION_MINOR POMAI_ABI_VERSION_MINOR
#define POMAI_C_ABI_VERSION_PATCH POMAI_ABI_VERSION_PATCH

#define POMAI_C_ABI_VERSION \
    ((POMAI_C_ABI_VERSION_MAJOR << 16u) | (POMAI_C_ABI_VERSION_MINOR << 8u) | POMAI_C_ABI_VERSION_PATCH)

#define POMAI_ABI_VERSION \
    ((POMAI_ABI_VERSION_MAJOR << 16u) | (POMAI_ABI_VERSION_MINOR << 8u) | POMAI_ABI_VERSION_PATCH)

// Returns packed ABI version using POMAI_C_ABI_VERSION encoding.
POMAI_API uint32_t pomai_abi_version(void);

// Returns PomaiDB engine version string.
POMAI_API const char* pomai_version_string(void);

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_VERSION_H
