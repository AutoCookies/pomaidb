#ifndef POMAI_C_VERSION_H
#define POMAI_C_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define POMAI_C_ABI_VERSION_MAJOR 1u
#define POMAI_C_ABI_VERSION_MINOR 0u
#define POMAI_C_ABI_VERSION_PATCH 0u

#define POMAI_C_ABI_VERSION \
    ((POMAI_C_ABI_VERSION_MAJOR << 16u) | (POMAI_C_ABI_VERSION_MINOR << 8u) | POMAI_C_ABI_VERSION_PATCH)

// Returns packed ABI version using POMAI_C_ABI_VERSION encoding.
uint32_t pomai_abi_version(void);

// Returns PomaiDB engine version string.
const char* pomai_version_string(void);

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_VERSION_H
