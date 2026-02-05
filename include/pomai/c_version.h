#ifndef POMAI_C_VERSION_H
#define POMAI_C_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

#include "pomai/version.h"

// ABI Versioning
#define POMAI_C_ABI_VERSION 1

// Returns a version string like "0.1.0"
const char* pomai_version_string();

// Returns the ABI version (monotonic integer)
unsigned int pomai_abi_version();

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_VERSION_H
