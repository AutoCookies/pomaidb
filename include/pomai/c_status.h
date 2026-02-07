#ifndef POMAI_C_STATUS_H
#define POMAI_C_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "pomai/c_types.h"

// Status object used by all C API entrypoints.
// Convention: NULL status pointer means success (POMAI_STATUS_OK).
// Non-NULL means failure and must be released with pomai_status_free().
typedef struct pomai_status_t pomai_status_t;

typedef enum {
    POMAI_STATUS_OK = 0,
    POMAI_STATUS_INVALID_ARGUMENT = 1,
    POMAI_STATUS_NOT_FOUND = 2,
    POMAI_STATUS_ALREADY_EXISTS = 3,
    POMAI_STATUS_IO_ERROR = 4,
    POMAI_STATUS_CORRUPTION = 5,
    POMAI_STATUS_RESOURCE_EXHAUSTED = 6,
    POMAI_STATUS_DEADLINE_EXCEEDED = 7,
    POMAI_STATUS_INTERNAL = 8,
    POMAI_STATUS_UNIMPLEMENTED = 9,
    POMAI_STATUS_PARTIAL_FAILURE = 10,
} pomai_status_code_t;

// Returns NULL (success sentinel). Useful for wrappers wanting a uniform symbol.
POMAI_API pomai_status_t* pomai_status_ok(void);

// Releases a non-NULL status object. Safe for NULL.
POMAI_API void pomai_status_free(pomai_status_t* status);

// Returns status code; NULL => POMAI_STATUS_OK.
POMAI_API int pomai_status_code(const pomai_status_t* status);

// Returns UTF-8 error message; NULL => "".
POMAI_API const char* pomai_status_message(const pomai_status_t* status);

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_STATUS_H
