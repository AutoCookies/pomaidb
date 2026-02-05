#ifndef POMAI_C_STATUS_H
#define POMAI_C_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Opaque status object.
// If a function returns pomai_status_t*, NULL indicates success (POMAI_OK).
// If not NULL, it indicates an error and must be freed by pomai_status_free().
typedef struct pomai_status_t pomai_status_t;

typedef enum {
    POMAI_OK = 0,
    POMAI_INVALID_ARGUMENT = 1,
    POMAI_NOT_FOUND = 2,
    POMAI_ALREADY_EXISTS = 3,
    POMAI_PERMISSION_DENIED = 4,
    POMAI_RESOURCE_EXHAUSTED = 5,
    POMAI_FAILED_PRECONDITION = 6,
    POMAI_ABORTED = 7,
    POMAI_IO_ERROR = 8,
    POMAI_INTERNAL = 9,
    POMAI_PARTIAL_FAILURE = 10,
    POMAI_UNKNOWN = 11,
} pomai_error_code_t;

// Frees the status object. Safe to call with NULL.
void pomai_status_free(pomai_status_t* status);

// Returns the error code from the status object.
int pomai_status_code(const pomai_status_t* status);

// Returns the error message. The string is owned by the status object.
const char* pomai_status_message(const pomai_status_t* status);

#ifdef __cplusplus
}
#endif

#endif // POMAI_C_STATUS_H
