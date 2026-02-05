#include "pomai/c_status.h"
#include "pomai/status.h"
#include "capi_utils.h"

extern "C" {

void pomai_status_free(pomai_status_t* status) {
    if (status) {
        delete FromCStatusMutable(status);
    }
}

int pomai_status_code(const pomai_status_t* status) {
    if (!status) return 0; // POMAI_OK
    return static_cast<int>(FromCStatus(status)->code());
}

const char* pomai_status_message(const pomai_status_t* status) {
    if (!status) return "";
    return FromCStatus(status)->message().c_str();
}

} // extern "C"
