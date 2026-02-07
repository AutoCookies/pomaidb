#include "pomai/c_status.h"

#include "capi_utils.h"

extern "C" {

pomai_status_t* pomai_status_ok(void) {
    return nullptr;
}

void pomai_status_free(pomai_status_t* status) {
    delete status;
}

int pomai_status_code(const pomai_status_t* status) {
    if (status == nullptr) {
        return static_cast<int>(POMAI_STATUS_OK);
    }
    return static_cast<int>(status->code);
}

const char* pomai_status_message(const pomai_status_t* status) {
    if (status == nullptr) {
        return "";
    }
    return status->message.c_str();
}

}  // extern "C"
