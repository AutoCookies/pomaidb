#pragma once
#include "pomai/c_api.h"
#include "pomai/status.h"

// Internal helpers for C API implementation

inline pomai_status_t* ToCStatus(pomai::Status st) {
    if (st.ok()) return nullptr;
    return reinterpret_cast<pomai_status_t*>(new pomai::Status(std::move(st)));
}

inline const pomai::Status* FromCStatus(const pomai_status_t* s) {
    return reinterpret_cast<const pomai::Status*>(s);
}

inline pomai::Status* FromCStatusMutable(pomai_status_t* s) {
    return reinterpret_cast<pomai::Status*>(s);
}
