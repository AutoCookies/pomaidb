#pragma once

#include <memory>
#include <string>

#include "pomai/c_status.h"
#include "pomai/pomai.h"
#include "pomai/snapshot.h"
#include "pomai/iterator.h"

struct pomai_status_t {
    pomai_status_code_t code;
    std::string message;
};

struct pomai_db_t {
    std::unique_ptr<pomai::DB> db;
};

struct pomai_snapshot_t {
    std::shared_ptr<pomai::Snapshot> snap;
};

struct pomai_iter_t {
    std::unique_ptr<pomai::SnapshotIterator> iter;
};

inline pomai_status_code_t ToCCode(pomai::ErrorCode code) {
    switch (code) {
        case pomai::ErrorCode::kOk:
            return POMAI_STATUS_OK;
        case pomai::ErrorCode::kInvalidArgument:
            return POMAI_STATUS_INVALID_ARGUMENT;
        case pomai::ErrorCode::kNotFound:
            return POMAI_STATUS_NOT_FOUND;
        case pomai::ErrorCode::kAlreadyExists:
            return POMAI_STATUS_ALREADY_EXISTS;
        case pomai::ErrorCode::kResourceExhausted:
            return POMAI_STATUS_RESOURCE_EXHAUSTED;
        case pomai::ErrorCode::kIO:
            return POMAI_STATUS_IO_ERROR;
        case pomai::ErrorCode::kPartial:
            return POMAI_STATUS_PARTIAL_FAILURE;
        case pomai::ErrorCode::kAborted:
            return POMAI_STATUS_CORRUPTION;
        case pomai::ErrorCode::kPermissionDenied:
        case pomai::ErrorCode::kFailedPrecondition:
        case pomai::ErrorCode::kUnknown:
            return POMAI_STATUS_INTERNAL;
        case pomai::ErrorCode::kInternal:
            return POMAI_STATUS_INTERNAL;
    }
    return POMAI_STATUS_INTERNAL;
}

inline pomai_status_t* MakeStatus(pomai_status_code_t code, std::string message) {
    if (code == POMAI_STATUS_OK) {
        return nullptr;
    }
    return new pomai_status_t{code, std::move(message)};
}

inline pomai_status_t* ToCStatus(const pomai::Status& st) {
    if (st.ok()) {
        return nullptr;
    }
    return MakeStatus(ToCCode(st.code()), st.message());
}
