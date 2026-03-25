#include "pomai/c_api.h"

#include "pomai/membrane_capabilities.h"

#include "capi_utils.h"

extern "C" {

void pomai_membrane_capabilities_init(pomai_membrane_capabilities_t* caps) {
    if (caps == nullptr) return;
    caps->struct_size = static_cast<uint32_t>(sizeof(pomai_membrane_capabilities_t));
    caps->kind = POMAI_MEMBRANE_KIND_VECTOR;
    caps->stability = POMAI_MEMBRANE_STABILITY_STABLE;
    caps->reserved0 = 0;
    caps->reserved1 = 0;
    caps->read_path = false;
    caps->write_path = false;
    caps->unified_scan = false;
    caps->snapshot_isolated_scan = false;
}

pomai_status_t* pomai_membrane_kind_capabilities(uint8_t kind, pomai_membrane_capabilities_t* out_caps) {
    if (out_caps == nullptr) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "out_caps must be non-null");
    }
    if (out_caps->struct_size < sizeof(pomai_membrane_capabilities_t)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "membrane_capabilities.struct_size is too small");
    }
    if (!pomai::IsValidMembraneKindValue(kind)) {
        return MakeStatus(POMAI_STATUS_INVALID_ARGUMENT, "invalid membrane kind value");
    }
    const auto k = static_cast<pomai::MembraneKind>(kind);
    const pomai::MembraneKindCapabilities c = pomai::GetMembraneKindCapabilities(k);
    out_caps->kind = kind;
    out_caps->stability = (c.stability == pomai::MembraneStability::kStable) ? POMAI_MEMBRANE_STABILITY_STABLE
                                                                             : POMAI_MEMBRANE_STABILITY_EXPERIMENTAL;
    out_caps->reserved0 = 0;
    out_caps->reserved1 = 0;
    out_caps->read_path = c.read_path;
    out_caps->write_path = c.write_path;
    out_caps->unified_scan = c.unified_scan;
    out_caps->snapshot_isolated_scan = c.snapshot_isolated_scan;
    return nullptr;
}

} // extern "C"
