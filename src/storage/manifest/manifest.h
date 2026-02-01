#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "pomai/options.h"
#include "pomai/status.h"

namespace pomai::storage
{

    // Manifest describes membranes and their persistent configuration.
    // It is the source of truth on disk: "what membranes exist" + settings needed
    // to reopen safely.
    class Manifest
    {
    public:
        static pomai::Status EnsureInitialized(std::string_view root_path);

        static pomai::Status CreateMembrane(std::string_view root_path,
                                            const pomai::MembraneSpec &spec);

        static pomai::Status DropMembrane(std::string_view root_path,
                                          std::string_view name);

        static pomai::Status ListMembranes(std::string_view root_path,
                                           std::vector<std::string> *out);

        static pomai::Status GetMembrane(std::string_view root_path,
                                         std::string_view name,
                                         pomai::MembraneSpec *out);
    };

} // namespace pomai::storage
