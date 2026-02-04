#pragma once
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pomai/options.h"
#include "pomai/search.h"
#include "pomai/status.h"
#include "pomai/iterator.h"

namespace pomai::core
{

    class Engine;

    class MembraneManager
    {
    public:
        explicit MembraneManager(pomai::DBOptions base);
        ~MembraneManager();

        MembraneManager(const MembraneManager &) = delete;
        MembraneManager &operator=(const MembraneManager &) = delete;

        Status Open();
        Status Close();

        Status FlushAll();
        Status CloseAll();

        Status CreateMembrane(const pomai::MembraneSpec &spec);
        Status DropMembrane(std::string_view name);
        Status OpenMembrane(std::string_view name);
        Status CloseMembrane(std::string_view name);

        Status ListMembranes(std::vector<std::string> *out) const;

        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec);
        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out);
        Status Exists(std::string_view membrane, VectorId id, bool *exists);
        Status Delete(std::string_view membrane, VectorId id);
        Status Search(std::string_view membrane, std::span<const float> query, std::uint32_t topk, pomai::SearchResult *out);

        Status Freeze(std::string_view membrane);
        Status Compact(std::string_view membrane);
        Status NewIterator(std::string_view membrane, std::unique_ptr<pomai::SnapshotIterator> *out);

        // Default membrane convenience: use name "__default__"
        static constexpr std::string_view kDefaultMembrane = "__default__";

    private:
        Engine *GetEngineOrNull(std::string_view name) const;

        pomai::DBOptions base_;
        bool opened_ = false;

        // For now: keep engines in-memory; later you can add lazy-open by manifest.
        std::unordered_map<std::string, std::unique_ptr<Engine>> engines_;
    };

} // namespace pomai::core
