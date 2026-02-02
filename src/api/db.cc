#include "pomai/pomai.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/membrane/manager.h"

namespace pomai
{

    class DbImpl final : public DB
    {
    public:
        explicit DbImpl(DBOptions opt) : mgr_(std::move(opt))
        {
            // Keep behavior simple: open manager (creates default membrane if needed).
            (void)mgr_.Open();
        }

        // ---- DB lifetime ----
        Status Flush() override { return mgr_.FlushAll(); }
        Status Close() override { return mgr_.CloseAll(); }

        // ---- Default membrane convenience ----
        Status Put(VectorId id, std::span<const float> vec) override
        {
            return mgr_.Put(core::MembraneManager::kDefaultMembrane, id, vec);
        }

        Status Get(VectorId id, std::vector<float> *out) override
        {
            return mgr_.Get(core::MembraneManager::kDefaultMembrane, id, out);
        }

        Status Exists(VectorId id, bool *exists) override
        {
            return mgr_.Exists(core::MembraneManager::kDefaultMembrane, id, exists);
        }

        Status Delete(VectorId id) override
        {
            return mgr_.Delete(core::MembraneManager::kDefaultMembrane, id);
        }

        Status Search(std::span<const float> query, uint32_t topk, SearchResult *out) override
        {
            return mgr_.Search(core::MembraneManager::kDefaultMembrane, query, topk, out);
        }

        // ---- Membrane API ----
        Status CreateMembrane(const MembraneSpec &spec) override
        {
            return mgr_.CreateMembrane(spec);
        }

        Status DropMembrane(std::string_view name) override
        {
            return mgr_.DropMembrane(name);
        }

        Status OpenMembrane(std::string_view name) override
        {
            return mgr_.OpenMembrane(name);
        }

        Status CloseMembrane(std::string_view name) override
        {
            return mgr_.CloseMembrane(name);
        }

        Status ListMembranes(std::vector<std::string> *out) const override
        {
            return mgr_.ListMembranes(out);
        }

        // ---- Membrane-scoped operations ----
        Status Put(std::string_view membrane, VectorId id, std::span<const float> vec) override
        {
            return mgr_.Put(membrane, id, vec);
        }

        Status Get(std::string_view membrane, VectorId id, std::vector<float> *out) override
        {
            return mgr_.Get(membrane, id, out);
        }

        Status Exists(std::string_view membrane, VectorId id, bool *exists) override
        {
            return mgr_.Exists(membrane, id, exists);
        }

        Status Delete(std::string_view membrane, VectorId id) override
        {
            return mgr_.Delete(membrane, id);
        }

        Status Search(std::string_view membrane, std::span<const float> query,
                      uint32_t topk, SearchResult *out) override
        {
            return mgr_.Search(membrane, query, topk, out);
        }

    private:
        core::MembraneManager mgr_;
    };

    Status DB::Open(const DBOptions &options, std::unique_ptr<DB> *out)
    {
        if (!out)
            return Status::InvalidArgument("out=null");
        *out = std::make_unique<DbImpl>(options);
        return Status::Ok();
    }

} // namespace pomai
