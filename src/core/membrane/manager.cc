#include "core/membrane/manager.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "core/engine/engine.h"
#include "storage/manifest/manifest.h"


namespace pomai::core
{

    MembraneManager::MembraneManager(pomai::DBOptions base) : base_(std::move(base)) {}
    MembraneManager::~MembraneManager() = default;

    Status MembraneManager::Open()
    {
        opened_ = true;

        // Ensure default membrane exists and is opened.
        pomai::MembraneSpec spec;
        spec.name = std::string(kDefaultMembrane);
        spec.dim = base_.dim;
        spec.shard_count = base_.shard_count;

        auto st = CreateMembrane(spec);
        if (!st.ok() && st.code() != pomai::ErrorCode::kAlreadyExists)
            return st;

        st = OpenMembrane(kDefaultMembrane);
        if (!st.ok()) return st;

        // Restore other membranes from manifest
        std::vector<std::string> membranes;
        st = storage::Manifest::ListMembranes(base_.path, &membranes);
        if (!st.ok()) 
        {
             // If manifest corrupted or missing, we might want to fail hard?
             // Since we just created default membrane, EnsureInitialized must have run.
             return st;
        }

        for (const auto& name : membranes)
        {
            if (name == kDefaultMembrane) continue;
            
            // "Create" in memory without checking manifest (already checked via ListMembranes)
            // But we can use the existing CreateMembrane logic which re-loads manifest?
            // Actually, CreateMembrane checks if manifest exists.
            // Let's use GetMembrane to get spec, then create in-memory engine, then open.
            
            pomai::MembraneSpec mspec;
            st = storage::Manifest::GetMembrane(base_.path, name, &mspec);
            if (!st.ok()) return st;

            // Register engine in manager (in-memory)
            if (engines_.find(name) == engines_.end()) {
                pomai::DBOptions opt = base_;
                opt.dim = mspec.dim;
                opt.shard_count = mspec.shard_count;
                opt.path = base_.path + "/membranes/" + name;
                engines_.emplace(name, std::make_unique<Engine>(opt));
            }

            st = OpenMembrane(name);
            if (!st.ok()) return st;
        }

        return Status::Ok();
    }

    Status MembraneManager::Close()
    {
        return CloseAll();
    }

    Status MembraneManager::FlushAll()
    {
        for (auto &kv : engines_)
        {
            auto st = kv.second->Flush();
            if (!st.ok())
                return st;
        }
        return Status::Ok();
    }

    Status MembraneManager::CloseAll()
    {
        for (auto &kv : engines_)
            (void)kv.second->Close();
        engines_.clear();
        opened_ = false;
        return Status::Ok();
    }

    Engine *MembraneManager::GetEngineOrNull(std::string_view name) const
    {
        auto it = engines_.find(std::string(name));
        if (it == engines_.end())
            return nullptr;
        return it->second.get();
    }

    Status MembraneManager::CreateMembrane(const pomai::MembraneSpec &spec)
    {
        if (spec.name.empty())
            return Status::InvalidArgument("membrane name empty");
        if (spec.dim == 0)
            return Status::InvalidArgument("membrane dim must be > 0");
        if (spec.shard_count == 0)
            return Status::InvalidArgument("membrane shard_count must be > 0");

        if (engines_.find(spec.name) != engines_.end())
            return Status::AlreadyExists("membrane already exists");

        pomai::DBOptions opt = base_;
        opt.dim = spec.dim;
        opt.shard_count = spec.shard_count;

        // Keep simple on-disk layout (no manifest integration yet here).
        opt.path = base_.path + "/membranes/" + spec.name;

        engines_.emplace(spec.name, std::make_unique<Engine>(opt));
        return Status::Ok();
    }

    Status MembraneManager::DropMembrane(std::string_view name)
    {
        auto it = engines_.find(std::string(name));
        if (it == engines_.end())
            return Status::NotFound("membrane not found");
        (void)it->second->Close();
        engines_.erase(it);
        return Status::Ok();
    }

    Status MembraneManager::OpenMembrane(std::string_view name)
    {
        auto *e = GetEngineOrNull(name);
        if (!e)
            return Status::NotFound("membrane not found");
        return e->Open();
    }

    Status MembraneManager::CloseMembrane(std::string_view name)
    {
        auto *e = GetEngineOrNull(name);
        if (!e)
            return Status::NotFound("membrane not found");
        return e->Close();
    }

    Status MembraneManager::ListMembranes(std::vector<std::string> *out) const
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        out->clear();
        out->reserve(engines_.size());
        for (const auto &kv : engines_)
            out->push_back(kv.first);
        std::sort(out->begin(), out->end());
        return Status::Ok();
    }

    Status MembraneManager::Put(std::string_view membrane, VectorId id, std::span<const float> vec)
    {
        auto *e = GetEngineOrNull(membrane);
        if (!e)
            return Status::NotFound("membrane not found");
        return e->Put(id, vec);
    }

    Status MembraneManager::Get(std::string_view membrane, VectorId id, std::vector<float> *out)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *e = GetEngineOrNull(membrane);
        if (!e)
            return Status::NotFound("membrane not found");
        return e->Get(id, out);
    }

    Status MembraneManager::Exists(std::string_view membrane, VectorId id, bool *exists)
    {
        if (!exists)
            return Status::InvalidArgument("exists is null");
        auto *e = GetEngineOrNull(membrane);
        if (!e)
            return Status::NotFound("membrane not found");
        return e->Exists(id, exists);
    }

    Status MembraneManager::Delete(std::string_view membrane, VectorId id)
    {
        auto *e = GetEngineOrNull(membrane);
        if (!e)
            return Status::NotFound("membrane not found");
        return e->Delete(id);
    }

    Status MembraneManager::Search(std::string_view membrane, std::span<const float> query,
                                   std::uint32_t topk, pomai::SearchResult *out)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *e = GetEngineOrNull(membrane);
        if (!e)
            return Status::NotFound("membrane not found");
        return e->Search(query, topk, out);
    }

    Status MembraneManager::Freeze(std::string_view membrane)
    {
        auto *e = GetEngineOrNull(membrane);
        if (!e) return Status::NotFound("membrane not found");
        return e->Freeze();
    }

    Status MembraneManager::Compact(std::string_view membrane)
    {
        auto *e = GetEngineOrNull(membrane);
        if (!e) return Status::NotFound("membrane not found");
        return e->Compact();
    }

} // namespace pomai::core
