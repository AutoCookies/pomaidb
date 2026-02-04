#include "core/membrane/manager.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "core/engine/engine.h"
#include "storage/manifest/manifest.h"
#include "pomai/iterator.h"  // For SnapshotIterator


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
        if (st.code() == pomai::ErrorCode::kAlreadyExists)
        {
            // Already in manifest. Load valid spec and instantiate engine so we can Open it.
            pomai::MembraneSpec loaded_spec;
            st = storage::Manifest::GetMembrane(base_.path, spec.name, &loaded_spec);
            if (!st.ok()) return st;

            pomai::DBOptions opt = base_;
            opt.dim = loaded_spec.dim;
            opt.shard_count = loaded_spec.shard_count;
            opt.index_params = loaded_spec.index_params;
            opt.path = base_.path + "/membranes/" + spec.name;

            engines_.emplace(spec.name, std::make_unique<Engine>(opt));
            st = Status::Ok(); // clear error
        }
        else if (!st.ok())
        {
            return st;
        }

        st = OpenMembrane(kDefaultMembrane);
        if (!st.ok()) return st;

        // Restore other membranes from manifest
        std::vector<std::string> membranes;
        st = storage::Manifest::ListMembranes(base_.path, &membranes);
        if (!st.ok()) 
        {
             return st;
        }

        for (const auto& name : membranes)
        {
            if (name == kDefaultMembrane) continue;
            
            pomai::MembraneSpec mspec;
            st = storage::Manifest::GetMembrane(base_.path, name, &mspec);
            if (!st.ok()) return st;

            // Register engine in manager
            if (engines_.find(name) == engines_.end()) {
                pomai::DBOptions opt = base_;
                opt.dim = mspec.dim;
                opt.shard_count = mspec.shard_count;
                opt.index_params = mspec.index_params;
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

        // 1. Persist to Manifest
        // We use base_.path as the root_path for the DB.
        auto st = storage::Manifest::CreateMembrane(base_.path, spec);
        if (!st.ok()) return st;

        pomai::DBOptions opt = base_;
        opt.dim = spec.dim;
        opt.shard_count = spec.shard_count;
        opt.index_params = spec.index_params;

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

        // 1. Persist to Manifest
        auto st = storage::Manifest::DropMembrane(base_.path, name);
        if (!st.ok()) return st;

        // 2. Remove from Memory
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

    Status MembraneManager::PutBatch(std::string_view membrane,
                                     const std::vector<VectorId>& ids,
                                     const std::vector<std::span<const float>>& vectors)
    {
        auto *e = GetEngineOrNull(membrane);
        if (!e) return Status::NotFound("membrane not found");
        return e->PutBatch(ids, vectors);
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

    Status MembraneManager::NewIterator(std::string_view membrane, std::unique_ptr<pomai::SnapshotIterator> *out)
    {
        if (!out) return Status::InvalidArgument("out is null");
        auto *e = GetEngineOrNull(membrane);
        if (!e) return Status::NotFound("membrane not found");
        return e->NewIterator(out);
    }

} // namespace pomai::core
