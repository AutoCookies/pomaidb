#include "core/membrane/manager.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "core/engine/engine.h"
#include "storage/manifest/manifest.h"
#include "pomai/iterator.h"  // For SnapshotIterator
#include "core/rag/engine.h"


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

            MembraneState state;
            state.spec = loaded_spec;
            if (loaded_spec.kind == pomai::MembraneKind::kVector) {
                state.vector_engine = std::make_unique<Engine>(opt, loaded_spec.kind);
            } else {
                state.rag_engine = std::make_unique<RagEngine>(opt, loaded_spec);
            }
            membranes_.emplace(spec.name, std::move(state));
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
            if (membranes_.find(name) == membranes_.end()) {
                pomai::DBOptions opt = base_;
                opt.dim = mspec.dim;
                opt.shard_count = mspec.shard_count;
                opt.index_params = mspec.index_params;
                opt.path = base_.path + "/membranes/" + name;

                MembraneState state;
                state.spec = mspec;
                if (mspec.kind == pomai::MembraneKind::kVector) {
                    state.vector_engine = std::make_unique<Engine>(opt, mspec.kind);
                } else {
                    state.rag_engine = std::make_unique<RagEngine>(opt, mspec);
                }
                membranes_.emplace(name, std::move(state));
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
        for (auto &kv : membranes_)
        {
            if (kv.second.vector_engine) {
                auto st = kv.second.vector_engine->Flush();
                if (!st.ok())
                    return st;
            }
        }
        return Status::Ok();
    }

    Status MembraneManager::CloseAll()
    {
        for (auto &kv : membranes_) {
            if (kv.second.vector_engine) {
                (void)kv.second.vector_engine->Close();
            }
            if (kv.second.rag_engine) {
                (void)kv.second.rag_engine->Close();
            }
        }
        membranes_.clear();
        opened_ = false;
        return Status::Ok();
    }

    MembraneManager::MembraneState *MembraneManager::GetMembraneOrNull(std::string_view name)
    {
        auto it = membranes_.find(std::string(name));
        if (it == membranes_.end())
            return nullptr;
        return &it->second;
    }

    const MembraneManager::MembraneState *MembraneManager::GetMembraneOrNull(std::string_view name) const
    {
        auto it = membranes_.find(std::string(name));
        if (it == membranes_.end())
            return nullptr;
        return &it->second;
    }

    Status MembraneManager::CreateMembrane(const pomai::MembraneSpec &spec)
    {
        if (spec.name.empty())
            return Status::InvalidArgument("membrane name empty");
        if (spec.dim == 0)
            return Status::InvalidArgument("membrane dim must be > 0");
        if (spec.shard_count == 0)
            return Status::InvalidArgument("membrane shard_count must be > 0");

        if (membranes_.find(spec.name) != membranes_.end())
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

        MembraneState state;
        state.spec = spec;
        if (spec.kind == pomai::MembraneKind::kVector) {
            state.vector_engine = std::make_unique<Engine>(opt, spec.kind);
        } else {
            state.rag_engine = std::make_unique<RagEngine>(opt, spec);
        }
        membranes_.emplace(spec.name, std::move(state));
        return Status::Ok();
    }

    Status MembraneManager::DropMembrane(std::string_view name)
    {
        auto it = membranes_.find(std::string(name));
        if (it == membranes_.end())
            return Status::NotFound("membrane not found");

        // 1. Persist to Manifest
        auto st = storage::Manifest::DropMembrane(base_.path, name);
        if (!st.ok()) return st;

        // 2. Remove from Memory
        if (it->second.vector_engine) {
            (void)it->second.vector_engine->Close();
        }
        if (it->second.rag_engine) {
            (void)it->second.rag_engine->Close();
        }
        membranes_.erase(it);
        return Status::Ok();
    }

    Status MembraneManager::OpenMembrane(std::string_view name)
    {
        auto *state = GetMembraneOrNull(name);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind == pomai::MembraneKind::kVector) {
            return state->vector_engine->Open();
        }
        return state->rag_engine->Open();
    }

    Status MembraneManager::CloseMembrane(std::string_view name)
    {
        auto *state = GetMembraneOrNull(name);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind == pomai::MembraneKind::kVector) {
            return state->vector_engine->Close();
        }
        return state->rag_engine->Close();
    }

    Status MembraneManager::ListMembranes(std::vector<std::string> *out) const
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        out->clear();
        out->reserve(membranes_.size());
        for (const auto &kv : membranes_)
            out->push_back(kv.first);
        std::sort(out->begin(), out->end());
        return Status::Ok();
    }

    Status MembraneManager::Put(std::string_view membrane, VectorId id, std::span<const float> vec)
    {
        return PutVector(membrane, id, vec);
    }

    Status MembraneManager::Put(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta)
    {
        return PutVector(membrane, id, vec, meta);
    }

    Status MembraneManager::PutVector(std::string_view membrane, VectorId id, std::span<const float> vec)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for PutVector");
        return state->vector_engine->Put(id, vec);
    }

    Status MembraneManager::PutVector(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for PutVector");
        return state->vector_engine->Put(id, vec, meta);
    }

    Status MembraneManager::PutChunk(std::string_view membrane, const pomai::RagChunk& chunk)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kRag)
            return Status::InvalidArgument("RAG membrane required for PutChunk");
        if (chunk.tokens.empty()) {
            return Status::InvalidArgument("RAG membrane requires token_blob; vector-only payload rejected");
        }
        return state->rag_engine->PutChunk(chunk);
    }

    Status MembraneManager::PutBatch(std::string_view membrane,
                                     const std::vector<VectorId>& ids,
                                     const std::vector<std::span<const float>>& vectors)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for PutBatch");
        return state->vector_engine->PutBatch(ids, vectors);
    }

    Status MembraneManager::Get(std::string_view membrane, VectorId id, std::vector<float> *out)
    {
        return Get(membrane, id, out, nullptr);
    }

    Status MembraneManager::Get(std::string_view membrane, VectorId id, std::vector<float> *out, Metadata* out_meta)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for Get");
        return state->vector_engine->Get(id, out, out_meta);
    }

    Status MembraneManager::Exists(std::string_view membrane, VectorId id, bool *exists)
    {
        if (!exists)
            return Status::InvalidArgument("exists is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for Exists");
        return state->vector_engine->Exists(id, exists);
    }

    Status MembraneManager::Delete(std::string_view membrane, VectorId id)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for Delete");
        return state->vector_engine->Delete(id);
    }

    Status MembraneManager::Search(std::string_view membrane, std::span<const float> query,
                                   std::uint32_t topk, pomai::SearchResult *out)
    {
        // Default options (empty filters)
        return Search(membrane, query, topk, SearchOptions{}, out);
    }

    Status MembraneManager::Search(std::string_view membrane, std::span<const float> query,
                                   std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out)
    {
        return SearchVector(membrane, query, topk, opts, out);
    }

    Status MembraneManager::SearchVector(std::string_view membrane, std::span<const float> query,
                                         std::uint32_t topk, pomai::SearchResult *out)
    {
        return SearchVector(membrane, query, topk, SearchOptions{}, out);
    }

    Status MembraneManager::SearchVector(std::string_view membrane, std::span<const float> query,
                                         std::uint32_t topk, const SearchOptions& opts, pomai::SearchResult *out)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for Search");
        return state->vector_engine->Search(query, topk, opts, out);
    }

    Status MembraneManager::SearchRag(std::string_view membrane, const pomai::RagQuery& query,
                                      const pomai::RagSearchOptions& opts, pomai::RagSearchResult *out)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kRag)
            return Status::InvalidArgument("RAG membrane required for SearchRag");
        return state->rag_engine->Search(query, opts, out);
    }

    Status MembraneManager::Freeze(std::string_view membrane)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for Freeze");
        return state->vector_engine->Freeze();
    }

    Status MembraneManager::Compact(std::string_view membrane)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for Compact");
        return state->vector_engine->Compact();
    }

    Status MembraneManager::NewIterator(std::string_view membrane, std::unique_ptr<pomai::SnapshotIterator> *out)
    {
        if (!out) return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for iterator");
        return state->vector_engine->NewIterator(out);
    }

    Status MembraneManager::GetSnapshot(std::string_view membrane, std::shared_ptr<pomai::Snapshot>* out)
    {
        if (!out) return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for snapshot");
        return state->vector_engine->GetSnapshot(out);
    }

    Status MembraneManager::NewIterator(std::string_view membrane, const std::shared_ptr<pomai::Snapshot>& snap, std::unique_ptr<pomai::SnapshotIterator> *out)
    {
        if (!out) return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for iterator");
        return state->vector_engine->NewIterator(snap, out);
    }

} // namespace pomai::core
