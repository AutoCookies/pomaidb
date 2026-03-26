#include "core/membrane/manager.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <string_view>
#include <utility>
#include <vector>

#include "core/vector_engine/vector_engine.h"
#include "core/rag/rag_engine.h"
#include "core/text/text_membrane.h"
#include "core/timeseries/timeseries_engine.h"
#include "core/keyvalue/keyvalue_engine.h"
#include "core/sketch/sketch_engine.h"
#include "core/blob/blob_engine.h"
#include "core/spatial/spatial_engine.h"
#include "core/mesh/mesh_engine.h"
#include "core/sparse/sparse_engine.h"
#include "core/bitset/bitset_engine.h"
#include "pomai/pomai.h"
#include "pomai/iterator.h"  // For SnapshotIterator
#include "storage/manifest/manifest.h"
#include "util/logging.h"
#include "core/graph/graph_membrane_impl.h"
#include "core/connectivity/edge_gateway.h"


namespace pomai::core
{
    namespace {
    KeyValueEngine::RetentionPolicy KvPolicyFromSpec(const pomai::MembraneSpec& spec) {
        KeyValueEngine::RetentionPolicy p;
        p.ttl_sec = spec.ttl_sec;
        p.max_count = spec.retention_max_count;
        p.max_bytes = spec.retention_max_bytes;
        return p;
    }

    bool ParseCsvFloatsReplay(std::string_view s, std::vector<float>* out) {
        out->clear();
        while (!s.empty()) {
            std::size_t comma = s.find(',');
            std::string_view tok = (comma == std::string_view::npos) ? s : s.substr(0, comma);
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.remove_prefix(1);
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.remove_suffix(1);
            if (tok.empty()) return false;
            out->push_back(static_cast<float>(std::strtod(std::string(tok).c_str(), nullptr)));
            if (comma == std::string_view::npos) break;
            s.remove_prefix(comma + 1);
        }
        return !out->empty();
    }

    bool ParseCsvUint32Replay(std::string_view s, std::vector<uint32_t>* out) {
        out->clear();
        while (!s.empty()) {
            std::size_t comma = s.find(',');
            std::string_view tok = (comma == std::string_view::npos) ? s : s.substr(0, comma);
            while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) tok.remove_prefix(1);
            while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.remove_suffix(1);
            if (tok.empty()) return false;
            out->push_back(static_cast<uint32_t>(std::strtoul(std::string(tok).c_str(), nullptr, 10)));
            if (comma == std::string_view::npos) break;
            s.remove_prefix(comma + 1);
        }
        return !out->empty();
    }
    }

    namespace {
    class MeshLodTask final : public DatabaseTask {
    public:
        explicit MeshLodTask(MembraneManager* owner) : owner_(owner) {}

        Status Run() override {
            if (!owner_) return Status::Ok();
            owner_->RunMeshLodSlice();
            return Status::Ok();
        }

        std::string Name() const override { return "MeshLodTask"; }

    private:
        MembraneManager* owner_;
    };
    } // namespace

    MembraneManager::MembraneManager(pomai::DBOptions base) : base_(std::move(base)) {
        orchestrator_ = std::make_unique<QueryOrchestrator>(this, base_.max_query_frontier);
        edge_gateway_ = std::make_unique<EdgeGateway>(this);
    }
    MembraneManager::~MembraneManager() = default;

    Status MembraneManager::Open()
    {
        opened_ = true;
        const auto interval_ms = base_.mesh_lod_build_interval_ms == 0 ? 50u : base_.mesh_lod_build_interval_ms;
        scheduler_.RegisterPeriodic(std::make_unique<MeshLodTask>(this), std::chrono::milliseconds(interval_ms));

        // Ensure default membrane exists and is opened.
        pomai::MembraneSpec spec;
        spec.name = std::string(kDefaultMembrane);
        spec.dim = base_.dim;
        spec.shard_count = base_.shard_count;
        spec.index_params = base_.index_params;
        spec.metric = base_.metric;

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
                state.vector_engine = std::make_unique<VectorEngine>(opt, loaded_spec.kind, loaded_spec.metric, loaded_spec.ttl_sec,
                                                                     loaded_spec.retention_max_count, loaded_spec.retention_max_bytes,
                                                                     loaded_spec.sync_lsn);
            } else if (loaded_spec.kind == pomai::MembraneKind::kRag) {
                state.rag_engine = std::make_unique<RagEngine>(opt, loaded_spec);
            } else if (loaded_spec.kind == pomai::MembraneKind::kGraph) {
                std::unique_ptr<storage::Wal> wal = std::make_unique<storage::Wal>(opt.env, opt.path, 0, 64 * 1024 * 1024, opt.fsync,
                    opt.enable_encryption_at_rest, opt.encryption_key_hex);
                auto wst = wal->Open();
                if (!wst.ok()) return wst;
                state.graph_engine = std::make_unique<GraphMembraneImpl>(std::move(wal));
            } else if (loaded_spec.kind == pomai::MembraneKind::kText) {
                state.text_engine = std::make_unique<TextMembrane>(base_.max_text_docs);
            } else if (loaded_spec.kind == pomai::MembraneKind::kTimeSeries) {
                state.timeseries_engine = std::make_unique<TimeSeriesEngine>(opt.path, base_.max_lifecycle_entries);
            } else if (loaded_spec.kind == pomai::MembraneKind::kKeyValue) {
                state.keyvalue_engine = std::make_unique<KeyValueEngine>(opt.path, base_.max_kv_entries, KvPolicyFromSpec(loaded_spec));
            } else if (loaded_spec.kind == pomai::MembraneKind::kMeta) {
                state.meta_engine = std::make_unique<KeyValueEngine>(opt.path, base_.max_kv_entries, KvPolicyFromSpec(loaded_spec));
            } else if (loaded_spec.kind == pomai::MembraneKind::kSketch) {
                state.sketch_engine = std::make_unique<SketchEngine>(base_.max_sketch_entries);
            } else if (loaded_spec.kind == pomai::MembraneKind::kBlob) {
                state.blob_engine = std::make_unique<BlobEngine>(opt.path, static_cast<std::size_t>(base_.max_blob_bytes_mb) * 1024u * 1024u);
            } else if (loaded_spec.kind == pomai::MembraneKind::kSpatial) {
                state.spatial_engine = std::make_unique<SpatialEngine>(opt.path, base_.max_spatial_points);
            } else if (loaded_spec.kind == pomai::MembraneKind::kMesh) {
                state.mesh_engine = std::make_unique<MeshEngine>(opt.path, base_.max_mesh_objects, base_);
            } else if (loaded_spec.kind == pomai::MembraneKind::kSparse) {
                state.sparse_engine = std::make_unique<SparseEngine>(base_.max_sparse_entries);
            } else if (loaded_spec.kind == pomai::MembraneKind::kBitset) {
                state.bitset_engine = std::make_unique<BitsetEngine>(static_cast<std::size_t>(base_.max_bitset_bytes_mb) * 1024u * 1024u);
            } else if (loaded_spec.kind == pomai::MembraneKind::kAudio) {
                state.audio_engine = std::make_unique<AudioEngine>(opt.path, base_.max_audio_frames);
            } else if (loaded_spec.kind == pomai::MembraneKind::kBloom) {
                state.bloom_engine = std::make_unique<BloomEngine>(base_.max_bloom_entries);
            } else if (loaded_spec.kind == pomai::MembraneKind::kDocument) {
                state.document_engine = std::make_unique<DocumentEngine>(opt.path, base_.max_document_entries);
            } else {
                state.text_engine = std::make_unique<TextMembrane>(base_.max_text_docs);
            }
            state.lifecycle.SetMaxEntries(base_.max_lifecycle_entries);
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
                    state.vector_engine = std::make_unique<VectorEngine>(opt, mspec.kind, mspec.metric, mspec.ttl_sec,
                                                                         mspec.retention_max_count, mspec.retention_max_bytes,
                                                                         mspec.sync_lsn);
                } else if (mspec.kind == pomai::MembraneKind::kRag) {
                    state.rag_engine = std::make_unique<RagEngine>(opt, mspec);
                } else if (mspec.kind == pomai::MembraneKind::kGraph) {
                    std::unique_ptr<storage::Wal> wal = std::make_unique<storage::Wal>(opt.env, opt.path, 0, 64 * 1024 * 1024, opt.fsync,
                        opt.enable_encryption_at_rest, opt.encryption_key_hex);
                    auto wst = wal->Open();
                    if (!wst.ok()) return wst;
                    state.graph_engine = std::make_unique<GraphMembraneImpl>(std::move(wal));
                } else if (mspec.kind == pomai::MembraneKind::kText) {
                    state.text_engine = std::make_unique<TextMembrane>(base_.max_text_docs);
                } else if (mspec.kind == pomai::MembraneKind::kTimeSeries) {
                    state.timeseries_engine = std::make_unique<TimeSeriesEngine>(opt.path, base_.max_lifecycle_entries);
                } else if (mspec.kind == pomai::MembraneKind::kKeyValue) {
                    state.keyvalue_engine = std::make_unique<KeyValueEngine>(opt.path, base_.max_kv_entries, KvPolicyFromSpec(mspec));
                } else if (mspec.kind == pomai::MembraneKind::kMeta) {
                    state.meta_engine = std::make_unique<KeyValueEngine>(opt.path, base_.max_kv_entries, KvPolicyFromSpec(mspec));
                } else if (mspec.kind == pomai::MembraneKind::kSketch) {
                    state.sketch_engine = std::make_unique<SketchEngine>(base_.max_sketch_entries);
                } else if (mspec.kind == pomai::MembraneKind::kBlob) {
                    state.blob_engine = std::make_unique<BlobEngine>(opt.path, static_cast<std::size_t>(base_.max_blob_bytes_mb) * 1024u * 1024u);
                } else if (mspec.kind == pomai::MembraneKind::kSpatial) {
                    state.spatial_engine = std::make_unique<SpatialEngine>(opt.path, base_.max_spatial_points);
                } else if (mspec.kind == pomai::MembraneKind::kMesh) {
                    state.mesh_engine = std::make_unique<MeshEngine>(opt.path, base_.max_mesh_objects, base_);
                } else if (mspec.kind == pomai::MembraneKind::kSparse) {
                    state.sparse_engine = std::make_unique<SparseEngine>(base_.max_sparse_entries);
                } else if (mspec.kind == pomai::MembraneKind::kBitset) {
                    state.bitset_engine = std::make_unique<BitsetEngine>(static_cast<std::size_t>(base_.max_bitset_bytes_mb) * 1024u * 1024u);
                } else if (mspec.kind == pomai::MembraneKind::kAudio) {
                    state.audio_engine = std::make_unique<AudioEngine>(opt.path, base_.max_audio_frames);
                } else if (mspec.kind == pomai::MembraneKind::kBloom) {
                    state.bloom_engine = std::make_unique<BloomEngine>(base_.max_bloom_entries);
                } else if (mspec.kind == pomai::MembraneKind::kDocument) {
                    state.document_engine = std::make_unique<DocumentEngine>(opt.path, base_.max_document_entries);
                } else {
                    state.text_engine = std::make_unique<TextMembrane>(base_.max_text_docs);
                }
                state.lifecycle.SetMaxEntries(base_.max_lifecycle_entries);
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
            if (kv.second.timeseries_engine) {
                (void)kv.second.timeseries_engine->Close();
            }
            if (kv.second.keyvalue_engine) {
                (void)kv.second.keyvalue_engine->Close();
            }
            if (kv.second.meta_engine) {
                (void)kv.second.meta_engine->Close();
            }
            if (kv.second.blob_engine) {
                (void)kv.second.blob_engine->Close();
            }
            if (kv.second.spatial_engine) {
                (void)kv.second.spatial_engine->Close();
            }
            if (kv.second.mesh_engine) {
                (void)kv.second.mesh_engine->Close();
            }
            if (kv.second.audio_engine) {
                (void)kv.second.audio_engine->Close();
            }
            if (kv.second.bloom_engine) {
                (void)kv.second.bloom_engine->Close();
            }
            if (kv.second.document_engine) {
                (void)kv.second.document_engine->Close();
            }
        }
        membranes_.clear();
        if (edge_gateway_) (void)edge_gateway_->Stop();
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
        const bool dim_required = (spec.kind == pomai::MembraneKind::kVector ||
                                   spec.kind == pomai::MembraneKind::kRag ||
                                   spec.kind == pomai::MembraneKind::kGraph ||
                                   spec.kind == pomai::MembraneKind::kText);
        if (dim_required && spec.dim == 0)
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
            state.vector_engine = std::make_unique<VectorEngine>(opt, spec.kind, spec.metric, spec.ttl_sec, spec.retention_max_count,
                                                                 spec.retention_max_bytes, spec.sync_lsn);
        } else if (spec.kind == pomai::MembraneKind::kRag) {
            state.rag_engine = std::make_unique<RagEngine>(opt, spec);
        } else if (spec.kind == pomai::MembraneKind::kGraph) {
            std::unique_ptr<storage::Wal> wal = std::make_unique<storage::Wal>(opt.env, opt.path, 0, 64 * 1024 * 1024, opt.fsync,
                opt.enable_encryption_at_rest, opt.encryption_key_hex);
            auto wst = wal->Open();
            if (!wst.ok()) return wst;
            state.graph_engine = std::make_unique<GraphMembraneImpl>(std::move(wal));
        } else if (spec.kind == pomai::MembraneKind::kText) {
            state.text_engine = std::make_unique<TextMembrane>(base_.max_text_docs);
        } else if (spec.kind == pomai::MembraneKind::kTimeSeries) {
            state.timeseries_engine = std::make_unique<TimeSeriesEngine>(opt.path, base_.max_lifecycle_entries);
        } else if (spec.kind == pomai::MembraneKind::kKeyValue) {
            state.keyvalue_engine = std::make_unique<KeyValueEngine>(opt.path, base_.max_kv_entries, KvPolicyFromSpec(spec));
        } else if (spec.kind == pomai::MembraneKind::kMeta) {
            state.meta_engine = std::make_unique<KeyValueEngine>(opt.path, base_.max_kv_entries, KvPolicyFromSpec(spec));
        } else if (spec.kind == pomai::MembraneKind::kSketch) {
            state.sketch_engine = std::make_unique<SketchEngine>(base_.max_sketch_entries);
        } else if (spec.kind == pomai::MembraneKind::kBlob) {
            state.blob_engine = std::make_unique<BlobEngine>(opt.path, static_cast<std::size_t>(base_.max_blob_bytes_mb) * 1024u * 1024u);
        } else if (spec.kind == pomai::MembraneKind::kSpatial) {
            state.spatial_engine = std::make_unique<SpatialEngine>(opt.path, base_.max_spatial_points);
        } else if (spec.kind == pomai::MembraneKind::kMesh) {
            state.mesh_engine = std::make_unique<MeshEngine>(opt.path, base_.max_mesh_objects, base_);
        } else if (spec.kind == pomai::MembraneKind::kSparse) {
            state.sparse_engine = std::make_unique<SparseEngine>(base_.max_sparse_entries);
        } else if (spec.kind == pomai::MembraneKind::kBitset) {
            state.bitset_engine = std::make_unique<BitsetEngine>(static_cast<std::size_t>(base_.max_bitset_bytes_mb) * 1024u * 1024u);
        } else if (spec.kind == pomai::MembraneKind::kAudio) {
            state.audio_engine = std::make_unique<AudioEngine>(opt.path, base_.max_audio_frames);
        } else if (spec.kind == pomai::MembraneKind::kBloom) {
            state.bloom_engine = std::make_unique<BloomEngine>(base_.max_bloom_entries);
        } else if (spec.kind == pomai::MembraneKind::kDocument) {
            state.document_engine = std::make_unique<DocumentEngine>(opt.path, base_.max_document_entries);
        } else {
            state.text_engine = std::make_unique<TextMembrane>(base_.max_text_docs);
        }
        state.lifecycle.SetMaxEntries(base_.max_lifecycle_entries);
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
        } else if (state->spec.kind == pomai::MembraneKind::kRag) {
            return state->rag_engine->Open();
        } else if (state->spec.kind == pomai::MembraneKind::kTimeSeries) {
            return state->timeseries_engine->Open();
        } else if (state->spec.kind == pomai::MembraneKind::kKeyValue) {
            return state->keyvalue_engine->Open();
        } else if (state->spec.kind == pomai::MembraneKind::kMeta) {
            return state->meta_engine->Open();
        } else if (state->spec.kind == pomai::MembraneKind::kBlob) {
            return state->blob_engine->Open();
        } else if (state->spec.kind == pomai::MembraneKind::kSpatial) {
            return state->spatial_engine->Open();
        } else if (state->spec.kind == pomai::MembraneKind::kMesh) {
            return state->mesh_engine->Open();
        }
        // Graph/Text membrane currently initialize on create/restore.
        return Status::Ok();
    }

    Status MembraneManager::CloseMembrane(std::string_view name)
    {
        auto *state = GetMembraneOrNull(name);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind == pomai::MembraneKind::kVector) {
            return state->vector_engine->Close();
        } else if (state->spec.kind == pomai::MembraneKind::kRag) {
            return state->rag_engine->Close();
        } else if (state->spec.kind == pomai::MembraneKind::kTimeSeries) {
            return state->timeseries_engine->Close();
        } else if (state->spec.kind == pomai::MembraneKind::kKeyValue) {
            return state->keyvalue_engine->Close();
        } else if (state->spec.kind == pomai::MembraneKind::kMeta) {
            return state->meta_engine->Close();
        } else if (state->spec.kind == pomai::MembraneKind::kBlob) {
            return state->blob_engine->Close();
        } else if (state->spec.kind == pomai::MembraneKind::kSpatial) {
            return state->spatial_engine->Close();
        } else if (state->spec.kind == pomai::MembraneKind::kMesh) {
            return state->mesh_engine->Close();
        }
        return Status::Ok();
    }

    Status MembraneManager::UpdateMembraneRetention(std::string_view name, uint32_t ttl_sec,
                                                    uint32_t retention_max_count, uint64_t retention_max_bytes) {
        auto* state = GetMembraneOrNull(name);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->meta_engine && !state->keyvalue_engine) {
            return Status::InvalidArgument("retention update currently supported for META/KEYVALUE membranes");
        }
        auto st = storage::Manifest::UpdateRetentionPolicy(base_.path, name, ttl_sec, retention_max_count, retention_max_bytes);
        if (!st.ok()) return st;

        state->spec.ttl_sec = ttl_sec;
        state->spec.retention_max_count = retention_max_count;
        state->spec.retention_max_bytes = retention_max_bytes;
        KeyValueEngine::RetentionPolicy p{};
        p.ttl_sec = ttl_sec;
        p.max_count = retention_max_count;
        p.max_bytes = retention_max_bytes;
        if (state->meta_engine) return state->meta_engine->SetRetentionPolicy(p);
        return state->keyvalue_engine->SetRetentionPolicy(p);
    }

    Status MembraneManager::GetMembraneRetention(std::string_view name, uint32_t* ttl_sec,
                                                 uint32_t* retention_max_count, uint64_t* retention_max_bytes) const {
        if (!ttl_sec || !retention_max_count || !retention_max_bytes) {
            return Status::InvalidArgument("retention out args must be non-null");
        }
        const auto* state = GetMembraneOrNull(name);
        if (!state) return Status::NotFound("membrane not found");
        *ttl_sec = state->spec.ttl_sec;
        *retention_max_count = state->spec.retention_max_count;
        *retention_max_bytes = state->spec.retention_max_bytes;
        return Status::Ok();
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
        PollMaintenance();
        return PutVector(membrane, id, vec);
    }

    Status MembraneManager::Put(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta)
    {
        PollMaintenance();
        return PutVector(membrane, id, vec, meta);
    }

    Status MembraneManager::PutVector(std::string_view membrane, VectorId id, std::span<const float> vec)
    {
        PollMaintenance();
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind == pomai::MembraneKind::kText) {
            // kText ignores dense vector and indexes metadata text only.
            return state->text_engine->Put(id, std::to_string(id));
        }
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for PutVector");
        auto st = MaybeApplyBackpressure(state);
        if (!st.ok()) return st;
        state->lifecycle.OnWrite(id);
        st = state->vector_engine->Put(id, vec);
        if (st.ok()) {
            for (auto& hook : state->hooks) {
                hook->OnPostPut(id, vec, Metadata());
            }
        }
        return st;
    }

    Status MembraneManager::PutVector(std::string_view membrane, VectorId id, std::span<const float> vec, const Metadata& meta)
    {
        PollMaintenance();
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind == pomai::MembraneKind::kText) {
            const std::string text = !meta.text.empty() ? meta.text : meta.payload;
            if (text.empty()) return Status::InvalidArgument("kText PutVector requires metadata text/payload");
            state->lifecycle.OnWrite(id);
            return state->text_engine->Put(id, text);
        }
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for PutVector");
        auto st = MaybeApplyBackpressure(state);
        if (!st.ok()) return st;
        state->lifecycle.OnWrite(id);
        st = state->vector_engine->Put(id, vec, meta);
        if (st.ok()) {
            for (auto& hook : state->hooks) {
                hook->OnPostPut(id, vec, meta);
            }
        }
        return st;
    }

    Status MembraneManager::PutChunk(std::string_view membrane, const pomai::RagChunk& chunk)
    {
        PollMaintenance();
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
        PollMaintenance();
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for PutBatch");
        auto st = MaybeApplyBackpressure(state);
        if (!st.ok()) return st;
        st = state->vector_engine->PutBatch(ids, vectors);
        if (st.ok()) {
            for (size_t i = 0; i < ids.size(); ++i) {
                for (auto& hook : state->hooks) {
                    hook->OnPostPut(ids[i], vectors[i], Metadata());
                }
            }
        }
        return st;
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
        state->lifecycle.OnDelete(id);
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
        if (state->spec.kind == pomai::MembraneKind::kText) {
            return Status::InvalidArgument("use SearchLexical for kText membrane");
        }
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for Search");
        auto st = state->vector_engine->Search(query, topk, opts, out);
        if (st.ok()) for (const auto& h : out->hits) state->lifecycle.OnRead(h.id);
        return st;
    }

    Status MembraneManager::SearchLexical(std::string_view membrane, const std::string& query, uint32_t topk, std::vector<LexicalHit>* out) {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (state->spec.kind == pomai::MembraneKind::kText) {
            auto st = state->text_engine->Search(query, topk, out);
            if (st.ok()) for (const auto& h : *out) state->lifecycle.OnRead(h.id);
            return st;
        }
        if (!state->vector_engine) return Status::InvalidArgument("VECTOR membrane required for SearchLexical");
        auto st = state->vector_engine->SearchLexical(query, topk, out);
        if (st.ok()) for (const auto& h : *out) state->lifecycle.OnRead(h.id);
        return st;
    }

    Status MembraneManager::SearchBatch(std::string_view membrane, std::span<const float> queries,
                                        uint32_t num_queries, std::uint32_t topk, std::vector<pomai::SearchResult> *out)
    {
        return SearchBatch(membrane, queries, num_queries, topk, SearchOptions{}, out);
    }

    Status MembraneManager::SearchBatch(std::string_view membrane, std::span<const float> queries,
                                        uint32_t num_queries, std::uint32_t topk, const SearchOptions& opts, std::vector<pomai::SearchResult> *out)
    {
        if (!out)
            return Status::InvalidArgument("out is null");
        auto *state = GetMembraneOrNull(membrane);
        if (!state)
            return Status::NotFound("membrane not found");
        if (state->spec.kind != pomai::MembraneKind::kVector)
            return Status::InvalidArgument("VECTOR membrane required for SearchBatch");
        return state->vector_engine->SearchBatch(queries, num_queries, topk, opts, out);
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

    Status MembraneManager::SearchMultiModal(std::string_view membrane, const MultiModalQuery& query, SearchResult* out) {
        return orchestrator_ ? orchestrator_->Execute(membrane, query, out) : Status::InvalidArgument("not opened");
    }

    Status MembraneManager::TsPut(std::string_view membrane, uint64_t series_id, uint64_t timestamp, double value) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->timeseries_engine) return Status::InvalidArgument("TIMESERIES membrane required");
        return state->timeseries_engine->Put(series_id, timestamp, value);
    }

    Status MembraneManager::TsRange(std::string_view membrane, uint64_t series_id, uint64_t start_ts, uint64_t end_ts, std::vector<pomai::TimeSeriesPoint>* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->timeseries_engine) return Status::InvalidArgument("TIMESERIES membrane required");
        return state->timeseries_engine->Range(series_id, start_ts, end_ts, out);
    }

    Status MembraneManager::KvPut(std::string_view membrane, std::string_view key, std::string_view value) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->keyvalue_engine) return Status::InvalidArgument("KEYVALUE membrane required");
        return state->keyvalue_engine->Put(key, value);
    }

    Status MembraneManager::KvGet(std::string_view membrane, std::string_view key, std::string* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->keyvalue_engine) return Status::InvalidArgument("KEYVALUE membrane required");
        return state->keyvalue_engine->Get(key, out);
    }

    Status MembraneManager::KvDelete(std::string_view membrane, std::string_view key) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->keyvalue_engine) return Status::InvalidArgument("KEYVALUE membrane required");
        return state->keyvalue_engine->Delete(key);
    }

    Status MembraneManager::MetaPut(std::string_view membrane, std::string_view gid, std::string_view value) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->meta_engine) return Status::InvalidArgument("META membrane required");
        return state->meta_engine->Put(gid, value);
    }

    Status MembraneManager::MetaGet(std::string_view membrane, std::string_view gid, std::string* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->meta_engine) return Status::InvalidArgument("META membrane required");
        return state->meta_engine->Get(gid, out);
    }

    Status MembraneManager::MetaDelete(std::string_view membrane, std::string_view gid) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->meta_engine) return Status::InvalidArgument("META membrane required");
        return state->meta_engine->Delete(gid);
    }

    Status MembraneManager::LinkObjects(std::string_view gid, uint64_t vector_id, uint64_t graph_vertex_id, uint64_t mesh_id) {
        return object_linker_.LinkByGid(std::string(gid), vector_id, graph_vertex_id, mesh_id);
    }

    Status MembraneManager::UnlinkObjects(std::string_view gid) {
        return object_linker_.UnlinkByGid(std::string(gid));
    }

    std::optional<LinkedObject> MembraneManager::ResolveLinkedByVectorId(uint64_t vector_id) const {
        return object_linker_.ResolveByVectorId(vector_id);
    }

    Status MembraneManager::StartEdgeGateway(uint16_t http_port, uint16_t ingest_port) {
        if (!edge_gateway_) return Status::InvalidArgument("edge gateway unavailable");
        return edge_gateway_->Start(http_port, ingest_port);
    }

    Status MembraneManager::StartEdgeGatewaySecure(uint16_t http_port, uint16_t ingest_port, std::string_view auth_token) {
        if (!edge_gateway_) return Status::InvalidArgument("edge gateway unavailable");
        return edge_gateway_->Start(http_port, ingest_port, std::string(auth_token));
    }

    Status MembraneManager::StopEdgeGateway() {
        if (!edge_gateway_) return Status::Ok();
        return edge_gateway_->Stop();
    }

    Status MembraneManager::SketchAdd(std::string_view membrane, std::string_view key, uint64_t increment) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->sketch_engine) return Status::InvalidArgument("SKETCH membrane required");
        return state->sketch_engine->Add(key, increment);
    }

    Status MembraneManager::SketchEstimate(std::string_view membrane, std::string_view key, uint64_t* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->sketch_engine) return Status::InvalidArgument("SKETCH membrane required");
        return state->sketch_engine->Estimate(key, out);
    }

    Status MembraneManager::SketchSeen(std::string_view membrane, std::string_view key, bool* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->sketch_engine) return Status::InvalidArgument("SKETCH membrane required");
        return state->sketch_engine->Seen(key, out);
    }

    Status MembraneManager::SketchUniqueEstimate(std::string_view membrane, uint64_t* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->sketch_engine) return Status::InvalidArgument("SKETCH membrane required");
        return state->sketch_engine->UniqueEstimate(out);
    }

    Status MembraneManager::BlobPut(std::string_view membrane, uint64_t blob_id, std::span<const uint8_t> data) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->blob_engine) return Status::InvalidArgument("BLOB membrane required");
        return state->blob_engine->Put(blob_id, data);
    }

    Status MembraneManager::BlobGet(std::string_view membrane, uint64_t blob_id, std::vector<uint8_t>* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->blob_engine) return Status::InvalidArgument("BLOB membrane required");
        return state->blob_engine->Get(blob_id, out);
    }

    Status MembraneManager::BlobDelete(std::string_view membrane, uint64_t blob_id) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->blob_engine) return Status::InvalidArgument("BLOB membrane required");
        return state->blob_engine->Delete(blob_id);
    }

    Status MembraneManager::SpatialPut(std::string_view membrane, uint64_t entity_id, double latitude, double longitude) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->spatial_engine) return Status::InvalidArgument("SPATIAL membrane required");
        return state->spatial_engine->Put(entity_id, latitude, longitude);
    }

    Status MembraneManager::SpatialRadiusSearch(std::string_view membrane, double latitude, double longitude, double radius_meters, std::vector<pomai::SpatialPoint>* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->spatial_engine) return Status::InvalidArgument("SPATIAL membrane required");
        return state->spatial_engine->RadiusSearch(latitude, longitude, radius_meters, out);
    }

    Status MembraneManager::SpatialWithinPolygon(std::string_view membrane, const pomai::GeoPolygon& polygon, std::vector<pomai::SpatialPoint>* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->spatial_engine) return Status::InvalidArgument("SPATIAL membrane required");
        return state->spatial_engine->WithinPolygon(polygon, out);
    }

    Status MembraneManager::SpatialNearest(std::string_view membrane, double latitude, double longitude, uint32_t topk, std::vector<pomai::SpatialPoint>* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->spatial_engine) return Status::InvalidArgument("SPATIAL membrane required");
        return state->spatial_engine->Nearest(latitude, longitude, topk, out);
    }

    Status MembraneManager::MeshPut(std::string_view membrane, uint64_t mesh_id, std::span<const float> vertices_xyz) {
        PollMaintenance();
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->mesh_engine) return Status::InvalidArgument("MESH membrane required");
        auto st = state->mesh_engine->Put(mesh_id, vertices_xyz);
        if (!st.ok()) return st;
        return state->mesh_engine->ProcessLodJobs(1);
    }

    Status MembraneManager::MeshRmsd(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, double* out) {
        return MeshRmsd(membrane, mesh_a, mesh_b, MeshQueryOptions{}, out);
    }

    Status MembraneManager::MeshRmsd(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b,
                                     const MeshQueryOptions& opts, double* out) {
        PollMaintenance();
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->mesh_engine) return Status::InvalidArgument("MESH membrane required");
        return state->mesh_engine->Rmsd(mesh_a, mesh_b, opts, out);
    }

    Status MembraneManager::MeshIntersect(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b, bool* out) {
        return MeshIntersect(membrane, mesh_a, mesh_b, MeshQueryOptions{}, out);
    }

    Status MembraneManager::MeshIntersect(std::string_view membrane, uint64_t mesh_a, uint64_t mesh_b,
                                          const MeshQueryOptions& opts, bool* out) {
        PollMaintenance();
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->mesh_engine) return Status::InvalidArgument("MESH membrane required");
        return state->mesh_engine->Intersect(mesh_a, mesh_b, opts, out);
    }

    Status MembraneManager::MeshVolume(std::string_view membrane, uint64_t mesh_id, double* out) {
        return MeshVolume(membrane, mesh_id, MeshQueryOptions{}, out);
    }

    Status MembraneManager::MeshVolume(std::string_view membrane, uint64_t mesh_id,
                                       const MeshQueryOptions& opts, double* out) {
        PollMaintenance();
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->mesh_engine) return Status::InvalidArgument("MESH membrane required");
        return state->mesh_engine->Volume(mesh_id, opts, out);
    }

    Status MembraneManager::SparsePut(std::string_view membrane, uint64_t id, const pomai::SparseEntry& entry) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->sparse_engine) return Status::InvalidArgument("SPARSE membrane required");
        return state->sparse_engine->Put(id, entry);
    }

    Status MembraneManager::SparseDot(std::string_view membrane, uint64_t a, uint64_t b, double* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->sparse_engine) return Status::InvalidArgument("SPARSE membrane required");
        return state->sparse_engine->Dot(a, b, out);
    }

    Status MembraneManager::SparseIntersect(std::string_view membrane, uint64_t a, uint64_t b, uint32_t* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->sparse_engine) return Status::InvalidArgument("SPARSE membrane required");
        return state->sparse_engine->Intersect(a, b, out);
    }

    Status MembraneManager::SparseJaccard(std::string_view membrane, uint64_t a, uint64_t b, double* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->sparse_engine) return Status::InvalidArgument("SPARSE membrane required");
        return state->sparse_engine->Jaccard(a, b, out);
    }

    Status MembraneManager::BitsetPut(std::string_view membrane, uint64_t id, std::span<const uint8_t> bits) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bitset_engine) return Status::InvalidArgument("BITSET membrane required");
        return state->bitset_engine->Put(id, bits);
    }

    Status MembraneManager::BitsetAnd(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bitset_engine) return Status::InvalidArgument("BITSET membrane required");
        return state->bitset_engine->And(a, b, out);
    }

    Status MembraneManager::BitsetOr(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bitset_engine) return Status::InvalidArgument("BITSET membrane required");
        return state->bitset_engine->Or(a, b, out);
    }

    Status MembraneManager::BitsetXor(std::string_view membrane, uint64_t a, uint64_t b, std::vector<uint8_t>* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bitset_engine) return Status::InvalidArgument("BITSET membrane required");
        return state->bitset_engine->Xor(a, b, out);
    }

    Status MembraneManager::BitsetHamming(std::string_view membrane, uint64_t a, uint64_t b, double* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bitset_engine) return Status::InvalidArgument("BITSET membrane required");
        return state->bitset_engine->Hamming(a, b, out);
    }

    Status MembraneManager::BitsetJaccard(std::string_view membrane, uint64_t a, uint64_t b, double* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bitset_engine) return Status::InvalidArgument("BITSET membrane required");
        return state->bitset_engine->Jaccard(a, b, out);
    }

    Status MembraneManager::AddVertex(std::string_view membrane, VertexId id, TagId tag, const Metadata& meta)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->graph_engine) return Status::InvalidArgument("Graph membrane required");
        return static_cast<pomai::GraphMembrane*>(state->graph_engine.get())->AddVertex(id, tag, meta);
    }

    Status MembraneManager::AddEdge(std::string_view membrane, VertexId src, VertexId dst, EdgeType type, uint32_t rank, const Metadata& meta)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->graph_engine) return Status::InvalidArgument("Graph membrane required");
        return static_cast<pomai::GraphMembrane*>(state->graph_engine.get())->AddEdge(src, dst, type, rank, meta);
    }

    Status MembraneManager::GetNeighbors(std::string_view membrane, VertexId src, std::vector<pomai::Neighbor>* out) {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->graph_engine) return Status::InvalidArgument("GRAPH membrane required for GetNeighbors");
        return static_cast<pomai::GraphMembrane*>(state->graph_engine.get())->GetNeighbors(src, out);
    }

    Status MembraneManager::GetNeighbors(std::string_view membrane, VertexId src, EdgeType type, std::vector<pomai::Neighbor>* out) {
        auto *state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->graph_engine) return Status::InvalidArgument("GRAPH membrane required for GetNeighbors");
        return static_cast<pomai::GraphMembrane*>(state->graph_engine.get())->GetNeighbors(src, type, out);
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
        if (state->spec.kind == pomai::MembraneKind::kMeta && state->meta_engine) {
            return state->meta_engine->Compact();
        }
        if (state->spec.kind == pomai::MembraneKind::kKeyValue && state->keyvalue_engine) {
            return state->keyvalue_engine->Compact();
        }
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

    Status MembraneManager::PushSync(std::string_view name, SyncReceiver* receiver)
    {
        auto *state = GetMembraneOrNull(name);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->vector_engine) return Status::InvalidArgument("sync only supported for VECTOR membranes");

        auto st = state->vector_engine->PushSync(receiver);
        if (st.ok()) {
            uint64_t current_lsn = state->vector_engine->GetLastSyncedLSN();
            if (current_lsn > state->spec.sync_lsn) {
                state->spec.sync_lsn = current_lsn;
                // Update persistent manifest
                (void)storage::Manifest::UpdateSyncLSN(base_.path, name, current_lsn);
            }
        }
        return st;
    }

    void MembraneManager::AddPostPutHook(std::string_view membrane, std::shared_ptr<PostPutHook> hook)
    {
        auto *state = GetMembraneOrNull(membrane);
        if (state) {
            state->hooks.push_back(std::move(hook));
        }
    }

    Status MembraneManager::MaybeApplyBackpressure(MembraneState* state)
    {
        if (!state || !state->vector_engine)
            return Status::Ok();
        if (!base_.auto_freeze_on_pressure)
            return Status::Ok();
        if (base_.memtable_flush_threshold_mb == 0)
            return Status::Ok();

        const std::size_t used_bytes = state->vector_engine->MemTableBytesUsed();
        const std::size_t threshold_bytes =
            static_cast<std::size_t>(base_.memtable_flush_threshold_mb) * 1024u * 1024u;
        if (used_bytes < threshold_bytes)
            return Status::Ok();

        const unsigned used_mb = static_cast<unsigned>(used_bytes / (1024u * 1024u));
        POMAI_LOG_WARN("Membrane '{}' memtable pressure ({} MB). Triggering Auto-Freeze.",
                       state->spec.name, used_mb);
        return state->vector_engine->Freeze();
    }

    void MembraneManager::PollMaintenance() { scheduler_.Poll(); }
    void MembraneManager::RunMeshLodSlice() {
        const std::size_t jobs = base_.mesh_lod_jobs_per_tick == 0 ? 1u : static_cast<std::size_t>(base_.mesh_lod_jobs_per_tick);
        for (auto& kv : membranes_) {
            if (kv.second.mesh_engine) (void)kv.second.mesh_engine->ProcessLodJobs(jobs);
        }
    }

    // ── Audio membrane ────────────────────────────────────────────────────────

    Status MembraneManager::AudioPut(std::string_view membrane, uint64_t clip_id,
                                     uint64_t timestamp_ms, std::span<const float> embedding) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->audio_engine) return Status::InvalidArgument("AUDIO membrane required");
        return state->audio_engine->Put(clip_id, timestamp_ms, embedding);
    }

    Status MembraneManager::AudioDelete(std::string_view membrane, uint64_t clip_id) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->audio_engine) return Status::InvalidArgument("AUDIO membrane required");
        return state->audio_engine->Delete(clip_id);
    }

    Status MembraneManager::AudioSearch(std::string_view membrane, std::span<const float> query,
                                        uint64_t time_start_ms, uint64_t time_end_ms,
                                        uint32_t topk, std::vector<AudioHit>* out) {
        if (!out) return Status::InvalidArgument("out is null");
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->audio_engine) return Status::InvalidArgument("AUDIO membrane required");
        return state->audio_engine->Search(query, time_start_ms, time_end_ms, topk, out);
    }

    // ── Bloom filter membrane ─────────────────────────────────────────────────

    Status MembraneManager::BloomAdd(std::string_view membrane, uint64_t filter_id,
                                     std::string_view key) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bloom_engine) return Status::InvalidArgument("BLOOM membrane required");
        return state->bloom_engine->Add(filter_id, key);
    }

    Status MembraneManager::BloomMightContain(std::string_view membrane, uint64_t filter_id,
                                              std::string_view key, bool* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bloom_engine) return Status::InvalidArgument("BLOOM membrane required");
        return state->bloom_engine->MightContain(filter_id, key, out);
    }

    Status MembraneManager::BloomDrop(std::string_view membrane, uint64_t filter_id) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bloom_engine) return Status::InvalidArgument("BLOOM membrane required");
        return state->bloom_engine->Drop(filter_id);
    }

    Status MembraneManager::BloomEstimateFPR(std::string_view membrane, uint64_t filter_id,
                                             double* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->bloom_engine) return Status::InvalidArgument("BLOOM membrane required");
        return state->bloom_engine->EstimateFPR(filter_id, out);
    }

    // ── Document membrane ─────────────────────────────────────────────────────

    Status MembraneManager::DocumentPut(std::string_view membrane, uint64_t doc_id,
                                        std::string_view json_content) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->document_engine) return Status::InvalidArgument("DOCUMENT membrane required");
        return state->document_engine->Put(doc_id, json_content);
    }

    Status MembraneManager::DocumentGet(std::string_view membrane, uint64_t doc_id,
                                        std::string* out) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->document_engine) return Status::InvalidArgument("DOCUMENT membrane required");
        return state->document_engine->Get(doc_id, out);
    }

    Status MembraneManager::DocumentDelete(std::string_view membrane, uint64_t doc_id) {
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->document_engine) return Status::InvalidArgument("DOCUMENT membrane required");
        return state->document_engine->Delete(doc_id);
    }

    Status MembraneManager::DocumentSearch(std::string_view membrane, const std::string& query,
                                           uint32_t topk, std::vector<DocumentHit>* out) {
        if (!out) return Status::InvalidArgument("out is null");
        auto* state = GetMembraneOrNull(membrane);
        if (!state) return Status::NotFound("membrane not found");
        if (!state->document_engine) return Status::InvalidArgument("DOCUMENT membrane required");
        return state->document_engine->Search(query, topk, out);
    }

    Status MembraneManager::ReplayGatewaySyncEvent(uint64_t seq, std::string_view type, std::string_view membrane,
                                                   uint64_t id, uint64_t id2, uint32_t u32_a, uint32_t u32_b,
                                                   std::string_view aux_k, std::string_view aux_v) {
        (void)seq;
        const std::string t(type);
        if (t == "meta_put") return MetaPut(membrane, aux_k, aux_v);
        if (t == "kv_put") return KvPut(membrane, aux_k, aux_v);
        if (t == "document_put") return DocumentPut(membrane, id, aux_v);
        if (t == "graph_vertex_put")
            return AddVertex(membrane, static_cast<pomai::VertexId>(id), static_cast<pomai::TagId>(u32_a), pomai::Metadata{});
        if (t == "graph_edge_put")
            return AddEdge(membrane, static_cast<pomai::VertexId>(id), static_cast<pomai::VertexId>(id2),
                           static_cast<pomai::EdgeType>(u32_a), u32_b, pomai::Metadata{});
        if (t == "timeseries_put") {
            char* endp = nullptr;
            const std::string vs(aux_v);
            const double v = std::strtod(vs.c_str(), &endp);
            if (endp == vs.c_str()) return Status::InvalidArgument("timeseries replay: bad value");
            return TsPut(membrane, id, id2, v);
        }
        if (t == "vector_put") {
            std::vector<float> v;
            if (!ParseCsvFloatsReplay(aux_v, &v)) return Status::InvalidArgument("vector replay: empty vector");
            return PutVector(membrane, id, v);
        }
        if (t == "mesh_put") {
            std::vector<float> xyz;
            if (!ParseCsvFloatsReplay(aux_v, &xyz)) return Status::InvalidArgument("mesh replay: bad vertices");
            return MeshPut(membrane, id, xyz);
        }
        if (t == "audio_put") {
            std::vector<float> emb;
            if (!ParseCsvFloatsReplay(aux_v, &emb)) return Status::InvalidArgument("audio replay: bad embedding");
            return AudioPut(membrane, id, id2, emb);
        }
        if (t == "rag_chunk_put") {
            std::vector<uint32_t> tokens;
            if (!ParseCsvUint32Replay(aux_v, &tokens)) return Status::InvalidArgument("rag replay: bad tokens");
            pomai::RagChunk ch;
            ch.chunk_id = static_cast<pomai::ChunkId>(id);
            ch.doc_id = static_cast<pomai::DocId>(id2);
            ch.tokens = std::move(tokens);
            ch.chunk_text = {};
            ch.meta = pomai::Metadata{};
            return PutChunk(membrane, ch);
        }
        return Status::InvalidArgument(std::string("replay: unknown type ") + t);
    }

} // namespace pomai::core
