#include "core/kernel/pods/vector_pod.h"
#include "core/query/lexical_index.h"
#include <cstring>

namespace pomai::core {
    namespace {
    template <typename T>
    bool PayloadAs(std::span<const uint8_t> payload, const T** out) {
        if (payload.size() != sizeof(T)) return false;
        *out = reinterpret_cast<const T*>(payload.data());
        return true;
    }
    void SetStatus(Status* out, const Status& st) {
        if (out) *out = st;
    }
    } // namespace

    void VectorPod::Handle(Message&& msg) {
        switch (msg.opcode) {
            case Op::kPut: {
                // Payload is VectorId (8) + vec (dim*4)
                if (msg.payload.size() < sizeof(VectorId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector put payload too small"));
                    return;
                }
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                std::span<const float> vec(
                    reinterpret_cast<const float*>(msg.payload.data() + sizeof(VectorId)),
                    (msg.payload.size() - sizeof(VectorId)) / sizeof(float)
                );
                if (vec.empty()) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector put empty payload"));
                    return;
                }
                auto st = runtime_->Put(id, vec);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kPutWithMeta: {
                struct P {
                    VectorId id;
                    const float* vec_data;
                    size_t vec_size;
                    const Metadata* meta;
                };
                const P* p = nullptr;
                if (!PayloadAs(msg.payload, &p) || !p->vec_data || !p->meta || p->vec_size == 0) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector put-meta invalid payload"));
                    return;
                }
                auto st = runtime_->Put(p->id, std::span<const float>(p->vec_data, p->vec_size), *p->meta);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kSearch: {
                struct P {
                    uint32_t topk;
                    const float* query_data;
                    size_t query_size;
                    const SearchOptions* opts;
                };
                struct SearchResultEnvelope {
                    SearchResult* out;
                    Status* st;
                };
                const P* p = nullptr;
                auto* env = static_cast<SearchResultEnvelope*>(msg.result_ptr);
                if (!PayloadAs(msg.payload, &p) || !p->query_data || p->query_size == 0 || !p->opts || !env ||
                    !env->out || !env->st) {
                    if (env && env->st) {
                        *env->st = Status::InvalidArgument("vector search invalid payload");
                    }
                    return;
                }
                std::vector<SearchHit> hits;
                std::span<const float> query(p->query_data, p->query_size);
                auto st = runtime_->Search(query, p->topk, *p->opts, &hits);
                if (st.ok()) {
                    env->out->hits = std::move(hits);
                    env->out->routed_shards_count = 1;
                }
                *env->st = st;
                break;
            }
            case Op::kGet: {
                if (msg.payload.size() < sizeof(VectorId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector get payload too small"));
                    return;
                }
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::vector<float>*>(msg.result_ptr);
                    (void)runtime_->Get(id, out, nullptr);
                }
                break;
            }
            case Op::kGetWithMeta: {
                struct P {
                    VectorId id;
                    std::vector<float>* out_vec;
                    pomai::Metadata* out_meta;
                };
                const P* p = nullptr;
                if (!PayloadAs(msg.payload, &p) || !p->out_vec) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector get-meta invalid payload"));
                    return;
                }
                auto st = runtime_->Get(p->id, p->out_vec, p->out_meta);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kDelete: {
                if (msg.payload.size() < sizeof(VectorId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector delete payload too small"));
                    return;
                }
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                auto st = runtime_->Delete(id);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kFlush: {
                auto st = runtime_->Flush();
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kFreeze: {
                auto st = runtime_->Freeze();
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kExists: {
                if (msg.payload.size() < sizeof(VectorId)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector exists payload too small"));
                    return;
                }
                VectorId id = *reinterpret_cast<const VectorId*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<bool*>(msg.result_ptr);
                    (void)runtime_->Exists(id, out);
                }
                break;
            }
            case Op::kSync: {
                if (msg.result_ptr) {
                    auto* rx = static_cast<SyncReceiver*>(msg.result_ptr);
                    (void)runtime_->PushSync(rx);
                }
                break;
            }
            case Op::kSearchLexical: {
                struct P {
                    uint32_t topk;
                    const std::string* query;
                    std::vector<pomai::core::LexicalHit>* out;
                    Status* st;
                };
                const P* p = nullptr;
                if (!PayloadAs(msg.payload, &p) || !p->query || !p->out || !p->st) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector lexical invalid payload"));
                    return;
                }
                *p->st = runtime_->SearchLexical(*p->query, p->topk, p->out);
                break;
            }
            case Op::kPutBatch: {
                struct P {
                    const std::vector<VectorId>* ids;
                    const std::vector<std::span<const float>>* vectors;
                };
                const P* p = nullptr;
                if (!PayloadAs(msg.payload, &p) || !p->ids || !p->vectors) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector put-batch invalid payload"));
                    return;
                }
                auto st = runtime_->PutBatch(*p->ids, *p->vectors);
                SetStatus(msg.status_ptr, st);
                break;
            }
            case Op::kGetSnapshot: {
                if (msg.result_ptr) {
                    auto* out = static_cast<std::shared_ptr<Snapshot>*>(msg.result_ptr);
                    *out = runtime_->GetSnapshot();
                }
                break;
            }
            case Op::kNewIterator: {
                if (msg.payload.size() < sizeof(void*)) {
                    SetStatus(msg.status_ptr, Status::InvalidArgument("vector new-iterator payload too small"));
                    return;
                }
                const std::shared_ptr<Snapshot>* snap = *reinterpret_cast<const std::shared_ptr<Snapshot>* const*>(msg.payload.data());
                if (msg.result_ptr) {
                    auto* out = static_cast<std::unique_ptr<SnapshotIterator>*>(msg.result_ptr);
                    auto v_snap = std::static_pointer_cast<VectorSnapshot>(*snap);
                    (void)runtime_->NewIterator(v_snap, out);
                }
                break;
            }
            default:
                SetStatus(msg.status_ptr, Status::InvalidArgument("vector pod unsupported opcode"));
                break;
        }
    }

} // namespace pomai::core
