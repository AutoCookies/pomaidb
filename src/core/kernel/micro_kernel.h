#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/kernel/message.h"
#include "core/kernel/pod.h"
#include "pomai/status.h"
#include "ai/analytical_engine.h"
#include "core/metrics/metrics_registry.h"
#include "core/util/ring_buffer.h"

namespace pomai::core {

    /**
     * Pomegranate MicroKernel: The central coordinator for Pods.
     * Manages sequential execution of tasks via an in-memory message queue.
     * Optimized for Edge: Zero-Lock (Single-Threaded) and Static Memory (Zero-Allocation).
     */
    class MicroKernel {
    public:
        MicroKernel() = default;
        ~MicroKernel() { Stop(); }

        // Non-copyable
        MicroKernel(const MicroKernel&) = delete;
        MicroKernel& operator=(const MicroKernel&) = delete;

        /** Register a service pod. kernel takes ownership. */
        Status RegisterPod(std::unique_ptr<Pod> pod) {
            if (!pod) return Status::InvalidArgument("pod is null");
            PodId id = pod->Id();
            if (pods_.count(id)) return Status::AlreadyExists("pod already registered");
            
            pod->OnStart();
            pods_[id] = std::move(pod);
            return Status::Ok();
        }

        /** Unregister and stop a pod. */
        void UnregisterPod(PodId id) {
            auto it = pods_.find(id);
            if (it != pods_.end()) {
                it->second->OnStop();
                pods_.erase(it);
            }
        }

        /** Post a message for later execution. */
        void Enqueue(Message&& msg) {
            metrics::MetricsRegistry::Instance().Increment("kernel_messages_enqueued");
            
            // Apply ELM-based Backpressure
            auto* elm = AnalyticalEngine::Global().GetModel("kernel_pressure");
            if (!elm) {
                // Lazily Train strict mathematical relationship
                (void)AnalyticalEngine::Global().CreateELMModel("kernel_pressure", 2, 8, 1);
                elm = AnalyticalEngine::Global().GetModel("kernel_pressure");
                if (elm) {
                    float X[8] = {0.0f, 0.0f, 10.0f, 100.0f, 100.0f, 1000.0f, 200.0f, 5000.0f};
                    float Y[4] = {0.0f, 60.0f, 600.0f, 1500.0f};
                    (void)elm->Train(std::span<const float>(X, 8), std::span<const float>(Y, 4), 4);
                }
            }

            if (elm && elm->InputDim() == 2 && elm->OutputDim() >= 1) {
                float x[2] = { static_cast<float>(queue_.size()), static_cast<float>(msg.payload.size()) };
                float pred[1] = { 0.0f };
                elm->Predict(std::span<const float>(x, 2), std::span<float>(pred, 1));
                if (pred[0] > 1000.0f) { 
                    metrics::MetricsRegistry::Instance().Increment("kernel_load_shed");
                    if (msg.result_ptr) {
                        *static_cast<Status*>(msg.result_ptr) = Status::ResourceExhausted("ELM predicted system timeout under load");
                    }
                    return; 
                }
            }
            
            if (!queue_.push_back(std::move(msg))) {
                metrics::MetricsRegistry::Instance().Increment("kernel_queue_overflow");
                if (msg.result_ptr) {
                    *static_cast<Status*>(msg.result_ptr) = Status::ResourceExhausted("Kernel message queue overflow");
                }
            }
        }

        /** Synchronously execute one message from the queue. */
        bool DispatchOne() {
            if (queue_.empty()) return false;
            
            // Pop first to prevent infinite recursion in re-entrant ProcessAll calls
            std::optional<Message> msg_opt = queue_.pop_front();
            if (!msg_opt) return false;
            
            Message msg = std::move(*msg_opt);
            auto it = pods_.find(msg.target);
            if (it != pods_.end()) {
                metrics::MetricsRegistry::Instance().Increment("kernel_messages_dispatched");
                
                // Tracing for observability
                if (msg.trace.enabled) {
                    msg.trace.hop_count++;
                }

                it->second->Handle(std::move(msg));
            }
            
            return true;
        }

        /** Drain the entire queue. */
        void ProcessAll() {
            while (DispatchOne());
        }

        /** Shutdown all pods. */
        void Stop() {
            for (auto& kv : pods_) {
                kv.second->OnStop();
            }
            pods_.clear();
            queue_.clear();
        }

        /** Direct access to a pod (use sparingly). */
        Pod* GetPod(PodId id) {
            auto it = pods_.find(id);
            return (it != pods_.end()) ? it->second.get() : nullptr;
        }

    private:
        std::unordered_map<PodId, std::unique_ptr<Pod>> pods_;
        util::StaticRingBuffer<Message, 1024> queue_;
    };

} // namespace pomai::core
