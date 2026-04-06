#pragma once

#include "core/kernel/pod.h"
#include "core/graph/graph_membrane_impl.h"

namespace pomai::core {

    /**
     * GraphPod: Wraps GraphMembraneImpl to provide Pod capabilities.
     */
    class GraphPod : public Pod {
    public:
        explicit GraphPod(std::unique_ptr<GraphMembraneImpl> runtime)
            : runtime_(std::move(runtime)) {}

        void Handle(Message&& msg) override;

        PodId Id() const override { return PodId::kGraph; }
        std::string Name() const override { return "GraphService"; }

        MemoryQuota GetQuota() const override {
            return {runtime_->MemoryBytesUsed(), 0};
        }

        void OnStart() override { (void)runtime_->WarmUp(); }
        void OnStop() override { (void)runtime_->Flush(); }

    private:
        std::unique_ptr<GraphMembraneImpl> runtime_;
    };

} // namespace pomai::core
