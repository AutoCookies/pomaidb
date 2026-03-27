#pragma once

#include <memory>
#include <string>
#include <vector>
#include <span>
#include <variant>

#include "pomai/options.h"
#include "pomai/status.h"

namespace pomai::core {
enum class ModelType {
    kGGUF,       // Large Language Models / SLMs
    kTensor,     // ML models (TFLite)
    kAnalytical, // "No-Train" deterministic models (ELM, RKS)
};

/**
 * AIEngine provides a unified, edge-optimized interface for local AI inference.
 * It abstracts over GGUF (cheesebrain) and TFLite (tensor_core) backends.
 */
class AIEngine {
public:
    enum class RuntimeState {
        kUninitialized,
        kLoaded,
        kReady,
        kRunning,
        kCompleted,
        kError,
    };

    struct InferenceSummary {
        float score = 0.0f;              // 0..1 normalized confidence/severity
        bool action_required = false;    // deterministic rule trigger
        std::string label;               // short classification
        std::string explanation;         // operator-readable reason
    };

    AIEngine();
    ~AIEngine();

    // Disable copy for safety on memory-constrained edge devices.
    AIEngine(const AIEngine&) = delete;
    AIEngine& operator=(const AIEngine&) = delete;

    /** Load a model into the specific backend. */
    Status LoadModel(const std::string& path, ModelType type);

    /** 
     * Perform a single inference slice. 
     * Returns true if inference is complete, false if more slices are needed (re-queue).
     * This ensures the AI doesn't block the MicroKernel/TaskScheduler.
     */
    bool StepInference(float* progress = nullptr);

    /** Set input tensor data. Zero-copy where possible. */
    void SetInput(std::span<const float> data);

    /** Get output. Result index depends on the model. */
    std::span<const float> GetOutput(int index = 0) const;

    /** For GGUF: Get generated text chunk. */
    std::string GetTextResult() const;

    /**
     * No-training inference path for any supported membrane datatype.
     * Caller provides compact numeric features extracted from domain payload.
     */
    Status InferNoTrain(MembraneKind kind, std::span<const float> features, InferenceSummary* out) const;

    /** Runtime diagnostics for production control planes. */
    RuntimeState State() const;
    Status LastStatus() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pomai::core
