#include "ai/ai_engine.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <filesystem>

// GGUF (cheesebrain) headers
#include "ai/cheesebrain_core/cheese-context.h"
#include "ai/cheesebrain_core/cheese-model-loader.h"

// TFLite headers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "ai/analytical_engine.h"
#include "core/ai/no_train_dispatch.h"
#include <numeric>

namespace tflite { namespace ops { namespace builtin {
TfLiteRegistration* Register_ADD();
TfLiteRegistration* Register_MUL();
TfLiteRegistration* Register_SUB();
TfLiteRegistration* Register_RESHAPE();
}}}

namespace pomai::core {

struct AIEngine::Impl {
    ModelType type;
    RuntimeState state = RuntimeState::kUninitialized;
    mutable Status last_status = Status::Ok();
    
    // GGUF state
    std::unique_ptr<cheese_context> gguf_ctx;
    
    // TFLite state
    std::unique_ptr<tflite::Interpreter> tflite_interpreter;
    std::unique_ptr<tflite::FlatBufferModel> tflite_model;

    // Analytical Backend
    std::unique_ptr<AnalyticalEngine> analytical_engine;
    std::string text_result;

    Impl() : type(ModelType::kGGUF) {}

    void Reset() {
        gguf_ctx.reset();
        tflite_interpreter.reset();
        tflite_model.reset();
        analytical_engine.reset();
        text_result.clear();
        state = RuntimeState::kUninitialized;
        last_status = Status::Ok();
    }
};

AIEngine::AIEngine() : impl_(std::make_unique<Impl>()) {}
AIEngine::~AIEngine() = default;

Status AIEngine::LoadModel(const std::string& path, ModelType type) {
    impl_->Reset();
    impl_->type = type;
    impl_->state = RuntimeState::kLoaded;

    if (type == ModelType::kAnalytical) {
        impl_->analytical_engine = std::make_unique<AnalyticalEngine>();
        impl_->state = RuntimeState::kReady;
        impl_->last_status = Status::Ok();
        return Status::Ok();
    }

    if (path.empty()) {
        impl_->state = RuntimeState::kError;
        impl_->last_status = Status::InvalidArgument("model path is empty");
        return impl_->last_status;
    }
    if (!std::filesystem::exists(path)) {
        impl_->state = RuntimeState::kError;
        impl_->last_status = Status::NotFound("model file not found");
        return impl_->last_status;
    }

    if (type == ModelType::kGGUF) {
        try {
            // Keep path validation strict; mark GGUF as explicit not-supported
            // until streaming context wiring is fully enabled.
            cheese_model_params mparams = cheese_model_default_params();
            (void)mparams;
            impl_->state = RuntimeState::kError;
            impl_->last_status = Status::NotSupported("GGUF runtime wiring is not enabled yet");
            return impl_->last_status;
        } catch (const std::exception& e) {
            impl_->state = RuntimeState::kError;
            impl_->last_status = Status::IoError(e.what());
            return impl_->last_status;
        }
    } else {
        // TFLite loading
        impl_->tflite_model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
        if (!impl_->tflite_model) {
            impl_->state = RuntimeState::kError;
            impl_->last_status = Status::IoError("Failed to load TFLite model from " + path);
            return impl_->last_status;
        }

        tflite::MutableOpResolver resolver;
        // Register essential operations for custom models using the builtin op registration functions.
        // These are defined in the surgically restored kernel sources.
        resolver.AddBuiltin(tflite::BuiltinOperator_ADD, tflite::ops::builtin::Register_ADD());
        resolver.AddBuiltin(tflite::BuiltinOperator_MUL, tflite::ops::builtin::Register_MUL());
        resolver.AddBuiltin(tflite::BuiltinOperator_SUB, tflite::ops::builtin::Register_SUB());
        resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, tflite::ops::builtin::Register_RESHAPE());
        
        tflite::InterpreterBuilder builder(*impl_->tflite_model, resolver);
        if (builder(&impl_->tflite_interpreter) != kTfLiteOk) {
            impl_->state = RuntimeState::kError;
            impl_->last_status = Status::Internal("Failed to build TFLite interpreter");
            return impl_->last_status;
        }

        if (impl_->tflite_interpreter->AllocateTensors() != kTfLiteOk) {
            impl_->state = RuntimeState::kError;
            impl_->last_status = Status::Internal("Failed to allocate TFLite tensors");
            return impl_->last_status;
        }
        impl_->state = RuntimeState::kReady;
        impl_->last_status = Status::Ok();
        return Status::Ok();
    }
}

bool AIEngine::StepInference(float* progress) {
    if (progress) *progress = 0.0f;
    if (impl_->state != RuntimeState::kReady && impl_->state != RuntimeState::kRunning) {
        impl_->last_status = Status(ErrorCode::kFailedPrecondition, "model is not ready");
        return false;
    }
    impl_->state = RuntimeState::kRunning;

    if (impl_->type == ModelType::kTensor && impl_->tflite_interpreter) {
        if (impl_->tflite_interpreter->Invoke() == kTfLiteOk) {
            if (progress) *progress = 1.0f;
            impl_->state = RuntimeState::kCompleted;
            impl_->last_status = Status::Ok();
            return true;
        }
        impl_->state = RuntimeState::kError;
        impl_->last_status = Status::Internal("TFLite invoke failed");
        return false;
    }

    if (impl_->type == ModelType::kAnalytical && impl_->analytical_engine) {
        if (progress) *progress = 1.0f;
        impl_->state = RuntimeState::kCompleted;
        impl_->last_status = Status::Ok();
        return true;
    }

    impl_->state = RuntimeState::kError;
    impl_->last_status = Status::NotSupported("backend does not support step inference");
    return false;
}

void AIEngine::SetInput(std::span<const float> data) {
    if (impl_->type != ModelType::kTensor || !impl_->tflite_interpreter) {
        impl_->last_status = Status(ErrorCode::kFailedPrecondition, "tensor backend is not ready");
        return;
    }
    const TfLiteTensor* tensor = impl_->tflite_interpreter->input_tensor(0);
    if (!tensor) {
        impl_->last_status = Status::Internal("missing input tensor");
        return;
    }
    const size_t expected = tensor->bytes / sizeof(float);
    if (expected == 0 || data.size() != expected) {
        impl_->last_status = Status::InvalidArgument("input size mismatch");
        return;
    }
    float* input = impl_->tflite_interpreter->typed_input_tensor<float>(0);
    if (!input) {
        impl_->last_status = Status::Internal("input tensor type is not float");
        return;
    }
    std::copy(data.begin(), data.end(), input);
    impl_->last_status = Status::Ok();
}

std::span<const float> AIEngine::GetOutput(int index) const {
    if (index < 0) {
        impl_->last_status = Status::InvalidArgument("output index must be non-negative");
        return {};
    }
    if (impl_->type == ModelType::kTensor && impl_->tflite_interpreter) {
        if (static_cast<size_t>(index) >= impl_->tflite_interpreter->outputs().size()) {
            impl_->last_status = Status::InvalidArgument("output index out of range");
            return {};
        }
        const TfLiteTensor* out_tensor = impl_->tflite_interpreter->output_tensor(index);
        if (!out_tensor) {
            impl_->last_status = Status::Internal("missing output tensor");
            return {};
        }
        const float* output = impl_->tflite_interpreter->typed_output_tensor<float>(index);
        if (!output) {
            impl_->last_status = Status::Internal("output tensor type is not float");
            return {};
        }
        size_t size = out_tensor->bytes / sizeof(float);
        impl_->last_status = Status::Ok();
        return {output, size};
    }
    impl_->last_status = Status(ErrorCode::kFailedPrecondition, "tensor backend is not ready");
    return {};
}

std::string AIEngine::GetTextResult() const {
    return impl_->text_result;
}

Status AIEngine::InferNoTrain(MembraneKind kind, std::span<const float> features, InferenceSummary* out) const {
    if (!out) return Status::InvalidArgument("out is null");
    ai::InferenceSummary summary;
    const Status st = ai::InferNoTrainForKind(kind, features, &summary);
    if (!st.ok()) return st;
    out->score = summary.score;
    out->action_required = summary.action_required;
    out->label = summary.label;
    out->explanation = summary.explanation;
    impl_->last_status = Status::Ok();
    return Status::Ok();
}

AIEngine::RuntimeState AIEngine::State() const { return impl_->state; }

Status AIEngine::LastStatus() const { return impl_->last_status; }

} // namespace pomai::core
