#include "ai/analytical_engine.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <mutex>

namespace pomai::core {

// Simple matrix inversion using Gaussian Elimination for square matrices.
bool InvertMatrix(std::vector<float>& mat, size_t n) {
    std::vector<float> inv(n * n, 0.0f);
    for (size_t i = 0; i < n; ++i) inv[i * n + i] = 1.0f;

    for (size_t i = 0; i < n; ++i) {
        float pivot = mat[i * n + i];
        if (std::abs(pivot) < 1e-9f) return false; // Singular

        for (size_t j = 0; j < n; ++j) {
            mat[i * n + j] /= pivot;
            inv[i * n + j] /= pivot;
        }

        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                float factor = mat[k * n + i];
                for (size_t j = 0; j < n; ++j) {
                    mat[k * n + j] -= factor * mat[i * n + j];
                    inv[k * n + j] -= factor * inv[i * n + j];
                }
            }
        }
    }
    mat = std::move(inv);
    return true;
}

ELMModel::ELMModel(size_t input_dim, size_t hidden_dim, size_t output_dim, float l2_reg)
    : input_dim_(input_dim), hidden_dim_(hidden_dim), output_dim_(output_dim), l2_reg_(l2_reg) {
    input_weights_.resize(hidden_dim * input_dim);
    input_bias_.resize(hidden_dim);
    output_weights_.resize(output_dim * hidden_dim);
    InitializeWeights();
}

ELMModel::~ELMModel() = default;

void ELMModel::InitializeWeights() {
    std::mt19937 gen(42); // Deterministic seed
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& w : input_weights_) w = dist(gen);
    for (auto& b : input_bias_) b = dist(gen);
}

void ELMModel::MapToHidden(std::span<const float> input, std::span<float> hidden) const {
    for (size_t i = 0; i < hidden_dim_; ++i) {
        float sum = input_bias_[i];
        for (size_t j = 0; j < input_dim_; ++j) {
            sum += input_weights_[i * input_dim_ + j] * input[j];
        }
        hidden[i] = 1.0f / (1.0f + std::exp(-sum)); // Sigmoid activation
    }
}

Status ELMModel::Train(std::span<const float> inputs, 
                      std::span<const float> targets,
                      size_t num_samples) {
    if (num_samples == 0) return Status::InvalidArgument("No samples for training");

    // 1. Calculate H matrix [num_samples, hidden_dim]
    std::vector<float> H(num_samples * hidden_dim_);
    for (size_t s = 0; s < num_samples; ++s) {
        MapToHidden(inputs.subspan(s * input_dim_, input_dim_), 
                    std::span<float>(H.data() + s * hidden_dim_, hidden_dim_));
    }

    // 2. Solve W_out = T H^T (H H^T + lambda I)^-1
    // Actually, if num_samples >> hidden_dim, it's easier to solve (H^T H + lambda I) W = H^T T
    // H^T H has size [hidden_dim, hidden_dim]
    std::vector<float> HTH(hidden_dim_ * hidden_dim_, 0.0f);
    for (size_t i = 0; i < hidden_dim_; ++i) {
        for (size_t j = 0; j < hidden_dim_; ++j) {
            float sum = 0.0f;
            for (size_t s = 0; s < num_samples; ++s) {
                sum += H[s * hidden_dim_ + i] * H[s * hidden_dim_ + j];
            }
            HTH[i * hidden_dim_ + j] = sum + (i == j ? l2_reg_ : 0.0f);
        }
    }

    if (!InvertMatrix(HTH, hidden_dim_)) {
        return Status::Internal("Analytical solver failed: Singular matrix");
    }

    // W_out = (T H^T) * (HTH_inv)
    // T H^T has size [output_dim, hidden_dim]
    std::vector<float> THT(output_dim_ * hidden_dim_, 0.0f);
    for (size_t o = 0; o < output_dim_; ++o) {
        for (size_t h = 0; h < hidden_dim_; ++h) {
            float sum = 0.0f;
            for (size_t s = 0; s < num_samples; ++s) {
                sum += targets[s * output_dim_ + o] * H[s * hidden_dim_ + h];
            }
            THT[o * hidden_dim_ + h] = sum;
        }
    }

    // Final result output_weights_ = THT * HTH_inv
    for (size_t o = 0; o < output_dim_; ++o) {
        for (size_t h1 = 0; h1 < hidden_dim_; ++h1) {
            float sum = 0.0f;
            for (size_t h2 = 0; h2 < hidden_dim_; ++h2) {
                sum += THT[o * hidden_dim_ + h2] * HTH[h2 * hidden_dim_ + h1];
            }
            output_weights_[o * hidden_dim_ + h1] = sum;
        }
    }

    return Status::Ok();
}

void ELMModel::Predict(std::span<const float> input, 
                      std::span<float> output) const {
    std::vector<float> hidden(hidden_dim_);
    MapToHidden(input, hidden);
    for (size_t o = 0; o < output_dim_; ++o) {
        float sum = 0.0f;
        for (size_t h = 0; h < hidden_dim_; ++h) {
            sum += output_weights_[o * hidden_dim_ + h] * hidden[h];
        }
        output[o] = sum;
    }
}

struct AnalyticalEngine::Impl {
    std::map<std::string, std::unique_ptr<AnalyticalModel>> models;
};

AnalyticalEngine::AnalyticalEngine() : impl_(std::make_unique<Impl>()) {}
AnalyticalEngine::~AnalyticalEngine() = default;

AnalyticalEngine& AnalyticalEngine::Global() {
    static AnalyticalEngine instance;
    return instance;
}

Status AnalyticalEngine::CreateELMModel(const std::string& name, size_t in_dim, size_t hidden_dim, size_t out_dim) {
    impl_->models[name] = std::make_unique<ELMModel>(in_dim, hidden_dim, out_dim);
    return Status::Ok();
}

AnalyticalModel* AnalyticalEngine::GetModel(const std::string& name) {
    auto it = impl_->models.find(name);
    return it != impl_->models.end() ? it->second.get() : nullptr;
}

} // namespace pomai::core
