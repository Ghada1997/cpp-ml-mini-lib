#include "ml/logistic_regression.hpp"

#include <cmath>
#include <stdexcept>

namespace ml {

// Numerically stable sigmoid
double LogisticRegression::sigmoid(double z) {
    if (z >= 0) {
        double e = std::exp(-z);
        return 1.0 / (1.0 + e);
    } else {
        double e = std::exp(z);
        return e / (1.0 + e);
    }
}

LogisticRegression::LogisticRegression(double lr, int epochs)
    : lr_(lr), epochs_(epochs), bias_(0.0) {}

void LogisticRegression::fit(const std::vector<std::vector<double>>& X,
                             const std::vector<int>& y) {
    if (X.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid training data.");
    }

    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    // Validate dimensions
    for (const auto& row : X) {
        if (row.size() != n_features) {
            throw std::invalid_argument("Inconsistent feature dimensions in LogisticRegression.");
        }
    }

    weights_.assign(n_features, 0.0);
    bias_ = 0.0;

    for (int epoch = 0; epoch < epochs_; ++epoch) {
        std::vector<double> grad_w(n_features, 0.0);
        double grad_b = 0.0;

        for (size_t i = 0; i < n_samples; ++i) {
            double linear = bias_;
            for (size_t j = 0; j < n_features; ++j) {
                linear += weights_[j] * X[i][j];
            }

            double y_pred = sigmoid(linear);
            double error = y_pred - static_cast<double>(y[i]);

            for (size_t j = 0; j < n_features; ++j) {
                grad_w[j] += error * X[i][j];
            }
            grad_b += error;
        }

        for (size_t j = 0; j < n_features; ++j) {
            weights_[j] -= lr_ * grad_w[j] / n_samples;
        }
        bias_ -= lr_ * grad_b / n_samples;
    }
}

std::vector<double> LogisticRegression::predict_proba(
    const std::vector<std::vector<double>>& X) const {

    if (X.empty()) return {};

    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    if (weights_.size() != n_features) {
        throw std::invalid_argument("Feature dimension mismatch.");
    }

    std::vector<double> probs(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        double linear = bias_;
        for (size_t j = 0; j < n_features; ++j) {
            linear += weights_[j] * X[i][j];
        }
        probs[i] = sigmoid(linear);
    }

    return probs;
}

std::vector<int> LogisticRegression::predict(
    const std::vector<std::vector<double>>& X) const {

    auto probs = predict_proba(X);
    std::vector<int> labels(probs.size());

    for (size_t i = 0; i < probs.size(); ++i) {
        labels[i] = (probs[i] >= 0.5 ? 1 : 0);
    }

    return labels;
}

} // namespace ml
