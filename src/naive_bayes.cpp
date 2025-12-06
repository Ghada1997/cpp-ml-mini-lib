#include "ml/naive_bayes.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace ml {

MultinomialNaiveBayes::MultinomialNaiveBayes(double alpha)
    : alpha_(alpha) {}

void MultinomialNaiveBayes::fit(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {

    if (X.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid training data for Naive Bayes.");
    }

    size_t n_samples = X.size();
    n_features_ = static_cast<int>(X[0].size());

    // Dimensionality check
    for (const auto& row : X) {
        if (row.size() != static_cast<size_t>(n_features_)) {
            throw std::invalid_argument("Inconsistent feature dimensions in Naive Bayes.");
        }
    }

    // Class indexing
    std::unordered_map<int, int> label_to_idx;
    for (int label : y) {
        if (label_to_idx.find(label) == label_to_idx.end()) {
            label_to_idx[label] = static_cast<int>(label_to_idx.size());
        }
    }

    n_classes_ = static_cast<int>(label_to_idx.size());

    // Reverse mapping
    index_to_label_.resize(n_classes_);
    for (auto& [label, idx] : label_to_idx) {
        index_to_label_[idx] = label;
    }

    // Convert labels
    std::vector<int> y_idx(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        y_idx[i] = label_to_idx[y[i]];
    }

    // Count samples per class
    std::vector<int> class_count(n_classes_, 0);
    for (int c : y_idx) class_count[c]++;

    // Log priors
    class_log_prior_.assign(n_classes_, 0.0);
    for (int c = 0; c < n_classes_; ++c) {
        class_log_prior_[c] = std::log(
            static_cast<double>(class_count[c]) / n_samples);
    }

    // Feature counts
    feature_log_prob_.assign(
        n_classes_, std::vector<double>(n_features_, 0.0));

    std::vector<double> class_feature_sum(n_classes_, 0.0);

    for (size_t i = 0; i < n_samples; ++i) {
        int c = y_idx[i];
        for (int f = 0; f < n_features_; ++f) {
            double val = X[i][f];
            if (val < 0.0) {
                throw std::invalid_argument("Features must be non-negative.");
            }
            feature_log_prob_[c][f] += val;
            class_feature_sum[c] += val;
        }
    }

    // Convert to log probabilities with Laplace smoothing
    for (int c = 0; c < n_classes_; ++c) {
        double denom = class_feature_sum[c] + alpha_ * n_features_;

        for (int f = 0; f < n_features_; ++f) {
            double num = feature_log_prob_[c][f] + alpha_;
            feature_log_prob_[c][f] = std::log(num / denom);
        }
    }
}

std::vector<int> MultinomialNaiveBayes::predict(
    const std::vector<std::vector<double>>& X) const {

    if (X.empty()) return {};

    size_t n_samples = X.size();

    for (const auto& row : X) {
        if (row.size() != static_cast<size_t>(n_features_)) {
            throw std::invalid_argument("Feature dimension mismatch in NB predict.");
        }
    }

    std::vector<int> preds(n_samples);

    for (size_t i = 0; i < n_samples; ++i) {
        double best_log_prob = -1e30;
        int best_c = 0;

        for (int c = 0; c < n_classes_; ++c) {
            double logp = class_log_prior_[c];

            for (int f = 0; f < n_features_; ++f) {
                if (X[i][f] > 0.0) {
                    logp += X[i][f] * feature_log_prob_[c][f];
                }
            }

            if (logp > best_log_prob) {
                best_log_prob = logp;
                best_c = c;
            }
        }

        preds[i] = index_to_label_[best_c];  // restore original label
    }

    return preds;
}

} // namespace ml
