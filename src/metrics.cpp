#include "ml/metrics.hpp"

#include <stdexcept>

namespace ml::metrics {

double accuracy(const std::vector<int>& y_true,
                const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) {
        throw std::invalid_argument("Invalid input to accuracy.");
    }

    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) correct++;
    }

    return static_cast<double>(correct) / y_true.size();
}

double f1_score(const std::vector<int>& y_true,
                const std::vector<int>& y_pred) {

    if (y_true.size() != y_pred.size() || y_true.empty()) {
        throw std::invalid_argument("Invalid input to f1_score.");
    }

    int tp = 0, fp = 0, fn = 0;

    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 1 && y_pred[i] == 1) tp++;
        else if (y_pred[i] == 1 && y_true[i] == 0) fp++;
        else if (y_pred[i] == 0 && y_true[i] == 1) fn++;
    }

    double precision = (tp + fp == 0 ? 0.0 : double(tp) / (tp + fp));
    double recall    = (tp + fn == 0 ? 0.0 : double(tp) / (tp + fn));

    if (precision + recall == 0.0) return 0.0;

    return 2.0 * precision * recall / (precision + recall);
}

} // namespace ml::metrics
