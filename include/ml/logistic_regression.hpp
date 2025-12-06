#pragma once

#include <vector>

namespace ml {

/**
 * Brief: Binary logistic regression classifier using gradient descent.
 */

class LogisticRegression {
public:
    LogisticRegression(double lr = 0.1, int epochs = 1000);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y);

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    std::vector<double> predict_proba(const std::vector<std::vector<double>>& X) const;

    const std::vector<double>& weights() const { return weights_; }
    double bias() const { return bias_; }

private:
    double lr_;
    int epochs_;
    std::vector<double> weights_;
    double bias_ = 0.0; // initialize bias

    static double sigmoid(double z);
};

} // namespace ml
