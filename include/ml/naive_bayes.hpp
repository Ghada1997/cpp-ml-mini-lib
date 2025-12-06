#pragma once

#include <vector>

namespace ml {

/**
 * Brief: Multinomial Naive Bayes classifier (discrete features) (discrete feature counts / frequencies)
 */

class MultinomialNaiveBayes {
public:
    explicit MultinomialNaiveBayes(double alpha = 1.0); // Laplace smoothing

    // X: samples x features (non-negative)
    // y: labels in {0, 1, 2, ...}
    
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y);

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

private:
    double alpha_;

    int n_classes_ = 0;
    int n_features_ = 0;

    std::vector<double> class_log_prior_;           // size: n_classes
    std::vector<std::vector<double>> feature_log_prob_; // [class][feature]
    
    std::vector<int> index_to_label_;
};

} // namespace ml
