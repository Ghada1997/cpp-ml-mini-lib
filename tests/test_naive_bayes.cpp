#include "ml/naive_bayes.hpp"

#include <gtest/gtest.h>

TEST(NaiveBayesTest, SimpleTwoClassCounts) {
    // Very simple dataset: 2 features, 2 classes
    // Class 0: high in feature 0
    // Class 1: high in feature 1
    std::vector<std::vector<double>> X = {
        {5.0, 1.0}, // class 0
        {6.0, 1.0}, // class 0
        {1.0, 5.0}, // class 1
        {1.0, 6.0}  // class 1
    };
    std::vector<int> y = {0, 0, 1, 1};

    ml::MultinomialNaiveBayes nb(1.0);
    nb.fit(X, y);
    auto preds = nb.predict(X);

    int correct = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (preds[i] == y[i]) correct++;
    }
    double acc = static_cast<double>(correct) / static_cast<double>(y.size());
    EXPECT_GE(acc, 0.75);
}
