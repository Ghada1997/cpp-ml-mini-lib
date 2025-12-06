#include "ml/logistic_regression.hpp"
#include "ml/metrics.hpp"

#include <gtest/gtest.h>

TEST(LogisticRegressionTest, SimpleSeparableData) {
    std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<int> y = {0, 0, 0, 1};

    ml::LogisticRegression model(0.5, 2000);
    model.fit(X, y);

    auto preds = model.predict(X);
    double acc = ml::metrics::accuracy(y, preds);

    EXPECT_GE(acc, 0.75);
}
