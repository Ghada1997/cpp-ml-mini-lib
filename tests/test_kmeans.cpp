#include "ml/kmeans.hpp"

#include <gtest/gtest.h>

TEST(KMeansTest, SimpleClusterablePoints) {
    // Two obvious clusters around (0,0) and (10,10)
    std::vector<std::vector<double>> X = {
        {0.0, 0.0},
        {0.5, -0.2},
        {-0.3, 0.3},
        {10.0, 10.0},
        {9.5, 10.1},
        {10.2, 9.7}
    };

    ml::KMeans kmeans(2, 50);
    kmeans.fit(X);

    auto labels = kmeans.labels();
    ASSERT_EQ(labels.size(), X.size());

    // Just sanity: we expect at least one sample of each cluster
    bool has0 = false, has1 = false;
    for (int lab : labels) {
        if (lab == 0) has0 = true;
        if (lab == 1) has1 = true;
    }
    EXPECT_TRUE(has0 && has1);
}
