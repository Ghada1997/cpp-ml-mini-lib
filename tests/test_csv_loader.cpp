#include "ml/dataset.hpp"
#include <gtest/gtest.h>
#include <string>

TEST(CSVLoaderTest, LoadSupervisedTinyCSV) {
    std::string path = "../examples/data/logistic_toy.csv";
    
    auto data = ml::CSVLoader::load_supervised(path);

    // At least one sample should be loaded
    EXPECT_FALSE(data.X.empty());

    // X and y should match in length
    EXPECT_EQ(data.X.size(), data.y.size());

    // Features should not be empty
    EXPECT_GT(data.X[0].size(), 0);
}

TEST(CSVLoaderTest, LoadUnsupervisedTinyCSV) {
    std::string path = "../examples/data/kmeans_toy.csv";
    
    auto data = ml::CSVLoader::load_unsupervised(path);

    // Should load at least one row
    EXPECT_FALSE(data.X.empty());

    // Check feature count
    EXPECT_GT(data.X[0].size(), 0);
}
