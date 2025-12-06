#include <iostream>
#include <vector>
#include "ml/kmeans.hpp"
#include "ml/dataset.hpp"
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: run_kmeans <path_to_csv> <n_clusters>\n";
        return 1;
    }

    std::string path = argv[1];
    int n_clusters = std::stoi(argv[2]);

    try {
        ml::Dataset data = ml::CSVLoader::load_unsupervised(path);
        const auto& X = data.X;
        if (X.empty()) {
            std::cerr << "Empty dataset.\n";
            return 1;
        }

        ml::KMeans kmeans(n_clusters);
        kmeans.fit(X);

        const auto& labels = kmeans.labels();
        std::cout << "Cluster assignments:\n";
        for (size_t i = 0; i < labels.size(); ++i) {
            std::cout << "Sample " << i << ": cluster " << labels[i] << "\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
