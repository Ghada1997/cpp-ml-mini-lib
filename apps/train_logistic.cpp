#include "ml/dataset.hpp"
#include "ml/logistic_regression.hpp"
#include "ml/metrics.hpp"

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: train_logistic <path_to_supervised_csv>\n";
        return 1;
    }

    std::string path = argv[1];

    try {
        auto dataset = ml::CSVLoader::load_supervised(path);
        if (dataset.X.empty()) {
            std::cerr << "Empty dataset.\n";
            return 1;
        }

        ml::LogisticRegression model(0.1, 1000);
        model.fit(dataset.X, dataset.y);

        auto preds = model.predict(dataset.X);
        double acc = ml::metrics::accuracy(dataset.y, preds);
        double f1  = ml::metrics::f1_score(dataset.y, preds);

        std::cout << "Training accuracy (Logistic): " << acc << "\n";
        std::cout << "Training F1 (Logistic): " << f1 << "\n";

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
