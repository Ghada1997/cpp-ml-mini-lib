#include "ml/dataset.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace ml {

// Helper: trim whitespace
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    auto end   = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos ? "" : s.substr(start, end - start + 1));
}

Dataset CSVLoader::load_supervised(const std::string& path, char delimiter) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    Dataset dataset;
    std::string line;
    bool header_skipped = false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Optional: skip header if non-numeric
        if (!header_skipped) {
            std::stringstream temp(line);
            std::string token;
            std::getline(temp, token, delimiter);

            try { std::stod(trim(token)); }
            catch (...) { 
                header_skipped = true; 
                continue; 
            }
        }

        std::stringstream ss(line);
        std::string cell;

        std::vector<std::string> tokens;
        while (std::getline(ss, cell, delimiter)) {
            tokens.push_back(trim(cell));  // keep empty columns
        }

        if (tokens.size() < 2) continue;

        std::vector<double> features;
        features.reserve(tokens.size() - 1);

        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            if (tokens[i].empty())
                features.push_back(0.0);
            else
                features.push_back(std::stod(tokens[i]));
        }

        int label = std::stoi(tokens.back());

        dataset.X.push_back(std::move(features));
        dataset.y.push_back(label);
    }

    return dataset;
}

Dataset CSVLoader::load_unsupervised(const std::string& path, char delimiter) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    Dataset dataset;
    std::string line;
    bool header_skipped = false;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Skip header if first line contains text
        if (!header_skipped) {
            std::stringstream temp(line);
            std::string token;
            std::getline(temp, token, delimiter);

            try { std::stod(trim(token)); }
            catch (...) { 
                header_skipped = true; 
                continue; 
            }
        }

        std::stringstream ss(line);
        std::string cell;

        std::vector<double> features;
        while (std::getline(ss, cell, delimiter)) {
            cell = trim(cell);
            features.push_back(cell.empty() ? 0.0 : std::stod(cell));
        }

        if (!features.empty())
            dataset.X.push_back(std::move(features));
    }

    return dataset;
}

} // namespace ml
