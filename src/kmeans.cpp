#include "ml/kmeans.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace ml {

KMeans::KMeans(int n_clusters, int max_iters)
    : n_clusters_(n_clusters),
      max_iters_(max_iters) {}

// Compute squared Euclidean distance
double KMeans::squared_distance(const std::vector<double>& a,
                                const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Dimension mismatch in squared_distance.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void KMeans::fit(const std::vector<std::vector<double>>& X) {
    if (X.empty()) {
        throw std::invalid_argument("Empty dataset in KMeans::fit.");
    }

    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    // Validate feature dimensions
    for (const auto& row : X) {
        if (row.size() != n_features) {
            throw std::invalid_argument("Inconsistent feature dimensions in KMeans.");
        }
    }

    if (n_clusters_ <= 0 || n_clusters_ > static_cast<int>(n_samples)) {
        throw std::invalid_argument("Invalid number of clusters in KMeans.");
    }

    centroids_.assign(n_clusters_, std::vector<double>(n_features, 0.0));
    labels_.assign(n_samples, 0);

    // Random centroid initialization
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);
    for (int k = 0; k < n_clusters_; ++k) {
        centroids_[k] = X[dist(gen)];
    }

    // Lloyd's algorithm
    for (int iter = 0; iter < max_iters_; ++iter) {
        bool changed = false;

        // Assignment step
        for (size_t i = 0; i < n_samples; ++i) {
            double best_dist = std::numeric_limits<double>::max();
            int best_k = 0;

            for (int k = 0; k < n_clusters_; ++k) {
                double d = squared_distance(X[i], centroids_[k]);
                if (d < best_dist) {
                    best_dist = d;
                    best_k = k;
                }
            }

            if (labels_[i] != best_k) {
                labels_[i] = best_k;
                changed = true;
            }
        }

        if (!changed && iter > 0)
            break;

        // Update step
        std::vector<std::vector<double>> new_centroids(
            n_clusters_, std::vector<double>(n_features, 0.0));
        std::vector<int> counts(n_clusters_, 0);

        for (size_t i = 0; i < n_samples; ++i) {
            int c = labels_[i];
            counts[c]++;
            for (size_t f = 0; f < n_features; ++f) {
                new_centroids[c][f] += X[i][f];
            }
        }

        // Avoid dead clusters
        for (int k = 0; k < n_clusters_; ++k) {
            if (counts[k] == 0) continue;
            for (size_t f = 0; f < n_features; ++f) {
                new_centroids[k][f] /= static_cast<double>(counts[k]);
            }
        }

        centroids_ = std::move(new_centroids);
    }
}

} // namespace ml
