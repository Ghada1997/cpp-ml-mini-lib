#pragma once

#include <vector>

namespace ml {

/**
 * Brief: Basic implementation of K-Means clustering.
 */

class KMeans {
public:
    KMeans(int n_clusters, int max_iters = 300);

    /**
     * Brief: Fit the K-Means model to data.
     */

    void fit(const std::vector<std::vector<double>>& X);

    /**
     * return: The cluster index for each sample.
     */

    const std::vector<int>& labels() const { return labels_; }

    
    /**
     * return: The centroid for each cluster.
     */

    const std::vector<std::vector<double>>& centroids() const {
        return centroids_;
    }

private:
    int n_clusters_;
    int max_iters_;

    std::vector<std::vector<double>> centroids_;
    std::vector<int> labels_;

    static double squared_distance(const std::vector<double>& a,
                                   const std::vector<double>& b);
};

} // namespace ml
