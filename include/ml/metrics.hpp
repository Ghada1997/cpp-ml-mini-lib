#pragma once

#include <vector>

namespace ml::metrics {

/**
 * Brief: Compute classification accuracy.
 */

double accuracy(const std::vector<int>& y_true,
                const std::vector<int>& y_pred);

                
/**
 * Brief: Compute macro F1-score.
 */

double f1_score(const std::vector<int>& y_true,
                const std::vector<int>& y_pred);

} // namespace ml::metrics
